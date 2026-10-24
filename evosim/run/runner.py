"""Runner: drives a Sim for long unattended runs (and for the UI's per-frame time budget).

Per tick:  schedule -> random events -> policy ramps -> (policies every eval interval) -> sim.step ->
           watchdog / checkpoints / stats / archive / milestones / status.json / stop conditions.
Windows notes: kernels are short (the tick is ~40 launches, never one giant kernel) so TDR (2 s) is not
a risk; SetThreadExecutionState prevents system sleep; console close / Ctrl+C / SIGTERM write a final
checkpoint.
"""
from __future__ import annotations

import ctypes
import json
import os
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from ..config import Config
from ..io import snapshot
from ..sim import LedgerError, Sim
from . import env as envmod
from .archive import GenomeArchive
from .policies import PolicyEngine, StatsView
from .rundir import CheckpointManager, EventLog, Retention, RunDir, StatsWriter, Webhook


# ============================================================================= platform helpers
def prevent_sleep(on: bool):
    """Windows: keep the system awake while a run is active. No-op elsewhere."""
    if sys.platform != "win32":
        return False
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))
        return True
    except Exception:
        return False


_handler_refs = []


def install_signal_handlers(runner):
    """Ctrl+C, SIGTERM, Windows Ctrl+Break and console close -> request a graceful stop (final checkpoint)."""
    def handler(signum, frame):
        runner.request_stop(f"signal {signum}")
    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), handler)
            except (ValueError, OSError):
                pass
    if sys.platform == "win32":
        try:
            HandlerRoutine = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)

            def console(ctrl):
                runner.request_stop(f"console event {ctrl}")
                runner.wait_closed(timeout=4.5)          # Windows kills the process ~5 s after CTRL_CLOSE_EVENT
                return True
            ref = HandlerRoutine(console)
            _handler_refs.append(ref)
            ctypes.windll.kernel32.SetConsoleCtrlHandler(ref, True)
        except Exception:
            pass


# ============================================================================= watchdog
ACTIONS = ("log", "pause", "rollback", "stop")


@dataclass
class WatchdogConfig:
    interval: int = 200
    nan: str = "rollback"
    ledger: str = "rollback"
    tps_drop: str = "log"
    mem_growth: str = "log"
    gpu_error: str = "rollback"
    tps_drop_frac: float = 0.3             # warn when tps < frac * moving average
    mem_growth_frac: float = 0.5
    max_rollbacks_per_hour: int = 5
    stability_tweak: str = "extra_substep"  # none | dt_half | extra_substep


class Watchdog:
    def __init__(self, cfg: WatchdogConfig | None = None):
        self.cfg = cfg or WatchdogConfig()
        self.tps_avg = None
        self.mem0 = None

    def check(self, runner) -> list[tuple[str, str, dict]]:
        """Returns [(check_name, action, details)] for every failed check."""
        sim, c = runner.sim, self.cfg
        problems = []
        bad = sim.has_nan()
        if bad:
            problems.append(("nan", c.nan, {"field": bad}))
        if sim.last_ledger is not None and not sim.last_ledger.get("ok", True):
            problems.append(("ledger", c.ledger, {"errors": sim.last_ledger["errors"]}))
        tps = runner.recent_tps()
        if tps:
            if self.tps_avg is None:
                self.tps_avg = tps
            elif tps < c.tps_drop_frac * self.tps_avg:
                problems.append(("tps_drop", c.tps_drop, {"tps": tps, "average": self.tps_avg}))
            self.tps_avg = 0.95 * self.tps_avg + 0.05 * tps
        try:
            import psutil
            rss = psutil.Process().memory_info().rss
            if self.mem0 is None:
                self.mem0 = rss
            elif rss > self.mem0 * (1 + c.mem_growth_frac) and rss > 2**30:
                problems.append(("mem_growth", c.mem_growth, {"rss_mb": rss / 2**20, "start_mb": self.mem0 / 2**20}))
        except ImportError:
            pass
        return problems


# ============================================================================= milestones
class Milestones:
    def __init__(self, k_states=3, pred_cells=50, extinct_peak=200):
        self.k_states, self.pred_cells, self.extinct_peak = k_states, pred_cells, extinct_peak
        self.best_org = 0
        self.seen_states = False
        self.dominant = None
        self.pred_seen = set()
        self.peak = {}
        self.alive_lineages = set()

    def check(self, sim, emit):
        c = sim.state.cells
        al = np.nonzero(c.alive.numpy() != 0)[0]
        if len(al) == 0:
            return
        org = c.organism_id.numpy()[al]
        lin = c.lineage_id.numpy()[al]
        uo, oc = np.unique(org, return_counts=True)
        if oc.max() > self.best_org:
            self.best_org = int(oc.max())
            emit({"type": "milestone", "kind": "largest_organism", "cells": self.best_org, "tick": sim.tick})
        if not self.seen_states:
            st = c.state.numpy()[al]
            for o in uo[np.argsort(-oc)[:30]]:
                if len(np.unique(st[org == o])) >= self.k_states:
                    self.seen_states = True
                    emit({"type": "milestone", "kind": "organism_distinct_states", "k": self.k_states,
                          "tick": sim.tick})
                    break
        ul, lc = np.unique(lin, return_counts=True)
        dom = int(ul[np.argmax(lc)])
        if dom != self.dominant:
            if self.dominant is not None:
                emit({"type": "milestone", "kind": "dominant_lineage_changed", "old": self.dominant, "new": dom,
                      "tick": sim.tick})
            self.dominant = dom
        pred = c.traits.numpy()[al, 3] > 0.5
        if pred.any():
            pl, pc = np.unique(lin[pred], return_counts=True)
            for l, n in zip(pl, pc):
                if n >= self.pred_cells and int(l) not in self.pred_seen:
                    self.pred_seen.add(int(l))
                    emit({"type": "milestone", "kind": "predatory_lineage", "lineage": int(l), "cells": int(n),
                          "tick": sim.tick})
        now = set(int(x) for x in ul)
        for l, n in zip(ul, lc):
            self.peak[int(l)] = max(self.peak.get(int(l), 0), int(n))
        for l in self.alive_lineages - now:
            if self.peak.get(l, 0) >= self.extinct_peak:
                emit({"type": "milestone", "kind": "lineage_extinct", "lineage": l, "peak": self.peak[l],
                      "tick": sim.tick})
        self.alive_lineages = now


# ============================================================================= runner
class Runner:
    def __init__(self, sim: Sim, rundir: RunDir | None = None, policies: PolicyEngine | None = None,
                 schedule: envmod.Schedule | None = None, events: envmod.EventSystem | None = None,
                 watchdog: Watchdog | None = None, regions: dict | None = None,
                 ckpt_every_ticks=10000, ckpt_every_seconds=0, retention: Retention | None = None,
                 max_dir_bytes: int | None = None, min_free_bytes: int = 500 * 2**20,
                 archive_every=10000, milestone_every=1000, status_every=5.0, frame_every=0,
                 stats_format="parquet", webhook: str | None = None, webhook_severity="error",
                 stop: dict | None = None, hardware_stats=True, keep_sleep_off=False):
        self.sim = sim
        self.rd = rundir
        self.engine = policies or PolicyEngine(eval_interval=10)
        self.schedule = schedule or envmod.Schedule()
        self.events = events or envmod.EventSystem()
        self.watchdog = watchdog or Watchdog()
        self.regions: dict[str, envmod.Region] = regions or {}
        self.elog = EventLog(rundir.events_path if rundir else None)
        if webhook:
            self.elog.listeners.append(Webhook(webhook, webhook_severity))
        self.ckpt = CheckpointManager(rundir, retention, max_dir_bytes, emit=self.emit) if rundir else None
        self.archive = GenomeArchive(rundir.archive_path) if rundir else None
        self.stats = StatsWriter(rundir.stats_dir, fmt=stats_format, hardware=hardware_stats) if rundir else None
        self.milestones = Milestones()
        self.ckpt_every_ticks, self.ckpt_every_seconds = ckpt_every_ticks, ckpt_every_seconds
        self.archive_every, self.milestone_every, self.frame_every = archive_every, milestone_every, frame_every
        self.status_every = status_every
        self.min_free_bytes = min_free_bytes
        self.stop_cfg = stop or {}
        self.pause_requested = None
        self.stop_requested = None
        self.screenshot_requested = False
        self.frame_hook = None                       # callable(sim, path) provided by the UI (offscreen render)
        self.closed = False
        self._closed_evt = False
        self.salt_counter = int(sim.cfg.rng_salt)
        self.rollback_times: list[float] = []
        self.intervention_times: deque = deque(maxlen=10000)
        self.warnings: deque = deque(maxlen=50)
        self.wall0 = time.time()
        self._last_ckpt_tick = sim.tick
        self._last_ckpt_wall = time.time()
        self._last_status = 0.0
        self._tps_samples: deque = deque(maxlen=6)
        self._tps_mark = (time.perf_counter(), sim.tick)
        self._last_stats_tick = -1
        self._extinct_evals = 0
        self.log = rundir.setup_logging() if rundir else None
        if rundir:
            self._init_files()
        self.keep_sleep_off = keep_sleep_off
        for w in sim.cfg.exploit_warnings():
            self.emit({'type': 'config_exploit_warning', 'severity': 'warning', 'message': w})

    # ------------------------------------------------------------------ construction
    @classmethod
    def create(cls, cfg: Config, root="runs", name=None, device=None, policies=None, schedule=None,
               seed_population=True, rundir=None, **kw):
        rd = rundir or RunDir.create(root, name)
        sim = Sim(cfg, device=device, seed_population=seed_population)
        r = cls(sim, rd, policies=policies, schedule=schedule, **kw)
        r.emit({"type": "run_start", "tick": sim.tick, "severity": "info", "dir": rd.path})
        return r

    @classmethod
    def resume(cls, run_path, checkpoint=None, device=None, overrides: dict | None = None, **kw):
        rd = RunDir(run_path)
        cm = CheckpointManager(rd)
        if checkpoint:
            sim, extra = snapshot.load_snapshot(checkpoint, device=device)
            ck_meta = snapshot.read_meta(checkpoint)
        else:
            sim, extra, ck_meta = cm.load_latest_good(device=device)
            if sim is None:
                raise RuntimeError(f"no loadable good checkpoint in {rd.ckpt_dir}")
        r = cls(sim, rd, **kw)
        r._apply_extra(extra, full=True)
        for k, v in (overrides or {}).items():
            sim.cfg.set(k, v)
            r.log_param(k, sim.cfg.get(k), "CLI override")
        r.emit({"type": "run_resume", "tick": sim.tick, "severity": "info", "checkpoint": ck_meta})
        return r

    def _init_files(self):
        rd = self.rd
        if not os.path.exists(os.path.join(rd.path, "config.json")):
            self.sim.cfg.save(os.path.join(rd.path, "config.json"))
            rd.write_env_json(str(self.sim.device))
            if not os.path.exists(rd.notes_path):
                open(rd.notes_path, "w", encoding="utf-8").write("# Notes\n\n")
        self.save_control_files()

    def save_control_files(self):
        if not self.rd:
            return
        self.engine.save(os.path.join(self.rd.path, "policies.json"))
        self.schedule.save(os.path.join(self.rd.path, "schedule.json"))

    # ------------------------------------------------------------------ extra state in checkpoints
    def extra(self):
        return {"policies": self.engine.to_dict(True), "schedule": self.schedule.to_dict(),
                "events": self.events.to_dict(), "regions": [r.to_dict() for r in self.regions.values()],
                "salt_counter": self.salt_counter}

    def _apply_extra(self, extra, full=False):
        if not extra:
            return
        if full:
            self.engine = PolicyEngine.from_dict(extra.get("policies", {}))
            self.schedule = envmod.Schedule.from_dict(extra.get("schedule", {}))
            self.regions = {r["name"]: envmod.Region(**r) for r in extra.get("regions", [])}
            self.salt_counter = max(self.salt_counter, extra.get("salt_counter", 0))
        ev = envmod.EventSystem.from_dict(extra.get("events", {}))
        if not full:
            ev.next_tick = {}                      # redraw with the new salt
        self.events = ev

    # ------------------------------------------------------------------ logging / events
    def emit(self, ev: dict):
        ev = dict(ev)
        ev.setdefault("tick", self.sim.tick)
        ev = self.elog.emit(ev)
        if self.log:
            lvl = {"debug": 10, "info": 20, "warning": 30, "error": 40, "critical": 50}[ev.get("severity", "info")]
            self.log.log(lvl, json.dumps({k: v for k, v in ev.items() if k != "wall"}, default=str)[:500])
        t = ev.get("type")
        if t in ("policy_fired", "rollback", "run_restart") or (t == "environment_event" and
                                                                  ev.get("source") == "manual"):
            self.intervention_times.append(time.time())
        if ev.get("severity") in ("warning", "error", "critical"):
            self.warnings.append(ev)
        return ev

    def log_param(self, name, value, source):
        if self.rd:
            self.rd.append_history(self.sim.tick, name, value, source)

    def set_param(self, name, value, source="user"):
        self.sim.cfg.set(name, value)
        self.log_param(name, self.sim.cfg.get(name), source)

    # ------------------------------------------------------------------ requests
    def request_pause(self, why=""):
        self.pause_requested = why or "requested"

    def request_stop(self, why=""):
        self.stop_requested = why or "requested"

    def request_screenshot(self):
        self.screenshot_requested = True

    def wait_closed(self, timeout=5.0):
        t0 = time.time()
        while not self._closed_evt and time.time() - t0 < timeout:
            time.sleep(0.05)

    # ------------------------------------------------------------------ checkpoints
    def write_checkpoint(self, reason="auto", good=None, block=False):
        if not self.ckpt:
            return None
        if good is None:
            good = self.sim.has_nan() is None and (self.sim.last_ledger is None or self.sim.last_ledger.get("ok", True))
        path = self.ckpt.write(self.sim, self.extra(), good=good, reason=reason, block=block or reason in (
            "exit", "manual", "before_rollback", "before_restart"))
        self._last_ckpt_tick, self._last_ckpt_wall = self.sim.tick, time.time()
        if self.ckpt:
            self.ckpt.prune()
            self.ckpt.enforce_budget()
        return path

    def _checkpoint_due(self):
        if not self.ckpt:
            return False
        if self.ckpt_every_ticks and self.sim.tick - self._last_ckpt_tick >= self.ckpt_every_ticks:
            return True
        return bool(self.ckpt_every_seconds and time.time() - self._last_ckpt_wall >= self.ckpt_every_seconds)

    # ------------------------------------------------------------------ rollback / restart
    def rollback(self, n=1, tweaks=None, reason="watchdog") -> bool:
        now = time.time()
        self.rollback_times = [t for t in self.rollback_times if now - t < 3600]
        if len(self.rollback_times) >= self.watchdog.cfg.max_rollbacks_per_hour:
            self.emit({"type": "rollback_limit", "severity": "critical", "reason": reason,
                       "message": f"more than {self.watchdog.cfg.max_rollbacks_per_hour} rollbacks in an hour; stopping"})
            self.request_stop("rollback limit exceeded")
            return False
        if not self.ckpt or not self.ckpt.good():
            self.emit({"type": "rollback_failed", "severity": "error", "message": "no good checkpoint"})
            self.request_stop("no good checkpoint to roll back to")
            return False
        self.ckpt.wait()
        self.write_checkpoint(reason="before_rollback", good=False, block=True)
        goods = self.ckpt.good()
        idx = max(0, len(goods) - n)
        tick_before = self.sim.tick
        loaded = None
        for m in goods[idx::-1]:
            try:
                _, extra = snapshot.load_snapshot(m["path"], sim=self.sim)
                loaded = (m, extra)
                break
            except snapshot.CheckpointError as e:
                self.emit({"type": "checkpoint_corrupt", "severity": "warning", "path": m["path"], "error": str(e)})
        if loaded is None:
            self.emit({"type": "rollback_failed", "severity": "critical", "message": "all checkpoints corrupt"})
            self.request_stop("all checkpoints corrupt")
            return False
        m, extra = loaded
        self.salt_counter = max(self.salt_counter, int(self.sim.cfg.rng_salt)) + 1
        self.sim.cfg.set("rng_salt", self.salt_counter)
        self.log_param("rng_salt", self.salt_counter, "rollback")
        tw = tweaks if tweaks is not None else self.watchdog.cfg.stability_tweak
        applied = {}
        if tw == "dt_half" or (isinstance(tw, dict) and "dt_mult" in tw):
            f = 0.5 if tw == "dt_half" else tw["dt_mult"]
            self.set_param("dt", max(0.05, self.sim.cfg.dt * f), "rollback")
            applied["dt"] = self.sim.cfg.dt
        elif tw == "extra_substep" or (isinstance(tw, dict) and "substeps_add" in tw):
            add = 1 if tw == "extra_substep" else tw["substeps_add"]
            self.set_param("substeps", min(16, self.sim.cfg.substeps + add), "rollback")
            applied["substeps"] = self.sim.cfg.substeps
        self._apply_extra(extra, full=False)               # policy counters deliberately NOT restored
        self.rollback_times.append(now)
        self._last_ckpt_tick = self.sim.tick
        self.emit({"type": "rollback", "severity": "warning", "reason": reason, "from_tick": tick_before,
                   "to_tick": self.sim.tick, "salt": self.salt_counter, "tweaks": applied,
                   "checkpoint": os.path.basename(m["path"])})
        return True

    def restart(self, reason=""):
        self.write_checkpoint(reason="before_restart", good=False, block=True)
        self.sim.restart()
        self.emit({"type": "run_restart", "severity": "warning", "reason": reason, "tick": self.sim.tick})

    # ------------------------------------------------------------------ stepping
    def recent_tps(self):
        return self._tps_samples[-1] if self._tps_samples else None

    def _sample_tps(self):
        now = time.perf_counter()
        t0, k0 = self._tps_mark
        if now - t0 >= 2.0:
            self._tps_samples.append((self.sim.tick - k0) / (now - t0))
            self._tps_mark = (now, self.sim.tick)

    def _handle_problem(self, name, action, details):
        sev = "error" if action in ("rollback", "stop") else "warning"
        self.emit({"type": "watchdog", "severity": sev, "check": name, "action": action, "details": details})
        if action == "rollback":
            self.rollback(1, reason=f"watchdog:{name}")
        elif action == "pause":
            self.request_pause(f"watchdog:{name}")
        elif action == "stop":
            self.request_stop(f"watchdog:{name}")

    def tick_once(self):
        sim = self.sim
        if self.schedule.items:
            self.schedule.apply(sim, self.log_param)
        self.events.step(sim, self.emit)
        if self.engine.ramps:
            self.engine.apply_ramps(sim, self.log_param)
        if self.engine.policies and self.engine.due(sim.tick):
            self.engine.evaluate(self)
        try:
            sim.step(1)
        except LedgerError as e:
            self._handle_problem("ledger", self.watchdog.cfg.ledger, {"errors": e.report["errors"]})
            return
        except Exception as e:                           # noqa: BLE001  (CUDA / launch errors)
            self._handle_problem("gpu_error", self.watchdog.cfg.gpu_error, {"error": repr(e)})
            return
        tick = sim.tick
        if sim.last_stats and sim.last_stats.get("tick") != self._last_stats_tick and \
                sim.last_stats.get("tick") == tick:
            self._last_stats_tick = tick
            if self.stats:
                self.stats.add(sim.last_stats)
        if self.watchdog.cfg.interval and tick % self.watchdog.cfg.interval == 0:
            for name, action, details in self.watchdog.check(self):
                self._handle_problem(name, action, details)
        if self._checkpoint_due():
            self.write_checkpoint(reason="auto")
        if self.archive and self.archive_every and tick % self.archive_every == 0:
            self.archive.add_from_sim(sim)
        if self.milestone_every and tick % self.milestone_every == 0:
            self.milestones.check(sim, self.emit)
        if self.frame_every and self.ckpt and self.ckpt.frames_enabled and tick % self.frame_every == 0:
            self.save_frame()
        if self.screenshot_requested:
            self.screenshot_requested = False
            self.save_frame(prefix="shot")
        self._sample_tps()
        self._write_status()
        self._check_stop(tick)

    def advance(self, max_ticks=1, time_budget=None):
        """Run up to max_ticks ticks or until time_budget seconds (UI frames). Returns ticks done."""
        t0 = time.perf_counter()
        done = 0
        while done < max_ticks and not self.stop_requested and not self.pause_requested:
            self.tick_once()
            done += 1
            if time_budget is not None and time.perf_counter() - t0 >= time_budget:
                break
        return done

    # ------------------------------------------------------------------ stop conditions & status
    def _check_stop(self, tick):
        s = self.stop_cfg
        if s.get("max_ticks") and tick >= s["max_ticks"]:
            self.request_stop(f"max ticks {s['max_ticks']}")
        if s.get("max_wall") and time.time() - self.wall0 >= s["max_wall"]:
            self.request_stop(f"max wall time {s['max_wall']}s")
        if self.rd and tick % 500 == 0:
            free = __import__("shutil").disk_usage(self.rd.path).free
            if free < self.min_free_bytes:
                self.emit({"type": "disk_guard", "severity": "critical", "free_mb": free / 2**20})
                self.request_stop("minimum free disk space reached")
        if s.get("stop_on_extinction") and tick % self.engine.eval_interval == 0:
            if StatsView(self.sim).population == 0:
                self._extinct_evals += 1
                if self._extinct_evals >= s.get("extinction_grace_evals", 3):
                    self.request_stop("extinction with no recovery")
            else:
                self._extinct_evals = 0
        m = s.get("metric")
        if m and tick % max(1, s.get("metric_every", 100)) == 0:
            v = StatsView(self.sim).metric(m["metric"])
            ok = {">=": v >= m["value"], ">": v > m["value"], "<=": v <= m["value"], "<": v < m["value"]}[m["op"]]
            if ok:
                self.request_stop(f"{m['metric']} {m['op']} {m['value']} (value {v})")

    def status(self):
        sim = self.sim
        st = sim.last_stats or {}
        now = time.time()
        iv = [t for t in self.intervention_times if now - t < 3600]
        last = self.ckpt.last_result if self.ckpt else None
        return {"tick": sim.tick, "ticks_per_s": self.recent_tps(), "population": st.get("population"),
                "organisms": st.get("organisms"), "interventions_last_hour": len(iv),
                "last_checkpoint": last, "warnings": [{k: w[k] for k in ("type", "severity", "tick") if k in w}
                                                      | {"msg": str(w.get("message", w.get("check", "")))}
                                                      for w in list(self.warnings)[-10:]],
                "stop_requested": self.stop_requested, "paused": self.pause_requested,
                "wall_elapsed": now - self.wall0, "updated": now}

    def _write_status(self, force=False):
        if not self.rd:
            return
        now = time.time()
        if force or now - self._last_status >= self.status_every:
            self._last_status = now
            self.rd.write_json("status.json", self.status())

    # ------------------------------------------------------------------ frames
    def save_frame(self, prefix="frame"):
        if not self.rd:
            return None
        path = os.path.join(self.rd.frames_dir, f"{prefix}_{self.sim.tick:012d}.png")
        if self.frame_hook:
            try:
                self.frame_hook(self.sim, path)
                return path
            except Exception as e:                       # noqa: BLE001
                self.emit({"type": "frame_failed", "severity": "warning", "error": repr(e)})
        from ..io.frames import save_png, render_numpy
        save_png(path, render_numpy(self.sim))
        return path

    # ------------------------------------------------------------------ run loop (headless)
    def run(self, sleep_prevention=True):
        install_signal_handlers(self)
        if sleep_prevention:
            prevent_sleep(True)
        self.emit({"type": "run_loop_start", "tick": self.sim.tick, "severity": "info"})
        try:
            while not self.stop_requested:
                if self.pause_requested:                  # headless: treat 'pause' as stop (documented)
                    self.request_stop(f"pause requested ({self.pause_requested}); headless")
                    break
                self.tick_once()
        except BaseException as e:                         # noqa: BLE001  best-effort checkpoint on crash
            self.emit({"type": "crash", "severity": "critical", "error": repr(e)})
            try:
                self.write_checkpoint(reason="crash", good=False, block=True)
            finally:
                self.close(f"crash: {e!r}")
            raise
        self.close(self.stop_requested)

    def close(self, why=""):
        if self.closed:
            return
        self.closed = True
        try:
            if self.ckpt:
                self.ckpt.wait()
                self.write_checkpoint(reason="exit", block=True)
            if self.stats:
                self.stats.flush()
            self.save_control_files()
            self.emit({"type": "run_stop", "severity": "info", "reason": why, "tick": self.sim.tick})
            self._write_status(force=True)
            if self.archive:
                self.archive.close()
        finally:
            prevent_sleep(False)
            self._closed_evt = True
