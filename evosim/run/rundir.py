"""Run directory layout, structured event log, chunked stats files, checkpoint manager.

runs/<YYYYMMDD-HHMMSS>_<n>/ : config.json, config_history.jsonl, policies.json, schedule.json, env.json,
notes.md, checkpoints/, stats/, events.jsonl, log.txt, archive.sqlite, frames/, status.json
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..io import snapshot

SEVERITIES = {"debug": 0, "info": 1, "warning": 2, "error": 3, "critical": 4}


def _atomic_write_text(path, text):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ============================================================================= event log
class EventLog:
    def __init__(self, path=None, keep=2000):
        self.path = path
        self.recent = deque(maxlen=keep)
        self.listeners = []
        self._lock = threading.Lock()

    def emit(self, ev: dict):
        ev = dict(ev)
        ev.setdefault("wall", time.time())
        ev.setdefault("severity", "info")
        with self._lock:
            self.recent.append(ev)
            if self.path:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ev, default=_json_default) + "\n")
        for fn in self.listeners:
            try:
                fn(ev)
            except Exception:       # a broken listener must never kill the run
                pass
        return ev

    def read_all(self):
        if not self.path or not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Webhook:
    """Optional: POST events >= min_severity as JSON (e.g. to a Telegram/Discord relay). Off by default."""

    def __init__(self, url, min_severity="error"):
        self.url, self.min = url, SEVERITIES[min_severity]
        self.q: queue.Queue = queue.Queue(maxsize=100)
        threading.Thread(target=self._worker, daemon=True).start()

    def __call__(self, ev):
        if SEVERITIES.get(ev.get("severity", "info"), 1) >= self.min:
            try:
                self.q.put_nowait(ev)
            except queue.Full:
                pass

    def _worker(self):
        import urllib.request
        while True:
            ev = self.q.get()
            try:
                req = urllib.request.Request(self.url, data=json.dumps(ev, default=_json_default).encode(),
                                             headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=10).read()
            except Exception:
                pass


# ============================================================================= run directory
class RunDir:
    def __init__(self, path: str):
        self.path = path
        self.ckpt_dir = os.path.join(path, "checkpoints")
        self.stats_dir = os.path.join(path, "stats")
        self.frames_dir = os.path.join(path, "frames")
        for d in (path, self.ckpt_dir, self.stats_dir, self.frames_dir):
            os.makedirs(d, exist_ok=True)
        self.events_path = os.path.join(path, "events.jsonl")
        self.history_path = os.path.join(path, "config_history.jsonl")
        self.status_path = os.path.join(path, "status.json")
        self.archive_path = os.path.join(path, "archive.sqlite")
        self.notes_path = os.path.join(path, "notes.md")
        self.log_path = os.path.join(path, "log.txt")

    @classmethod
    def create(cls, root="runs", name=None):
        os.makedirs(root, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        n = 0
        while True:
            base = f"{stamp}_{n}" + (f"_{name}" if name else "")
            path = os.path.join(root, base)
            try:
                os.makedirs(path)
                break
            except FileExistsError:
                n += 1
        return cls(path)

    def write_json(self, name, obj):
        _atomic_write_text(os.path.join(self.path, name), json.dumps(obj, indent=1, default=_json_default))

    def read_json(self, name, default=None):
        p = os.path.join(self.path, name)
        if not os.path.exists(p):
            return default
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def append_history(self, tick, name, value, source, extra=None):
        rec = {"tick": int(tick), "wall": time.time(), "param": name, "value": value, "source": source}
        rec.update(extra or {})
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")

    def size_bytes(self):
        total = 0
        for r, _, files in os.walk(self.path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(r, f))
                except OSError:
                    pass
        return total

    def setup_logging(self, level="INFO", max_bytes=5_000_000, backups=3):
        lg = logging.getLogger(f"evosim.{os.path.basename(self.path)}")
        lg.setLevel(getattr(logging, level.upper(), logging.INFO))
        lg.handlers.clear()
        h = logging.handlers.RotatingFileHandler(self.log_path, maxBytes=max_bytes, backupCount=backups,
                                                 encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        lg.addHandler(h)
        lg.propagate = False
        return lg

    def write_env_json(self, device_name=""):
        def pkg(n):
            try:
                import importlib.metadata as md
                return md.version(n)
            except Exception:
                return None
        commit = None
        try:
            commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=3,
                                    cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__)))).stdout.strip() or None
        except Exception:
            pass
        driver = None
        try:
            import warp as wp
            driver = str(wp.get_cuda_driver_version()) if hasattr(wp, "get_cuda_driver_version") else None
        except Exception:
            pass
        self.write_json("env.json", {"git_commit": commit, "python": sys.version, "os": platform.platform(),
                                     "gpu": device_name, "cuda_driver": driver,
                                     "packages": {n: pkg(n) for n in ("warp-lang", "numpy", "glfw", "moderngl",
                                                                      "imgui-bundle", "zstandard", "pyarrow",
                                                                      "pyyaml", "nvidia-ml-py")}})


# ============================================================================= stats writer
class StatsWriter:
    """Rows are buffered and written as a new Parquet (or CSV) file per chunk, so a crash loses at most
    one chunk. Files are written atomically."""

    def __init__(self, directory, chunk_rows=50, fmt="parquet", hardware=True):
        self.dir = directory
        self.chunk_rows = chunk_rows
        self.fmt = fmt
        self.rows = []
        self.hardware = hardware
        self.n_files = len([f for f in os.listdir(directory) if f.startswith("stats_")]) if os.path.isdir(directory) else 0
        self._nvml = None
        if hardware:
            try:
                import pynvml
                pynvml.nvmlInit()
                self._nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
            except Exception:
                self._nvml = None
        try:
            import psutil
            self._proc = psutil.Process()
        except Exception:
            self._proc = None
        if fmt == "parquet":
            try:
                import pyarrow  # noqa: F401
            except ImportError:
                self.fmt = "csv"

    def hw_columns(self):
        out = {}
        if self._nvml:
            nv, h = self._nvml
            try:
                out["gpu_temp_c"] = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
                out["gpu_clock_mhz"] = nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_GRAPHICS)
                out["gpu_power_w"] = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
                out["vram_used_mb"] = nv.nvmlDeviceGetMemoryInfo(h).used / 2**20
            except Exception:
                pass
        if self._proc:
            out["proc_rss_mb"] = self._proc.memory_info().rss / 2**20
        return out

    def add(self, row: dict):
        r = dict(row)
        if self.hardware:
            r.update(self.hw_columns())
        self.rows.append(r)
        if len(self.rows) >= self.chunk_rows:
            self.flush()

    def flush(self):
        if not self.rows:
            return None
        keys = sorted({k for r in self.rows for k in r})
        self.n_files += 1
        base = os.path.join(self.dir, f"stats_{self.n_files:06d}")
        if self.fmt == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq
            cols = {k: [r.get(k) for r in self.rows] for k in keys}
            tbl = pa.table(cols)
            tmp = base + ".parquet.tmp"
            pq.write_table(tbl, tmp, compression="zstd")
            os.replace(tmp, base + ".parquet")
            path = base + ".parquet"
        else:
            tmp = base + ".csv.tmp"
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                f.write(",".join(keys) + "\n")
                for r in self.rows:
                    f.write(",".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + "\n")
            os.replace(tmp, base + ".csv")
            path = base + ".csv"
        self.rows = []
        return path

    @staticmethod
    def read_all(directory):
        """All chunks as a list of dict rows (for the UI live plots / exports)."""
        rows = []
        for f in sorted(os.listdir(directory)):
            p = os.path.join(directory, f)
            if f.endswith(".parquet"):
                import pyarrow.parquet as pq
                t = pq.read_table(p).to_pydict()
                n = len(next(iter(t.values()))) if t else 0
                rows += [{k: t[k][i] for k in t} for i in range(n)]
            elif f.endswith(".csv"):
                import csv
                with open(p, encoding="utf-8") as fh:
                    for r in csv.DictReader(fh):
                        rows.append({k: _num(v) for k, v in r.items()})
        return rows


def _num(v):
    if v == "" or v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return v


# ============================================================================= checkpoints
@dataclass
class Retention:
    last_n: int = 10
    hourly_hours: int = 24
    daily: bool = True


class CheckpointManager:
    def __init__(self, rundir: RunDir, retention: Retention | None = None, max_bytes: int | None = None,
                 emit=None):
        self.rd = rundir
        self.retention = retention or Retention()
        self.max_bytes = max_bytes
        self.emit = emit or (lambda ev: None)
        self._thread: threading.Thread | None = None
        self._pending = False
        self._lock = threading.Lock()
        self.last_result: dict | None = None
        self.frames_enabled = True

    # ------------------------------------------------------------------ writing
    def busy(self):
        return self._pending

    def wait(self):
        t = self._thread
        if t is not None:
            t.join()

    def write(self, sim, extra=None, good=True, reason="auto", block=False, wall_time=None):
        """Capture (synchronous GPU->host copy), then compress+write on a background thread. At most one
        write is pending: if one is, an automatic write is skipped (returns None); a blocking/forced
        write waits for it."""
        if self._pending:
            if not block:
                return None
            self.wait()
        cap = snapshot.capture(sim, extra)
        meta = {"tick": int(sim.tick), "wall_time": wall_time or time.time(), "population": int(
            (cap["cell_idx"]).shape[0]), "good": bool(good), "reason": reason}
        path = os.path.join(self.rd.ckpt_dir, f"ckpt_{int(sim.tick):012d}_{int(meta['wall_time'] * 1000) % 10**9:09d}.evck")

        def job():
            try:
                snapshot.write_container(path, snapshot.cap_to_bytes(cap), meta)
                self.last_result = {"ok": True, "path": path, **meta}
                self.emit({"type": "checkpoint_written", "severity": "info", "path": os.path.basename(path), **meta})
            except Exception as e:            # noqa: BLE001
                self.last_result = {"ok": False, "error": repr(e), **meta}
                self.emit({"type": "checkpoint_failed", "severity": "error", "error": repr(e), "tick": meta["tick"]})
            finally:
                self._pending = False
        self._pending = True
        if block:
            job()
        else:
            self._thread = threading.Thread(target=job, daemon=True)
            self._thread.start()
        return path

    # ------------------------------------------------------------------ listing
    def list(self):
        out = []
        for f in sorted(os.listdir(self.rd.ckpt_dir)):
            if not f.endswith(".evck") or f.startswith(".tmp_"):
                continue
            p = os.path.join(self.rd.ckpt_dir, f)
            try:
                m = snapshot.read_meta(p)
            except Exception:
                m = {"tick": -1, "good": False, "corrupt": True}
            m.update(path=p, keep=os.path.exists(p + ".keep"), size=os.path.getsize(p))
            out.append(m)
        out.sort(key=lambda m: (m.get("tick", -1), m.get("wall_time", 0)))
        return out

    def good(self):
        return [m for m in self.list() if m.get("good") and not m.get("corrupt")]

    def mark_keep(self, path, keep=True):
        if keep:
            open(path + ".keep", "w").close()
        elif os.path.exists(path + ".keep"):
            os.unlink(path + ".keep")

    def delete(self, path):
        for p in (path, path + ".keep"):
            if os.path.exists(p):
                os.unlink(p)

    def load_latest_good(self, device=None, sim=None):
        """Newest checkpoint that verifies; falls back to older ones when a file is corrupt."""
        for m in reversed(self.good()):
            try:
                s, extra = snapshot.load_snapshot(m["path"], device=device, sim=sim)
                return s, extra, m
            except snapshot.CheckpointError as e:
                self.emit({"type": "checkpoint_corrupt", "severity": "warning", "path": m["path"], "error": str(e)})
        return None, None, None

    # ------------------------------------------------------------------ retention & budget
    def retained_set(self, items, now=None):
        """Which checkpoints the tiered policy keeps: last N, 1/hour for 24 h, 1/day after, keep-forever."""
        now = now or time.time()
        keep = set()
        newest_first = sorted(items, key=lambda m: (m.get("tick", -1), m.get("wall_time", 0)), reverse=True)
        for m in newest_first[: self.retention.last_n]:
            keep.add(m["path"])
        hours, days = {}, {}
        for m in newest_first:
            age = now - m.get("wall_time", now)
            if age <= self.retention.hourly_hours * 3600:
                hours.setdefault(int(age // 3600), m["path"])
            elif self.retention.daily:
                days.setdefault(int(age // 86400), m["path"])
        keep |= set(hours.values()) | set(days.values())
        for m in items:
            if m.get("keep"):
                keep.add(m["path"])
        good = [m for m in newest_first if m.get("good") and not m.get("corrupt")]
        if good:
            keep.add(good[0]["path"])             # never drop the newest good checkpoint
        return keep

    def prune(self, now=None):
        items = self.list()
        keep = self.retained_set(items, now)
        removed = []
        for m in items:
            if m["path"] not in keep and not m.get("corrupt"):
                self.delete(m["path"])
                removed.append(m["path"])
            elif m.get("corrupt"):
                self.delete(m["path"])
                removed.append(m["path"])
        if removed:
            self.emit({"type": "checkpoint_pruned", "severity": "info", "count": len(removed)})
        return removed

    def enforce_budget(self):
        """Prune by retention; if still over budget delete oldest unprotected checkpoints; then stop
        saving frames; finally warn. Returns the list of actions taken."""
        actions = []
        if not self.max_bytes:
            return actions
        if self.rd.size_bytes() <= self.max_bytes:
            return actions
        if self.prune():
            actions.append("retention")
        if self.rd.size_bytes() > self.max_bytes:
            items = [m for m in self.list() if not m.get("keep")]
            good = self.good()
            protect = {good[-1]["path"]} if good else set()
            for m in items:                                           # oldest first
                if self.rd.size_bytes() <= self.max_bytes:
                    break
                if m["path"] in protect:
                    continue
                self.delete(m["path"])
                actions.append("oldest_checkpoint")
        if self.rd.size_bytes() > self.max_bytes and self.frames_enabled:
            self.frames_enabled = False
            shutil.rmtree(self.rd.frames_dir, ignore_errors=True)
            os.makedirs(self.rd.frames_dir, exist_ok=True)
            actions.append("frames_disabled")
        if self.rd.size_bytes() > self.max_bytes:
            actions.append("over_budget")
            self.emit({"type": "disk_budget_exceeded", "severity": "warning", "size": self.rd.size_bytes(),
                       "budget": self.max_bytes})
        return actions
