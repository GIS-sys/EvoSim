"""EvoSim interactive viewer: glfw window + moderngl scene + Dear ImGui (imgui-bundle) panels.

The simulation runs in batches under a per-frame time budget, so the UI stays responsive at any speed.
"""
from __future__ import annotations

import json
import math
import os
import time
import traceback
from collections import deque

import numpy as np
import warp as wp

from .. import spawn
from ..config import PARAM_SPECS, Config
from ..genome import assembler, isa
from ..io import snapshot
from ..run import env as envmod
from ..run.archive import GenomeArchive
from ..run.policies import PolicyEngine
from ..run.rundir import RunDir
from ..run.runner import Runner, Watchdog, WatchdogConfig
from ..sim import Sim
from ..state import C_CELL_FREE
from . import render as rnd

STATE_PATH = os.path.join(os.path.expanduser("~"), ".evosim", "ui_state.json")
BRUSHES = ("none", "kill", "inject seed", "inject archive", "inject clipboard", "paint nutrients", "paint obstacles",
           "drag cell")
RUN_MODES = ("free run", "until selected divides", "follow lineage", "until selected dies", "until depth +1",
             "until population crosses", "until new largest organism", "until lineage share >", "until ledger fails")
OVERLAYS = ("none", "light map", "opacity grid", "nutrient field", "chemical field", "spatial hash")


class App:
    def __init__(self, args=None, runner: Runner | None = None):
        self.args = args
        self.ui_state = self._load_ui_state()
        if runner is None:
            cfg = Config()
            src = getattr(args, "config", None) or (
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "presets",
                             f"{args.preset}.json") if getattr(args, "preset", None) else None)
            if src:
                cfg = Config.load(src)
            for kv in (getattr(args, "set", None) or []):
                k, _, v = kv.partition("=")
                try:
                    cfg.set(k, json.loads(v))
                except json.JSONDecodeError:
                    cfg.set(k, v)
            if getattr(args, "run_dir", None):
                # Open a run directory WITHOUT touching it: load its latest good checkpoint as a fork (in memory, no run
                # directory attached) and show its stats files. The original run (possibly still active) is never written.
                from ..run.rundir import CheckpointManager, StatsWriter
                rd0 = RunDir(args.run_dir)
                sim0, extra0, meta0 = CheckpointManager(rd0).load_latest_good(device=getattr(args, "device", None))
                if sim0 is None:
                    raise SystemExit(f"no loadable good checkpoint in {rd0.ckpt_dir}")
                runner = self._new_runner(sim0)
                runner._apply_extra(extra0, full=True)
                self.watch_dir = args.run_dir
                self.watch_note = f"fork of {args.run_dir} at tick {sim0.tick} (original run untouched)"
            else:
                runner = self._new_runner(Sim(cfg, device=getattr(args, "device", None)))
        self.runner = runner
        self.pending_restart_params: dict = {}
        # ---- run control
        self.running = False
        self.uncapped = False
        self.target_tps = 60.0
        self.render_every = 1
        self.render_on = True
        self.step_n = 100
        self.acc = 0.0
        self.run_mode = 0
        self.mode_param = {"pop": 1000.0, "pop_dir": 0, "share": 50.0, "depth": 0, "daughter_policy": 0}
        self.mode_state = {}
        self.stop_reason = ""
        self.epoch_on = False
        self.epoch_len = 10000
        self.epoch_topk = 5
        self.last_epoch = 0
        # ---- view
        W, H = self.sim.state.dims.world
        self.cam = np.array([W / 2, H / 2], np.float64)
        self.zoom = 1.0
        self.fit_pending = True
        self.tiled = False
        self.color_mode = 0
        self.trait_index = 0
        self.overlay = 0
        self.overlay_chem = 0
        self.overlay_alpha = 0.6
        self.show_bonds = True
        self.show_regions = True
        self.radius_scale = 1.0
        self.follow = False
        # ---- selection / interaction
        self.sel: list[int] = []                    # uids
        self.brush = 0
        self.brush_radius = 20.0
        self.brush_amount = 0.2
        self.brush_k = 10
        self.brush_file = "seeds/seed_replicator.asm"
        self.mouse_world = np.zeros(2)
        self.cursor_world = np.zeros(2)
        self.layout_name = 'debug'
        self.reset_windows = True
        self.param_search = ''
        self._press = None
        self.frame: dict | None = None
        self.frame_tick = -1
        self._ov_counter = 0
        # ---- tools
        self.rewind = deque()
        self.rewind_every = 500
        self.rewind_max = 20
        self.rewind_bytes = 0
        self._last_rewind_tick = -10**9
        self.history = deque(maxlen=20000)                      # stats snapshots for plots
        self.lineage_hist = deque(maxlen=2000)
        self.ledger_hist = deque(maxlen=2000)
        self._last_hist_tick = -1
        self.errors: dict[str, str] = {}
        self.toast = ""
        self.toast_until = 0.0
        self.fav = set(self.ui_state.get("favorites", []))
        self.panels = {k: v for k, v in self.ui_state.get("panels", {}).items()}
        self.set_layout("debug", first=True) if not self.panels else None
        self.screenshot_next = False
        self.sandbox = None
        self.bp = {"opcode": -1, "addr": -1, "energy": 0.0, "bitten": False, "divide": False, "hit": ""}
        self.log_filter = {"type": "", "min_sev": 0}
        self.archive_view = None
        self.diff_a = self.diff_b = ""
        self.policy_edit = {}
        self.sched_param = "sun_I"
        self.cursor_event = "catastrophe"
        self.region_new = {"name": "region1", "shape": "rect", "x": 100.0, "y": 100.0, "w": 200.0, "h": 200.0}
        self.run_notes = ""
        self.window = None
        self.gl_errors = {}
        self.watch_dir = getattr(self, 'watch_dir', None)
        self.watch_note = getattr(self, 'watch_note', '')
        if self.watch_dir:
            self.reload_watch_stats()
        self.tick_wall = time.perf_counter()

    # ================================================================== construction helpers
    def _new_runner(self, sim, rundir=None, **kw):
        wd = Watchdog(WatchdogConfig(interval=100, nan="pause", ledger="pause", gpu_error="pause"))
        return Runner(sim, rundir, watchdog=wd, hardware_stats=False, stats_format="csv", ckpt_every_ticks=0, **kw)

    @property
    def sim(self) -> Sim:
        return self.runner.sim

    @property
    def cfg(self) -> Config:
        return self.runner.sim.cfg

    def notify(self, msg, secs=4.0):
        self.toast, self.toast_until = msg, time.time() + secs

    def _load_ui_state(self):
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_ui_state(self):
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        self.ui_state["favorites"] = sorted(self.fav)
        self.ui_state["panels"] = self.panels
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.ui_state, f, indent=1)

    def set_layout(self, name, first=False):
        every = ("controls", "params", "inspector", "stats", "profiler", "policies", "schedule", "events",
                 "checkpoints", "archive", "eventlog", "organism", "tools", "run")
        debug = {"controls", "params", "inspector", "organism", "tools", "profiler"}
        monitor = {"controls", "stats", "policies", "events", "eventlog", "run", "checkpoints", "profiler"}
        show = debug if name == "debug" else monitor
        self.panels = {k: (k in show) for k in every}
        self.layout_name = name
        self.reset_windows = True
        if name == "monitor":
            self.render_every = max(self.render_every, 10)
        elif not first:
            self.render_every = 1

    # ================================================================== frame data
    def gather(self):
        """GPU -> host copy of what rendering and picking need."""
        s = self.sim
        c = s.state.cells
        al = np.nonzero(c.alive.numpy() != 0)[0]
        f = {"al": al, "pos": c.pos.numpy()[al], "mass": c.mass.numpy()[al], "angle": c.angle.numpy()[al],
             "uid": c.uid.numpy()[al], "energy": c.energy.numpy()[al], "traits": c.traits.numpy()[al],
             "lineage": c.lineage_id.numpy()[al], "organism": c.organism_id.numpy()[al],
             "state": c.state.numpy()[al], "age": c.age.numpy()[al], "depth": c.division_depth.numpy()[al],
             "origin": c.origin.numpy()[al], "gid": c.genome_id.numpy()[al]}
        f["ghash"] = (s.state.genomes.hash.numpy()[f["gid"]] & 0xFFFFFFFF) if len(al) else f["gid"]
        f["max_energy"] = s.cfg.max_energy_per_mass * f["mass"]
        rho = s.cfg.density * (1.0 + s.cfg.armor_density * f["traits"][:, isa.T_ARMOR]) if len(al) else f["mass"]
        f["rad"] = np.sqrt(np.maximum(f["mass"], 1e-6) / (np.pi * rho)) if len(al) else f["mass"]
        self.frame, self.frame_tick = f, s.tick
        return f

    def slot_of_uid(self, uid):
        f = self.frame
        if f is None or len(f["uid"]) == 0:
            return None
        k = np.nonzero(f["uid"] == uid)[0]
        return int(f["al"][k[0]]) if len(k) else None

    def frame_index(self, uid):
        f = self.frame
        if f is None:
            return None
        k = np.nonzero(f["uid"] == uid)[0]
        return int(k[0]) if len(k) else None

    def selected_slot(self):
        return self.slot_of_uid(self.sel[0]) if self.sel else None

    def cell_scalar(self, name, slot, dtype=None):
        a = getattr(self.sim.state.cells, name)
        return a[slot:slot + 1].numpy()[0]

    def cell_row(self, name, slot):
        return getattr(self.sim.state.cells, name)[slot:slot + 1].numpy()[0]

    # ================================================================== render
    def render_scene(self, ww, wh, fbw, fbh):
        s = self.sim
        st = s.state
        r = self.renderer
        if self.render_on:
            f = self.gather()
            n = len(f["al"])
            if n:
                col = rnd.cell_colors(rnd.COLOR_MODES[self.color_mode], self.trait_index, f)
                r.set_cells(f["pos"], f["rad"], f["angle"], col)
            else:
                r.n_inst = 0
            lines = []
            if self.show_bonds:
                b = st.bonds
                al = b.alive.numpy() != 0
                if al.any():
                    pa = st.cells.pos.numpy()
                    ia, ib = b.a.numpy()[al], b.b.numpy()[al]
                    lines.append(rnd.bond_lines(pa[ia], pa[ib], b.strain.numpy()[al], st.dims.world, self.cfg.break_strain))
            if self.show_regions:
                for reg in self.runner.regions.values():
                    if reg.shape == "rect":
                        lines.append(rnd.rect_lines(reg.x, reg.y, reg.w, reg.h, (1.0, 0.8, 0.2, 0.8)))
                    else:
                        lines.append(rnd.circle_lines(reg.x, reg.y, reg.w, (1.0, 0.8, 0.2, 0.8)))
            lines.append(self.selection_lines())
            if lines:
                r.set_lines(np.concatenate([l for l in lines if len(l)]) if any(len(l) for l in lines) else np.zeros((0, 6)))
            sel_idx = [i for i in (self.frame_index(u) for u in self.sel) if i is not None]
            if sel_idx:
                r.set_selection(f["pos"][sel_idx], f["rad"][sel_idx])
            else:
                r.n_sel = 0
            self.update_overlay()
        if self.follow and self.sel:
            sl = self.frame_index(self.sel[0]) if self.frame is not None else None
            if sl is not None:
                self.cam = self.frame["pos"][sl].astype(np.float64)
        tint = (0.2, 0.9, 0.3) if self.overlay in (3, 4) else (1.0, 0.95, 0.6) if self.overlay == 1 else (0.6, 0.7, 1.0)
        r.draw(self.cam, (ww, wh), self.zoom, st.dims.world, self.tiled, overlay_alpha=self.overlay_alpha,
               overlay_tint=tint, radius_scale=self.radius_scale)
        if self.screenshot_next:
            self.screenshot_next = False
            os.makedirs("screenshots", exist_ok=True)
            from ..io.frames import save_png
            path = os.path.join("screenshots", f"shot_{self.sim.tick:09d}.png")
            save_png(path, r.read_pixels(fbw, fbh))
            self.notify(f"saved {path}")

    def selection_lines(self):
        """Selected cell's contact sectors, sun / point-light rays and contacts."""
        sl = self.selected_slot()
        if sl is None or self.frame is None:
            return np.zeros((0, 6), np.float32)
        cfg, st = self.cfg, self.sim.state
        i = self.frame_index(self.sel[0])
        p = self.frame["pos"][i].astype(np.float64)
        rad = float(self.frame["rad"][i])
        ang = float(self.frame["angle"][i])
        out = []

        def seg(a, b, c):
            out.append([a[0], a[1], *c])
            out.append([b[0], b[1], *c])
        for k, col in enumerate(((1, .3, .3, .9), (.3, 1, .3, .9), (.3, .3, 1, .9), (1, 1, .3, .9))):
            # sector k: F=0 R=1 B=2 L=3 ; boundaries at +-45 deg around the sector centre (body frame; +y = left)
            centre = ang + (0, -math.pi / 2, math.pi, math.pi / 2)[k]
            for sgn in (-1, 1):
                a = centre + sgn * math.pi / 4
                seg(p, p + (rad * 1.6) * np.array([math.cos(a), math.sin(a)]), (.6, .6, .6, .5))
            seg(p + rad * np.array([math.cos(centre), math.sin(centre)]),
                p + rad * 1.5 * np.array([math.cos(centre), math.sin(centre)]), col)
        sun = np.array([float(self.sim._p.d_sun_dir[0]), float(self.sim._p.d_sun_dir[1])])
        seg(p, p + sun * cfg.shadow_ray_len, (1, .95, .3, .8))
        W = st.dims.world
        for pl in cfg.point_lights[:cfg.max_point_lights]:
            d = np.array([pl["x"], pl["y"]]) - p
            d -= np.array(W) * np.round(d / np.array(W))
            seg(p, p + d, (1, .7, .2, .6))
        nc = int(st.scratch.n_contacts[sl:sl + 1].numpy()[0])
        if nc:
            cs = st.scratch.contacts[sl:sl + 1].numpy()[0][:nc]
            allpos = st.cells.pos.numpy()
            for j in cs:
                d = allpos[j].astype(np.float64) - p
                d -= np.array(W) * np.round(d / np.array(W))
                seg(p, p + d, (1, 0.4, 0.8, 0.9))
        return np.array(out, np.float32).reshape(-1, 6)

    def update_overlay(self):
        self._ov_counter += 1
        if self.overlay == 0:
            self.renderer.clear_overlay()
            return
        if self._ov_counter % 3 and self.renderer.tex is not None:
            return
        st, s = self.sim.state, self.sim
        f = st.fields
        if self.overlay == 1:
            out = wp.zeros((st.dims.lnx, st.dims.lny), dtype=float, device=st.device)
            from ..kernels import light
            p = s.params()
            wp.launch(light.sun_transmittance_map_kernel, (st.dims.lnx, st.dims.lny), [f, out, p], device=st.device)
            self.renderer.set_overlay(out.numpy(), 0.0, None)
        elif self.overlay == 2:
            self.renderer.set_overlay(f.opacity.numpy())
        elif self.overlay == 3:
            self.renderer.set_overlay(f.nutrient.numpy(), 0.0, None)
        elif self.overlay == 4:
            self.renderer.set_overlay(f.chem.numpy()[min(self.overlay_chem, st.dims.K - 1)], 0.0, None)
        elif self.overlay == 5:
            p = s.params()
            nx, ny = p.d_hash_nx, p.d_hash_ny
            cnt = (st.hash.end.numpy()[:nx * ny] - st.hash.start.numpy()[:nx * ny]).clip(0).reshape(ny, nx).T
            self.renderer.set_overlay(cnt.astype(np.float32), 0.0, None)

    # ================================================================== interaction
    def screen_to_world(self, mx, my, ww, wh):
        return np.array([self.cam[0] + (mx - ww / 2) / self.zoom, self.cam[1] - (my - wh / 2) / self.zoom])

    def handle_mouse(self, io, ww, wh):
        from imgui_bundle import imgui
        mx, my = io.mouse_pos.x, io.mouse_pos.y
        self.mouse_world = self.screen_to_world(mx, my, ww, wh)
        if io.want_capture_mouse:
            self._press = None
            return
        self.cursor_world = self.mouse_world.copy()
        if io.mouse_wheel != 0:
            before = self.screen_to_world(mx, my, ww, wh)
            self.zoom = float(np.clip(self.zoom * (1.15 ** io.mouse_wheel), 0.05, 400.0))
            after = self.screen_to_world(mx, my, ww, wh)
            self.cam += before - after
        if imgui.is_mouse_dragging(2) or imgui.is_mouse_dragging(1):
            self.cam -= np.array([io.mouse_delta.x, -io.mouse_delta.y]) / self.zoom
            self.follow = False
        if imgui.is_mouse_clicked(0):
            self._press = (mx, my, time.time())
        if self.brush and imgui.is_mouse_down(0):
            self.apply_brush(imgui.is_mouse_clicked(0))
        elif self._press and imgui.is_mouse_dragging(0) and not self.brush:
            self.cam -= np.array([io.mouse_delta.x, -io.mouse_delta.y]) / self.zoom
            self.follow = False
        if imgui.is_mouse_released(0) and self._press:
            moved = math.hypot(mx - self._press[0], my - self._press[1])
            if moved < 4 and not self.brush:
                self.pick(self.mouse_world, io.key_shift)
            self._press = None

    def pick(self, wpos, add=False):
        f = self.frame
        if f is None or len(f["al"]) == 0:
            return
        W = np.array(self.sim.state.dims.world)
        d = f["pos"] - wpos
        d -= W * np.round(d / W)
        dist = np.hypot(d[:, 0], d[:, 1]) - f["rad"]
        k = int(np.argmin(dist))
        if dist[k] > 6.0 / self.zoom:
            if not add:
                self.sel = []
            return
        uid = int(f["uid"][k])
        if add:
            self.sel = [u for u in self.sel if u != uid] if uid in self.sel else self.sel + [uid]
        else:
            self.sel = [uid]
        self.sim.set_traced([self.slot_of_uid(u) for u in self.sel if self.slot_of_uid(u) is not None][:8])
        self.apply_breakpoints()

    def select_organism(self):
        sl = self.selected_slot()
        if sl is None:
            return
        mem = self.sim.organism_members(sl)
        self.gather()
        self.sel = [int(u) for u in self.sim.state.cells.uid.numpy()[mem]][:256]

    def center_on_selected(self):
        i = self.frame_index(self.sel[0]) if self.sel else None
        if i is not None:
            self.cam = self.frame["pos"][i].astype(np.float64)

    def fit_view(self, ww, wh):
        W, H = self.sim.state.dims.world
        self.zoom = 0.95 * min(ww / W, wh / H)
        self.cam = np.array([W / 2, H / 2], np.float64)

    def select_slot(self, slot):
        if slot is None:
            return
        if self.frame is None or self.frame_tick != self.sim.tick:
            self.gather()
        self.sel = [int(self.cell_scalar("uid", slot))]
        self.sim.set_traced([slot])
        self.apply_breakpoints()

    # ---- find
    def find(self, what, value=None):
        f = self.gather()
        if len(f["al"]) == 0:
            return self.notify("no cells")
        if what == "uid":
            k = np.nonzero(f["uid"] == int(value))[0]
            if not len(k):
                return self.notify(f"uid {value} not alive")
            k = int(k[0])
        elif what == "largest organism":
            u, inv, cnt = np.unique(f["organism"], return_inverse=True, return_counts=True)
            k = int(np.nonzero(inv == np.argmax(cnt))[0][0])
        elif what == "oldest":
            k = int(np.argmax(f["age"]))
        elif what == "most common genome":
            u, inv, cnt = np.unique(f["gid"], return_inverse=True, return_counts=True)
            k = int(np.nonzero(inv == np.argmax(cnt))[0][0])
        elif what == "random of lineage":
            idx = np.nonzero(f["lineage"] == int(value))[0]
            if not len(idx):
                return self.notify(f"lineage {value} not alive")
            k = int(np.random.default_rng().choice(idx))
        else:
            return
        self.select_slot(int(f["al"][k]))
        self.cam = f["pos"][k].astype(np.float64)
        self.zoom = max(self.zoom, 6.0)

    def lineage_step(self, direction):
        """Next / previous cell in lineage: parent (prev) or a living daughter (next) of the selected cell."""
        sl = self.selected_slot()
        if sl is None:
            return
        c = self.sim.state.cells
        uid = int(self.cell_scalar("uid", sl))
        if direction < 0:
            p = int(self.cell_scalar("parent_uid", sl))
            tgt = self.slot_of_uid(p)
        else:
            f = self.gather()
            k = np.nonzero(c.parent_uid.numpy()[f["al"]] == uid)[0]
            tgt = int(f["al"][k[0]]) if len(k) else None
        if tgt is None:
            return self.notify("no living " + ("parent" if direction < 0 else "daughter"))
        self.select_slot(tgt)
        self.center_on_selected()

    # ================================================================== brushes
    def apply_brush(self, first_click):
        s, st = self.sim, self.sim.state
        b = BRUSHES[self.brush]
        w = self.mouse_world
        W = np.array(st.dims.world)
        rs = np.random.default_rng(int(time.time() * 1000) % 2**31)
        if self.frame is None:
            self.gather()
        f = self.frame
        if b == "kill":
            if len(f["al"]):
                d = f["pos"] - w
                d -= W * np.round(d / W)
                hit = f["al"][np.hypot(d[:, 0], d[:, 1]) <= self.brush_radius]
                if len(hit):
                    s.kill(hit)
        elif b in ("inject seed", "inject archive", "inject clipboard") and first_click:
            gens = []
            if b == "inject seed":
                gens = [s.load_genome_file(self.brush_file)]
            elif b == "inject archive" and self.runner.archive and self.archive_view:
                g = self.runner.archive.get(self.archive_view)
                gens = [g] if g else []
            elif b == "inject clipboard":
                from imgui_bundle import imgui
                txt = imgui.get_clipboard_text() or ""
                gens = [assembler.assemble(txt, s.state.dims.K, s.state.dims.KS, s.state.dims.MB)] if txt.strip() else []
                if not gens:
                    self.notify("clipboard has no assembler text")
            if gens:
                s.inject(self.brush_k, gens, rs, placement="near", center=tuple(w), radius=self.brush_radius)
        elif b == "paint nutrients":
            self._paint(st.fields.nutrient, w, add=self.brush_amount * 0.2, book=True)
        elif b == "paint obstacles":
            if not self.cfg.obstacles_enabled:
                self.cfg.set("obstacles_enabled", True)
            self._paint(st.fields.obstacle, w, set_value=0 if wp_key_ctrl() else 1)
        elif b == "drag cell":
            sl = self.selected_slot()
            if sl is None and first_click and len(f["al"]):
                self.pick(w)
                sl = self.selected_slot()
            if sl is not None:
                for name, val in (("pos", w.astype(np.float32)), ("vel", np.zeros(2, np.float32))):
                    a = getattr(st.cells, name)
                    arr = a.numpy()
                    arr[sl] = val
                    a.assign(arr)

    def _paint(self, field, w, add=None, set_value=None, book=False):
        st = self.sim.state
        d = st.dims
        X = (np.arange(d.fnx) + 0.5) * d.fh
        Y = (np.arange(d.fny) + 0.5) * d.fh
        W, H = d.world
        dx = X - w[0]
        dx -= W * np.round(dx / W)
        dy = Y - w[1]
        dy -= H * np.round(dy / H)
        mask = (dx[:, None] ** 2 + dy[None, :] ** 2) <= self.brush_radius ** 2
        arr = field.numpy()
        if add is not None:
            arr[mask] += add
            if book:
                self.sim.book_intervention(field=float(add * mask.sum() * d.fh * d.fh))
        else:
            arr[mask] = set_value
        field.assign(arr)

    # ================================================================== genome / cell edits
    def poke(self, name, slot, value):
        a = getattr(self.sim.state.cells, name)
        arr = a.numpy()
        arr[slot] = value
        a.assign(arr)

    def set_cell_genome(self, slot, words):
        s = self.sim
        (gid,) = spawn.upload_genomes(s, [words])
        g = s.state.genomes
        rc = g.refcount.numpy()
        old = int(self.cell_scalar("genome_id", slot))
        rc[old] -= 1
        rc[gid] += 1
        g.refcount.assign(rc)
        self.poke("genome_id", slot, gid)
        self.poke("ip", slot, 0)

    def genome_words(self, slot):
        g = self.sim.state.genomes
        gid = int(self.cell_scalar("genome_id", slot))
        n = int(g.length[gid:gid + 1].numpy()[0])
        return [int(x) & 0xFFFFFFFF for x in g.words[gid:gid + 1].numpy()[0][:n]], gid

    def mutate_now(self, slot):
        s = self.sim
        w, gid = self.genome_words(slot)
        for _ in range(30):
            new = spawn.mutate_genomes(s, [gid], mult=3.0)[0]
            if int(new) != gid:
                g = s.state.genomes
                rc = g.refcount.numpy()
                rc[gid] -= 1
                rc[new] += 1
                g.refcount.assign(rc)
                self.poke("genome_id", slot, int(new))
                return self.notify("mutation applied")
        self.notify("mutation draw was identical 30 times; raise the mutation rates")

    # ================================================================== apply breakpoints
    def apply_breakpoints(self):
        t = self.sim.state.trace
        op, ad = self.bp["opcode"], self.bp["addr"]
        t.bp_op.fill_(op)
        t.bp_addr.fill_(ad)
        t.hit.zero_()

    def breakpoint_hit(self):
        """Host-side checks after a tick for traced cells. Returns a reason string or ''."""
        s = self.sim
        if not self.sel:
            return ""
        if int(s.state.trace.hit.numpy().max()) > 0:
            return "instruction breakpoint"
        for u in self.sel[:8]:
            sl = self.slot_of_uid(u)
            if sl is None:
                continue
            if self.bp["energy"] > 0 and float(self.cell_scalar("energy", sl)) < self.bp["energy"]:
                return f"traced cell energy < {self.bp['energy']}"
            fl = int(self.cell_scalar("action_flags", sl))
            if self.bp["bitten"] and fl & isa.ACT_BITTEN:
                return "traced cell was bitten"
            if self.bp["divide"] and fl & (isa.ACT_SPLIT | isa.ACT_SPLITB):
                return "traced cell issued a split"
        return ""

    # ================================================================== run control
    def any_breakpoint(self):
        b = self.bp
        return bool(self.sel) and (b["opcode"] >= 0 or b["addr"] >= 0 or b["energy"] > 0 or b["bitten"] or b["divide"])

    def start_mode(self):
        self.mode_state = {}
        m = self.run_mode
        sl = self.selected_slot()
        if m in (1, 2, 3, 4) and sl is None:
            self.notify("select a cell first")
            return False
        if m == 4:
            self.mode_state["target"] = int(self.cell_scalar("division_depth", sl)) + 1
            self.mode_state["desc"] = {int(self.cell_scalar("uid", sl))}
        if m == 5:
            self.mode_state["above"] = self.sim.population() < self.mode_param["pop"]
        if m == 6:
            self.mode_state["best"] = self.runner.milestones.best_org
        return True

    def check_mode_after_tick(self):
        """Per-tick condition of event-driven modes. Returns a stop reason or ''."""
        m = self.run_mode
        s = self.sim
        c = s.state.cells
        if m == 0:
            return ""
        if self.runner.pause_requested:
            return "paused: " + str(self.runner.pause_requested)
        if m in (1, 2, 3, 4) and self.sel:
            uid = self.sel[0]
            sl = self.slot_of_uid(uid)
            if sl is None or int(c.uid[sl:sl + 1].numpy()[0]) != uid or int(c.alive[sl:sl + 1].numpy()[0]) == 0:
                # selected slot reused/dead: reconcile via the device
                pass
            alive_uid = c.uid[sl:sl + 1].numpy()[0] if sl is not None else -1
            alive = bool(c.alive[sl:sl + 1].numpy()[0]) if sl is not None else False
            if m == 3:
                if not alive or alive_uid != uid:
                    return "selected cell died"
                self.frame = None
            if m in (1, 2) and alive and alive_uid == uid and int(c.action_flags[sl:sl + 1].numpy()[0]) & isa.ACT_SPLIT_OK:
                pol = self.mode_param["daughter_policy"]
                dau = int(s.state.scratch.daughter[sl:sl + 1].numpy()[0])
                bonded = bool(int(c.action_flags[sl:sl + 1].numpy()[0]) & isa.ACT_SPLITB)
                if m == 2 and dau >= 0:
                    pick = {0: True, 1: not bonded, 2: bool(np.random.default_rng().random() < 0.5)}[pol]
                    if pick:
                        self.sel = [int(c.uid[dau:dau + 1].numpy()[0])]
                        self.frame = None
                        self.gather()
                        s.set_traced([dau])
                return "selected cell divided"
            if m in (1, 2, 4) and (not alive or alive_uid != uid) and m != 3:
                if m == 4:
                    pass
                else:
                    return "selected cell died before dividing"
        if m == 4:
            f = np.nonzero(c.alive.numpy() != 0)[0]
            uid = c.uid.numpy()
            par = c.parent_uid.numpy()
            dep = c.division_depth.numpy()
            desc = self.mode_state["desc"]
            for i in f[(c.action_flags.numpy()[f] & isa.ACT_BORN) != 0]:
                if int(par[i]) in desc:
                    desc.add(int(uid[i]))
                    if int(dep[i]) >= self.mode_state["target"]:
                        return f"descendant reached division depth {int(dep[i])}"
        if m == 5:
            pop = s.population()
            if (pop >= self.mode_param["pop"]) == self.mode_state["above"] and False:
                pass
            if self.mode_state["above"] != (pop >= self.mode_param["pop"]) or (
                    not self.mode_state["above"] and pop >= self.mode_param["pop"]):
                return f"population crossed {self.mode_param['pop']:.0f}"
        if m == 6 and s.tick % 10 == 0:
            self.runner.milestones.check(s, self.runner.emit)
            if self.runner.milestones.best_org > self.mode_state["best"]:
                return f"new largest organism ({self.runner.milestones.best_org} cells)"
        if m == 7 and s.tick % 10 == 0:
            from ..run.policies import StatsView
            share, who = StatsView(s).dominance("lineage")
            if share * 100 > self.mode_param["share"]:
                return f"lineage {who} holds {share * 100:.0f}%"
        if m == 8 and s.last_ledger is not None and not s.last_ledger.get("ok", True):
            return "ledger check failed"
        return ""

    def advance(self, dt_frame):
        r = self.runner
        if not self.running:
            return
        if r.stop_requested:
            self.running = False
            self.stop_reason = f"stop: {r.stop_requested}"
            return
        budget = 0.012 if self.render_on else 0.05
        per_tick_mode = self.run_mode != 0 or self.any_breakpoint()
        if self.uncapped:
            n = max(1, self.render_every)
        else:
            self.acc = min(self.acc + self.target_tps * dt_frame, 2000)
            n = int(self.acc)
            self.acc -= n
        done = 0
        t0 = time.perf_counter()
        while done < n and self.running:
            if per_tick_mode:
                r.advance(1)
                done += 1
                why = self.check_mode_after_tick() or self.breakpoint_hit()
                if why:
                    self.running = False
                    self.stop_reason = why
                    self.notify(why)
            else:
                k = r.advance(n - done, time_budget=budget - (time.perf_counter() - t0))
                done += max(1, k)
            if r.pause_requested:
                self.running = False
                self.stop_reason = "paused: " + str(r.pause_requested)
                r.pause_requested = None
                self.notify(self.stop_reason)
            if r.stop_requested:
                self.running = False
                self.stop_reason = f"stop: {r.stop_requested}"
            if time.perf_counter() - t0 > budget + 0.02:
                break
        self.post_tick()

    def step(self, n=1):
        was = self.running
        self.runner.advance(n)
        self.post_tick()
        self.gather()
        self.running = was

    def post_tick(self):
        s = self.sim
        if self.epoch_on and s.tick - self.last_epoch >= self.epoch_len:
            self.apply_epoch()
        if s.last_stats and s.last_stats.get("tick", -1) != self._last_hist_tick:
            self._last_hist_tick = s.last_stats["tick"]
            self.history.append(dict(s.last_stats))
            c = s.state.cells
            al = c.alive.numpy() != 0
            lin = c.lineage_id.numpy()[al]
            if len(lin):
                u, cn = np.unique(lin, return_counts=True)
                self.lineage_hist.append((s.tick, dict(zip(u.tolist(), cn.tolist()))))
            if s.last_ledger:
                self.ledger_hist.append((s.tick, dict(s.last_ledger["cell_terms"]), dict(s.last_ledger["global_terms"])))
        if self.rewind_every and s.tick - self._last_rewind_tick >= self.rewind_every:
            self.push_rewind()

    def push_rewind(self):
        s = self.sim
        cap = snapshot.capture(s, self.runner.extra())
        self.rewind.append((s.tick, cap))
        self._last_rewind_tick = s.tick
        while len(self.rewind) > self.rewind_max:
            self.rewind.popleft()
        self.rewind_bytes = sum(sum(v.nbytes for v in c.values()) for _, c in self.rewind)

    def rewind_to(self, idx):
        if not (0 <= idx < len(self.rewind)):
            return
        tick, cap = self.rewind[idx]
        snapshot.restore(self.sim, cap)
        while len(self.rewind) > idx + 1:
            self.rewind.pop()                                   # continuing from here forks the timeline
        self._last_rewind_tick = tick
        self.history = deque((h for h in self.history if h["tick"] <= tick), maxlen=self.history.maxlen)
        self.gather()
        self.notify(f"rewound to tick {tick} (timeline forks from here)")

    def apply_epoch(self):
        """Epoch rule: keep the top-K lineages by population and re-spread them to the previous population."""
        s = self.sim
        self.last_epoch = s.tick
        c = s.state.cells
        al = np.nonzero(c.alive.numpy() != 0)[0]
        if len(al) == 0:
            return
        lin = c.lineage_id.numpy()[al]
        u, cn = np.unique(lin, return_counts=True)
        keep = set(u[np.argsort(-cn)[:self.epoch_topk]].tolist())
        drop = al[~np.isin(lin, list(keep))]
        target = len(al)
        s.kill(drop)
        s.step(1)
        al2 = np.nonzero(c.alive.numpy() != 0)[0]
        need = max(0, min(target - len(al2), int(s.state.counters.numpy()[C_CELL_FREE])))
        if need and len(al2):
            rs = np.random.default_rng(s.tick)
            src = al2[rs.integers(0, len(al2), size=need)]
            s.clone_cells(src, rs, mutate_mult=0.0, placement="uniform")
        self.runner.emit({"type": "epoch", "tick": s.tick, "kept_lineages": sorted(keep), "culled": int(len(drop))})

    def reload_watch_stats(self):
        """Load the stats files of the watched run directory into the plot history (read-only)."""
        from ..run.rundir import StatsWriter
        try:
            rows = StatsWriter.read_all(os.path.join(self.watch_dir, "stats"))
        except Exception as e:                                   # noqa: BLE001
            self.notify(f"could not read stats: {e}")
            return
        self.history = deque((r for r in rows if r.get("tick") is not None and r["tick"] <= self.sim.tick or True),
                             maxlen=self.history.maxlen if hasattr(self, "history") else 20000)

    # ================================================================== sim (re)creation
    def adopt_runner(self, runner):
        self.runner = runner
        self.frame = None
        self.sel = []
        self.rewind.clear()
        self.history.clear()
        self.lineage_hist.clear()
        self._last_hist_tick = -1
        self.fit_pending = True
        self.running = False

    def restart_with_params(self):
        cfg = self.cfg.copy()
        for k, v in self.pending_restart_params.items():
            cfg.set(k, v)
        self.pending_restart_params = {}
        old = self.runner
        r = self._new_runner(Sim(cfg, device=old.sim.device), old.rd, policies=old.engine, schedule=old.schedule,
                             events=old.events, regions=old.regions)
        self.adopt_runner(r)
        self.notify("simulation restarted with new restart-time parameters")

    def start_recording(self):
        if self.runner.rd:
            return self.notify("already recording to " + self.runner.rd.path)
        rd = RunDir.create("runs", "ui")
        old = self.runner
        r = self._new_runner(old.sim, rd, policies=old.engine, schedule=old.schedule, events=old.events,
                             regions=old.regions, ckpt_every_ticks=10000)
        self.runner = r
        r.emit({"type": "run_start", "severity": "info", "dir": rd.path})
        self.notify("recording to " + rd.path)

    def fork_checkpoint(self, path):
        sim, extra = snapshot.load_snapshot(path, device=self.sim.device)
        rd = RunDir.create("runs", "fork")
        r = self._new_runner(sim, rd)
        r._apply_extra(extra, full=True)
        r.write_checkpoint(reason="fork_origin", block=True)
        self.adopt_runner(r)
        self.notify("forked into " + rd.path)

    def load_checkpoint_inplace(self, path):
        snapshot.load_snapshot(path, sim=self.sim)
        self.gather()
        self.notify(f"loaded tick {self.sim.tick}")


def wp_key_ctrl():
    from imgui_bundle import imgui
    return imgui.get_io().key_ctrl


# ====================================================================== main
def run_app(args):
    import glfw
    import moderngl
    from imgui_bundle import imgui, implot
    from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
    from . import panels

    app = App(args)
    if not glfw.init():
        raise RuntimeError("glfw.init() failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    win = glfw.create_window(1600, 950, "EvoSim", None, None)
    if not win:
        glfw.terminate()
        raise RuntimeError("could not create an OpenGL 3.3 window (update the graphics driver)")
    glfw.make_context_current(win)
    glfw.swap_interval(1)
    ctx = moderngl.create_context()
    app.window = win
    app.renderer = rnd.SceneRenderer(ctx)
    imgui.create_context()
    implot.create_context()
    io = imgui.get_io()
    io.config_flags |= imgui.ConfigFlags_.docking_enable
    io.set_ini_filename(os.path.join(os.path.dirname(STATE_PATH), "imgui.ini"))
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    impl = GlfwRenderer(win)

    def _get_clipboard(_ctx):
        # glfw returns None for an empty / non-text clipboard; the stock backend would crash on .decode
        try:
            t = glfw.get_clipboard_string(win)
        except Exception:                                   # noqa: BLE001
            return ""
        return t.decode("utf-8", "replace") if isinstance(t, bytes) else (t or "")
    imgui.get_platform_io().platform_get_clipboard_text_fn = _get_clipboard
    smoke = int(getattr(args, "smoke", 0) or 0)
    frame_no = 0
    last = time.perf_counter()
    try:
        while not glfw.window_should_close(win):
            glfw.poll_events()
            impl.process_inputs()
            now = time.perf_counter()
            dt = min(now - last, 0.1)
            last = now
            ww, wh = glfw.get_window_size(win)
            fbw, fbh = glfw.get_framebuffer_size(win)
            if ww <= 0 or wh <= 0:
                continue
            if app.fit_pending:
                app.fit_view(ww, wh)
                app.fit_pending = False
            if smoke:
                panels.smoke_actions(app, frame_no)
            app.advance(dt)
            ctx.error                                    # clear anything pending from earlier phases
            if app.render_on and (not app.uncapped or frame_no % max(1, app.render_every) == 0 or not app.running):
                app.render_scene(ww, wh, fbw, fbh)
                e = ctx.error
                if e != "GL_NO_ERROR" and "scene" not in app.gl_errors:
                    app.gl_errors["scene"] = f"{e} (frame {frame_no}, overlay {app.overlay}, screenshot={app.screenshot_next})"
                    print("[ui] GL error after scene:", app.gl_errors["scene"])
            else:
                ctx.viewport = (0, 0, fbw, fbh)
            imgui.new_frame()
            app.handle_mouse(io, ww, wh)
            panels.draw_all(app, ww, wh)
            imgui.render()
            impl.render(imgui.get_draw_data())
            glfw.swap_buffers(win)
            frame_no += 1
            if smoke and frame_no >= smoke:
                os.makedirs("screenshots", exist_ok=True)
                from ..io.frames import save_png
                save_png(os.path.join("screenshots", "ui_smoke.png"), app.renderer.read_pixels(fbw, fbh))
                break
    finally:
        try:
            app.save_ui_state()
            if app.runner.rd:
                app.runner.close("ui closed")
        finally:
            try:
                impl.shutdown()
            except Exception:
                pass
            glfw.terminate()
    if app.errors:
        print("panel errors:", json.dumps(app.errors, indent=1))
        return 3 if smoke else 0
    return 0
