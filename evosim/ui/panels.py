"""All ImGui panels. Each draws into its own window and is isolated by safe(): an exception in one
panel is shown inside that panel (and listed on exit) instead of killing the UI."""
from __future__ import annotations

import difflib
import json
import math
import os
import time
import traceback

import numpy as np
from imgui_bundle import imgui, implot

from ..config import GROUPS, PARAM_SPECS, Config
from ..genome import assembler, isa
from ..genome.reference_vm import CellVM, VMInputs, run_tick
from ..io import snapshot
from ..run import env as envmod
from ..run.policies import BUILTIN_NAMES, Policy, builtin
from ..run.rundir import SEVERITIES, StatsWriter
from ..state import EB_NAMES, LEDGER_NAMES
from . import render as rnd
from .app import BRUSHES, OVERLAYS, RUN_MODES, STATE_PATH

V2, V4 = imgui.ImVec2, imgui.ImVec4
GOOD, BAD, WARN, DIM = V4(0.4, 0.9, 0.4, 1), V4(1, 0.4, 0.4, 1), V4(1, 0.75, 0.3, 1), V4(0.6, 0.6, 0.65, 1)
PRESET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "presets")

LAYOUT = {   # (x, y, w, h) as fractions of the window
    "controls": (0.0, 0.03, 0.2, 0.33), "params": (0.0, 0.36, 0.2, 0.57), "inspector": (0.8, 0.03, 0.2, 0.60),
    "organism": (0.8, 0.63, 0.2, 0.30), "tools": (0.2, 0.03, 0.17, 0.5), "profiler": (0.2, 0.55, 0.17, 0.38),
    "stats": (0.2, 0.03, 0.6, 0.55), "policies": (0.0, 0.36, 0.3, 0.57), "events": (0.3, 0.58, 0.25, 0.35),
    "eventlog": (0.55, 0.58, 0.25, 0.35), "run": (0.8, 0.03, 0.2, 0.4), "checkpoints": (0.8, 0.43, 0.2, 0.5),
    "schedule": (0.3, 0.1, 0.4, 0.6), "archive": (0.3, 0.1, 0.4, 0.6),
}
TITLES = {"controls": "Run controls", "params": "Parameters", "inspector": "Cell inspector", "stats": "Stats & plots",
          "profiler": "Profiler", "policies": "Policies", "schedule": "Schedule", "events": "Environment events",
          "checkpoints": "Checkpoints", "archive": "Genome archive", "eventlog": "Event log",
          "organism": "Organism view", "tools": "Tools", "run": "Run"}


def kv(k, v, color=None):
    imgui.text_colored(DIM, f"{k}")
    imgui.same_line(130)
    if color:
        imgui.text_colored(color, str(v))
    else:
        imgui.text(str(v))


def tab(app, label, fn, *a):
    """One tab item; the tab is always closed, and an exception is shown inside it (and recorded)."""
    sel, _ = imgui.begin_tab_item(label)
    if sel:
        try:
            fn(app, *a)
        except Exception:                                # noqa: BLE001
            tb = traceback.format_exc()
            if label not in app.errors:
                app.errors[label] = tb
                print(f"[ui] tab {label!r} failed:\n{tb}")
            imgui.text_colored(BAD, f"error: {tb.splitlines()[-1]}")
        finally:
            imgui.end_tab_item()


def tip(text):
    if imgui.is_item_hovered():
        imgui.set_tooltip(text)


# ====================================================================== dispatcher
def draw_all(app, ww, wh):
    imgui.dock_space_over_viewport(0, imgui.get_main_viewport(), imgui.DockNodeFlags_.passthru_central_node)
    handle_keys(app)
    safe(app, "menu", main_menu)
    fns = {"controls": controls, "params": params, "inspector": inspector, "stats": stats, "profiler": profiler,
           "policies": policies, "schedule": schedule, "events": events_panel, "checkpoints": checkpoints,
           "archive": archive, "eventlog": eventlog, "organism": organism, "tools": tools, "run": run_panel}
    for key, fn in fns.items():
        if not app.panels.get(key):
            continue
        if app.reset_windows and key in LAYOUT:
            x, y, w, h = LAYOUT[key]
            imgui.set_next_window_pos(V2(x * ww, y * wh), imgui.Cond_.always)
            imgui.set_next_window_size(V2(w * ww, h * wh), imgui.Cond_.always)
        else:
            imgui.set_next_window_size(V2(0.22 * ww, 0.4 * wh), imgui.Cond_.first_use_ever)
        exp, opened = imgui.begin(TITLES[key], True)
        if not opened:
            app.panels[key] = False
        if exp:
            safe(app, key, fn)
        imgui.end()
    app.reset_windows = False
    safe(app, "status", status_bar, ww, wh)
    if app.toast and time.time() < app.toast_until:
        imgui.set_next_window_pos(V2(ww * 0.5, wh * 0.06), imgui.Cond_.always, V2(0.5, 0))
        imgui.begin("##toast", None, imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.always_auto_resize |
                    imgui.WindowFlags_.no_inputs | imgui.WindowFlags_.no_nav)
        imgui.text(app.toast)
        imgui.end()


def safe(app, name, fn, *a):
    try:
        fn(app, *a)
    except Exception:                                    # noqa: BLE001
        tb = traceback.format_exc()
        if name not in app.errors:
            app.errors[name] = tb
            print(f"[ui] panel {name!r} failed:\n{tb}")
        imgui.text_colored(BAD, f"panel error: {tb.splitlines()[-1]}")


def handle_keys(app):
    io = imgui.get_io()
    if io.want_text_input:
        return
    K = imgui.Key
    if imgui.is_key_pressed(K.space, False):
        toggle_run(app)
    if imgui.is_key_pressed(K.period, False):
        app.running = False
        app.step(1)
    if imgui.is_key_pressed(K.n, False):
        app.running = False
        app.step(app.step_n)
    if imgui.is_key_pressed(K.f, False):
        app.follow = not app.follow
    if imgui.is_key_pressed(K.s, False) and not io.key_ctrl:
        app.screenshot_next = True
    if imgui.is_key_pressed(K.left_bracket, False):
        app.lineage_step(-1)
    if imgui.is_key_pressed(K.right_bracket, False):
        app.lineage_step(+1)
    if imgui.is_key_pressed(K.r, False):
        app.rewind_to(len(app.rewind) - 2 if len(app.rewind) > 1 else 0)
    if imgui.is_key_pressed(K.k, False) and app.selected_slot() is not None:
        app.sim.kill([app.selected_slot()])
    if imgui.is_key_pressed(K.escape, False):
        app.sel = []


def toggle_run(app):
    if app.running:
        app.running = False
        return
    if app.start_mode():
        app.running = True
        app.stop_reason = ""
        app.runner.pause_requested = None


def main_menu(app):
    if imgui.begin_main_menu_bar():
        if imgui.begin_menu("File"):
            if imgui.menu_item("Save snapshot (.evck)", "", False)[0]:
                os.makedirs("snapshots", exist_ok=True)
                p = os.path.join("snapshots", f"snap_{app.sim.tick:09d}.evck")
                snapshot.save_snapshot(app.sim, p, extra=app.runner.extra())
                app.notify("saved " + p)
            if imgui.menu_item("Start recording run directory", "", False)[0]:
                app.start_recording()
            if imgui.menu_item("Save config as preset...", "", False)[0]:
                app.panels["params"] = True
            if imgui.menu_item("Quit", "", False)[0]:
                import glfw
                glfw.set_window_should_close(app.window, True)
            imgui.end_menu()
        if imgui.begin_menu("View"):
            for k, t in TITLES.items():
                ch, v = imgui.menu_item(t, "", bool(app.panels.get(k)))
                if ch:
                    app.panels[k] = v
            imgui.separator()
            if imgui.menu_item("Debug layout", "", app.layout_name == "debug")[0]:
                app.set_layout("debug")
            if imgui.menu_item("Monitor layout (low render rate)", "", app.layout_name == "monitor")[0]:
                app.set_layout("monitor")
            if imgui.menu_item("Reset window positions", "", False)[0]:
                app.reset_windows = True
            imgui.end_menu()
        if imgui.begin_menu("Help"):
            imgui.text("Space play/pause  . step  n step N  f follow  s screenshot")
            imgui.text("[ ] parent/daughter  r rewind  k kill  Esc deselect")
            imgui.text("Left-drag pan, wheel zoom, click select, shift-click add")
            imgui.end_menu()
        imgui.end_main_menu_bar()


# ====================================================================== status bar
def status_bar(app, ww, wh):
    imgui.set_next_window_pos(V2(0, wh - 26), imgui.Cond_.always)
    imgui.set_next_window_size(V2(ww, 26), imgui.Cond_.always)
    flags = (imgui.WindowFlags_.no_decoration | imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_saved_settings |
             imgui.WindowFlags_.no_docking | imgui.WindowFlags_.no_bring_to_front_on_focus)
    imgui.begin("##status", None, flags)
    r, s = app.runner, app.sim
    st = s.last_stats or {}
    pop = s.population() if (s.tick % 10 == 0 or not app.running) else st.get("population", 0)
    el = time.time() - r.wall0
    tps = r.recent_tps() or 0.0
    eta = ""
    stop = r.stop_cfg
    if stop.get("max_ticks") and tps > 0:
        eta = f" | ETA {(stop['max_ticks'] - s.tick) / tps:.0f}s"
    elif stop.get("max_wall"):
        eta = f" | ETA {max(0, stop['max_wall'] - el):.0f}s"
    imgui.text(f"tick {s.tick} | {el:,.0f}s | {tps:.1f} t/s{eta} | pop {pop} | orgs {st.get('organisms', '-')}")
    imgui.same_line()
    if r.ckpt:
        nxt = max(0, r.ckpt_every_ticks - (s.tick - r._last_ckpt_tick)) if r.ckpt_every_ticks else None
        last = r.ckpt.last_result
        imgui.text(f"| ckpt next {nxt if nxt is not None else '-'} t, last "
                   f"{'ok' if last and last.get('ok') else ('FAIL' if last else '-')}")
    else:
        imgui.text_colored(DIM, "| not recording")
    imgui.same_line()
    iv = len([t for t in r.intervention_times if time.time() - t < 3600])
    imgui.text(f"| interventions/h {iv}")
    imgui.same_line()
    nw = len(r.warnings)
    imgui.text_colored(WARN if nw else DIM, f"| warnings {nw}")
    gt = (r.stats.hw_columns().get("gpu_temp_c") if r.stats and (s.tick % 100 == 0 or True) and r.stats._nvml else None)
    if gt:
        imgui.same_line()
        imgui.text(f"| GPU {gt:.0f}C")
    imgui.same_line()
    if app.stop_reason:
        imgui.text_colored(WARN, "| " + app.stop_reason)
    imgui.end()


# ====================================================================== run controls
def controls(app):
    r = app.runner
    col = GOOD if app.running else WARN
    if imgui.button("Pause" if app.running else "Play", V2(70, 0)):
        toggle_run(app)
    imgui.same_line()
    if imgui.button("Step 1"):
        app.running = False
        app.step(1)
    imgui.same_line()
    if imgui.button("Step N"):
        app.running = False
        app.step(app.step_n)
    imgui.same_line()
    imgui.set_next_item_width(80)
    app.step_n = imgui.input_int("##n", app.step_n, 0, 0)[1]
    imgui.text_colored(col, "running" if app.running else "paused")
    if app.stop_reason:
        imgui.same_line()
        imgui.text_colored(WARN, app.stop_reason)
    imgui.separator()
    imgui.text("Speed")
    _, app.uncapped = imgui.checkbox("Uncapped", app.uncapped)
    if not app.uncapped:
        imgui.set_next_item_width(-1)
        app.target_tps = imgui.slider_float("##tps", app.target_tps, 1.0, 600.0, "%.0f ticks/s", imgui.SliderFlags_.logarithmic)[1]
    else:
        imgui.set_next_item_width(-1)
        app.render_every = imgui.slider_int("render every k-th frame / ticks per frame", app.render_every, 1, 200)[1]
    _, app.render_on = imgui.checkbox("Render (off = fastest)", app.render_on)
    imgui.separator()
    imgui.text("Run mode (pauses when its condition is met)")
    imgui.set_next_item_width(-1)
    ch, app.run_mode = imgui.combo("##mode", app.run_mode, list(RUN_MODES))
    mp = app.mode_param
    if app.run_mode == 2:
        mp["daughter_policy"] = imgui.combo("daughter policy", mp["daughter_policy"], ["first daughter", "free daughter", "random"])[1]
    if app.run_mode == 5:
        mp["pop"] = imgui.input_float("population X", mp["pop"], 0, 0, "%.0f")[1]
    if app.run_mode == 7:
        mp["share"] = imgui.input_float("lineage share % Y", mp["share"], 0, 0, "%.0f")[1]
    if app.run_mode in (1, 2, 3, 4) and app.selected_slot() is None:
        imgui.text_colored(WARN, "select a cell in the viewport first")
    imgui.separator()
    _, app.epoch_on = imgui.checkbox("Epoch mode", app.epoch_on)
    if app.epoch_on:
        app.epoch_len = imgui.input_int("epoch length E (ticks)", app.epoch_len, 0, 0)[1]
        app.epoch_topk = imgui.input_int("keep top-K lineages", app.epoch_topk, 0, 0)[1]
        if imgui.button("Step 1 generation (to next epoch boundary)"):
            app.running = False
            r.advance(max(1, app.epoch_len - (app.sim.tick - app.last_epoch)))
            app.post_tick()


# ====================================================================== parameters
def _modified(app, name):
    return app.cfg.get(name) != PARAM_SPECS[name].default


def _param_widget(app, spec):
    name = spec.name
    pending = name in app.pending_restart_params
    cur = app.pending_restart_params[name] if pending else app.cfg.get(name)
    imgui.set_next_item_width(-1)
    v, changed = cur, False
    if spec.kind == "float":
        flags = imgui.SliderFlags_.logarithmic if (spec.log and spec.lo is not None and spec.lo >= 0) else 0
        lo, hi = spec.lo, spec.hi
        if spec.log and lo == 0:
            lo = hi * 1e-5
        changed, v = imgui.slider_float(f"##{name}", float(cur), float(lo), float(hi), "%.5g", flags)
    elif spec.kind == "int":
        if spec.hi > 100000:
            changed, v = imgui.input_int(f"##{name}", int(cur), 0, 0)
        else:
            changed, v = imgui.slider_int(f"##{name}", int(cur), int(spec.lo), int(spec.hi))
    elif spec.kind == "bool":
        changed, v = imgui.checkbox(f"##{name}", bool(cur))
    elif spec.kind == "enum":
        changed, i = imgui.combo(f"##{name}", spec.choices.index(cur), list(spec.choices))
        v = spec.choices[i]
    else:
        changed, v = imgui.input_text(f"##{name}", str(cur))
    if changed:
        if spec.kind in ("float", "int"):
            v = min(max(v, spec.lo), spec.hi)
        if spec.live:
            app.runner.set_param(name, v, "user")
        else:
            app.pending_restart_params[name] = v


def params(app):
    imgui.set_next_item_width(-1)
    app.param_search = imgui.input_text_with_hint("##search", "search parameters...", getattr(app, "param_search", ""))[1]
    q = app.param_search.lower().strip()
    if app.pending_restart_params:
        imgui.text_colored(WARN, f"{len(app.pending_restart_params)} restart-time change(s) pending")
        if imgui.button("Restart simulation with these"):
            app.restart_with_params()
        imgui.same_line()
        if imgui.button("Discard"):
            app.pending_restart_params = {}
    if imgui.begin_tab_bar("ptabs"):
        tab(app, "All", _params_all, q)
        tab(app, "Presets", _presets_tab)
        tab(app, "Point lights", _lights_tab)
        imgui.end_tab_bar()


def _params_all(app, q):
    favs = [n for n in PARAM_SPECS if n in app.fav]
    if favs and not q:
        imgui.text_colored(WARN, "Pinned")
        for n in favs:
            _param_row(app, PARAM_SPECS[n])
        imgui.separator()
    for g in GROUPS:
        items = [s for s in PARAM_SPECS.values() if s.group == g and
                 (not q or q in s.name.lower() or q in s.desc.lower())]
        if not items:
            continue
        if q:
            imgui.set_next_item_open(True)
        if imgui.collapsing_header(f"{g}  ({sum(_modified(app, s.name) for s in items)} modified)"):
            for s in items:
                _param_row(app, s)


def _param_row(app, spec):
    imgui.push_id(spec.name)
    mod = _modified(app, spec.name)
    pinned = spec.name in app.fav
    if imgui.small_button("*" if pinned else "+"):
        (app.fav.discard if pinned else app.fav.add)(spec.name)
    tip("pin to favourites")
    imgui.same_line()
    if mod:
        if imgui.small_button("R"):
            app.runner.set_param(spec.name, spec.default, "user") if spec.live else app.pending_restart_params.__setitem__(spec.name, spec.default)
        tip("reset to default")
        imgui.same_line()
    imgui.text_colored(WARN if mod else V4(1, 1, 1, 1), spec.name + ("" if spec.live else "  [restart]"))
    tip(f"{spec.desc}\nunit: {spec.unit or '-'}  default: {spec.default}  range: {spec.lo}..{spec.hi}")
    _param_widget(app, spec)
    imgui.pop_id()


def _presets_tab(app):
    files = sorted(f[:-5] for f in os.listdir(PRESET_DIR) if f.endswith(".json")) if os.path.isdir(PRESET_DIR) else []
    app.preset_idx = min(getattr(app, "preset_idx", 0), max(0, len(files) - 1))
    if files:
        app.preset_idx = imgui.combo("preset", app.preset_idx, files)[1]
        if imgui.button("Load (live params apply now)"):
            new = Config.load(os.path.join(PRESET_DIR, files[app.preset_idx] + ".json"))
            for k, spec in PARAM_SPECS.items():
                if new.get(k) != app.cfg.get(k):
                    if spec.live:
                        app.runner.set_param(k, new.get(k), "preset")
                    else:
                        app.pending_restart_params[k] = new.get(k)
            app.cfg.point_lights = list(new.point_lights)
        imgui.same_line()
        if imgui.button("Reset all to defaults"):
            for k, spec in PARAM_SPECS.items():
                if spec.live and app.cfg.get(k) != spec.default:
                    app.runner.set_param(k, spec.default, "user")
    app.save_name = imgui.input_text("save as", getattr(app, "save_name", "my_preset"))[1]
    if imgui.button("Save current config as preset"):
        os.makedirs(PRESET_DIR, exist_ok=True)
        app.cfg.save(os.path.join(PRESET_DIR, app.save_name + ".json"))
        app.notify(f"saved presets/{app.save_name}.json")
    imgui.separator()
    imgui.text("Diff two presets (A = left, B = right; 'current' = running config)")
    names = ["current"] + files
    app.diff_i = imgui.combo("A", getattr(app, "diff_i", 0), names)[1]
    app.diff_j = imgui.combo("B", getattr(app, "diff_j", min(1, len(names) - 1)), names)[1]

    def load(n):
        return app.cfg if n == "current" else Config.load(os.path.join(PRESET_DIR, n + ".json"))
    d = load(names[app.diff_i]).diff(load(names[app.diff_j]))
    imgui.text(f"{len(d)} difference(s)")
    for k, (a, b) in sorted(d.items()):
        imgui.text(f"{k}: {a}  ->  {b}")
    imgui.separator()
    imgui.text("Parameter change log (this session)")
    if app.runner.rd and os.path.exists(app.runner.rd.history_path):
        with open(app.runner.rd.history_path, encoding="utf-8") as f:
            lines = f.readlines()[-12:]
        for ln in reversed(lines):
            h = json.loads(ln)
            imgui.text_colored(DIM, f"t{h['tick']} {h['source']}: {h['param']} = {h['value']}")
    else:
        imgui.text_colored(DIM, "(start recording to keep config_history.jsonl)")


def _lights_tab(app):
    pl = app.cfg.point_lights
    if imgui.button("Add point light at cursor"):
        c = getattr(app, "cursor_world", app.mouse_world)
        pl.append({"x": float(c[0]), "y": float(c[1]), "I": 1.0, "vx": 0.0, "vy": 0.0})
        app.cfg.set("max_point_lights", app.cfg.max_point_lights)
    rm = None
    for i, L in enumerate(pl):
        imgui.push_id(i)
        imgui.text(f"light {i}")
        imgui.set_next_item_width(-1)
        L["x"] = imgui.drag_float("x", L["x"], 1.0)[1]
        L["y"] = imgui.drag_float("y", L["y"], 1.0)[1]
        L["I"] = imgui.drag_float("intensity", L["I"], 0.01, 0.0, 20.0)[1]
        L["vx"] = imgui.drag_float("vx (path)", L.get("vx", 0.0), 0.001)[1]
        L["vy"] = imgui.drag_float("vy (path)", L.get("vy", 0.0), 0.001)[1]
        if imgui.button("remove"):
            rm = i
        imgui.pop_id()
    if rm is not None:
        pl.pop(rm)
    app.cfg.set("ambient_I", app.cfg.ambient_I)


# ====================================================================== tools
def tools(app):
    if imgui.collapsing_header("View", imgui.TreeNodeFlags_.default_open):
        app.color_mode = imgui.combo("colour", app.color_mode, list(rnd.COLOR_MODES))[1]
        if rnd.COLOR_MODES[app.color_mode] == "specific trait":
            app.trait_index = imgui.combo("trait", app.trait_index, list(isa.TRAIT_NAMES))[1]
        app.overlay = imgui.combo("overlay", app.overlay, list(OVERLAYS))[1]
        if app.overlay == 4:
            app.overlay_chem = imgui.slider_int("channel", app.overlay_chem, 0, app.sim.state.dims.K - 1)[1]
        app.overlay_alpha = imgui.slider_float("overlay opacity", app.overlay_alpha, 0.0, 1.0)[1]
        _, app.tiled = imgui.checkbox("3x3 tiled torus", app.tiled)
        _, app.show_bonds = imgui.checkbox("bonds", app.show_bonds)
        _, app.show_regions = imgui.checkbox("regions", app.show_regions)
        _, app.follow = imgui.checkbox("follow selected (f)", app.follow)
        app.radius_scale = imgui.slider_float("cell size x", app.radius_scale, 0.3, 3.0)[1]
        if imgui.button("Fit world"):
            app.fit_pending = True
        imgui.same_line()
        if imgui.button("Screenshot (s)"):
            app.screenshot_next = True
    if imgui.collapsing_header("Selection & find", imgui.TreeNodeFlags_.default_open):
        if imgui.button("Select organism"):
            app.select_organism()
        imgui.same_line()
        if imgui.button("Center"):
            app.center_on_selected()
        imgui.same_line()
        if imgui.button("< parent"):
            app.lineage_step(-1)
        imgui.same_line()
        if imgui.button("daughter >"):
            app.lineage_step(1)
        for lbl, what in (("Largest organism", "largest organism"), ("Oldest cell", "oldest"),
                          ("Most common genome", "most common genome")):
            if imgui.button(lbl):
                app.find(what)
        app.find_uid = imgui.input_text("uid", getattr(app, "find_uid", "0"))[1]
        imgui.same_line()
        if imgui.button("Go##uid") and app.find_uid.strip().isdigit():
            app.find("uid", int(app.find_uid))
        app.find_lin = imgui.input_text("lineage", getattr(app, "find_lin", "1"))[1]
        imgui.same_line()
        if imgui.button("Random##lin") and app.find_lin.strip().isdigit():
            app.find("random of lineage", int(app.find_lin))
        imgui.text_colored(DIM, f"selected: {len(app.sel)} cell(s)")
    if imgui.collapsing_header("Brushes (left-drag in viewport)", imgui.TreeNodeFlags_.default_open):
        app.brush = imgui.combo("brush", app.brush, list(BRUSHES))[1]
        b = BRUSHES[app.brush]
        app.brush_radius = imgui.slider_float("radius", app.brush_radius, 2.0, 200.0)[1]
        if b.startswith("inject"):
            app.brush_k = imgui.input_int("cells per click", app.brush_k, 0, 0)[1]
        if b == "inject seed":
            app.brush_file = imgui.input_text("seed .asm", app.brush_file)[1]
        if b == "inject archive":
            imgui.text_colored(DIM, f"archive genome: {app.archive_view or '(pick one in the archive panel)'}")
        if b == "inject clipboard":
            imgui.text_colored(DIM, "assembles the .asm text currently on the clipboard")
        if b == "paint nutrients":
            app.brush_amount = imgui.slider_float("amount", app.brush_amount, 0.0, 2.0)[1]
        if b == "paint obstacles":
            imgui.text_colored(DIM, "hold Ctrl to erase")
    if imgui.collapsing_header("Rewind buffer"):
        app.rewind_every = imgui.input_int("snapshot every S ticks (0=off)", app.rewind_every, 0, 0)[1]
        app.rewind_max = imgui.input_int("keep R snapshots", app.rewind_max, 0, 0)[1]
        est = (app.rewind_bytes / max(1, len(app.rewind))) * app.rewind_max if app.rewind else 0
        imgui.text(f"{len(app.rewind)} stored, {app.rewind_bytes / 2**20:.1f} MiB; full buffer ~ {est / 2**20:.0f} MiB")
        if len(app.rewind) > 1:
            i = imgui.slider_int("jump to", len(app.rewind) - 1, 0, len(app.rewind) - 1)[1]
            imgui.text_colored(DIM, f"tick {app.rewind[i][0]}")
            if imgui.button("Rewind here (forks timeline)"):
                app.running = False
                app.rewind_to(i)
        if imgui.button("Take snapshot now"):
            app.push_rewind()
    if imgui.collapsing_header("Layouts"):
        if imgui.button("Debug layout"):
            app.set_layout("debug")
        imgui.same_line()
        if imgui.button("Monitor layout"):
            app.set_layout("monitor")
        imgui.same_line()
        if imgui.button("Reset positions"):
            app.reset_windows = True
        app.layout_save_name = imgui.input_text("name", getattr(app, "layout_save_name", "mine"))[1]
        if imgui.button("Save docking layout"):
            app.ui_state.setdefault("layouts", {})[app.layout_save_name] = {
                "ini": imgui.save_ini_settings_to_memory(), "panels": dict(app.panels)}
            app.save_ui_state()
        for n, L in list(app.ui_state.get("layouts", {}).items()):
            if imgui.button(f"load '{n}'"):
                imgui.load_ini_settings_from_memory(L["ini"])
                app.panels = dict(L["panels"])


# ====================================================================== inspector
def inspector(app):
    sl = app.selected_slot()
    if sl is None:
        imgui.text_colored(DIM, "Click a cell in the viewport to inspect it.")
        if app.sel:
            imgui.text_colored(WARN, f"cell uid {app.sel[0]} is no longer alive")
        return
    s = app.sim
    c = s.state.cells
    g = lambda n: app.cell_scalar(n, sl)
    uid = int(g("uid"))
    imgui.text_colored(GOOD, f"cell uid {uid}")
    imgui.same_line()
    if imgui.small_button("trace on"):
        s.set_traced([app.slot_of_uid(u) for u in app.sel if app.slot_of_uid(u) is not None][:8])
        app.apply_breakpoints()
    if imgui.begin_tab_bar("insp"):
        for name, fn in (("State", _insp_state), ("Sensors", _insp_sensors), ("Energy", _insp_energy),
                         ("Genome", _insp_genome), ("Trace", _insp_trace), ("Actions", _insp_actions),
                         ("Breakpoints", _insp_bp), ("Edit", _insp_edit), ("Sandbox", _insp_sandbox),
                         ("Diff", _insp_diff)):
            tab(app, name, fn, sl)
        imgui.end_tab_bar()


def _insp_state(app, sl):
    s = app.sim
    c = s.state.cells
    g = lambda n: app.cell_scalar(n, sl)
    gid = int(g("genome_id"))
    org = int(g("organism_id"))
    orgsize = int((c.organism_id.numpy()[app.frame["al"]] == org).sum()) if app.frame is not None else "?"
    imgui.text_colored(WARN, "Identity")
    kv("uid / parent", f"{int(g('uid'))} / {int(g('parent_uid'))}")
    kv("lineage", int(g("lineage_id")))
    kv("organism", f"{org}  ({orgsize} cells)")
    kv("division depth", int(g("division_depth")))
    kv("age / born", f"{int(g('age'))} / {int(g('birth_tick'))}")
    kv("origin", isa.ORIGIN_NAMES[int(g("origin")) % 4])
    kv("genome id / hash", f"{gid} / {int(s.state.genomes.hash[gid:gid + 1].numpy()[0]) & (2 ** 64 - 1):016x}")
    imgui.separator()
    imgui.text_colored(WARN, "State")
    p, v = app.cell_row("pos", sl), app.cell_row("vel", sl)
    kv("position", f"{p[0]:.2f}, {p[1]:.2f}")
    kv("velocity", f"{v[0]:.3f}, {v[1]:.3f}  |v|={np.hypot(*v):.3f}")
    kv("heading / spin", f"{float(g('angle')):.3f} rad / {float(g('ang_vel')):.4f}")
    kv("mass / energy", f"{float(g('mass')):.3f} / {float(g('energy')):.3f}")
    kv("state register", int(g("state")))
    kv("ip / wait", f"{int(g('ip'))} / {int(g('wait'))}")
    regs = app.cell_row("regs", sl)
    imgui.text("registers: " + "  ".join(f"r{i}={x:.3g}" for i, x in enumerate(regs)))
    imgui.separator()
    tr, tt = app.cell_row("traits", sl), app.cell_row("trait_targets", sl)
    if imgui.begin_table("traits", 3, imgui.TableFlags_.row_bg):
        for i, n in enumerate(isa.TRAIT_NAMES):
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text(n)
            imgui.table_next_column()
            imgui.progress_bar(float(tr[i]), V2(-1, 0), f"{tr[i]:.2f}")
            imgui.table_next_column()
            imgui.text(f"-> {tt[i]:.2f}")
        imgui.end_table()
    bonds = app.cell_row("bonds", sl)
    imgui.text_colored(WARN, "Bonds")
    other = lambda b: int(s.state.bonds.b[b:b + 1].numpy()[0]) if int(s.state.bonds.a[b:b + 1].numpy()[0]) == sl else int(s.state.bonds.a[b:b + 1].numpy()[0])
    any_b = False
    for k, b in enumerate(bonds):
        if b >= 0:
            any_b = True
            o = other(int(b))
            imgui.text(f"slot {k}: -> uid {int(app.cell_scalar('uid', o))}  strain {float(s.state.bonds.strain[int(b):int(b) + 1].numpy()[0]):+.3f}")
    if not any_b:
        imgui.text_colored(DIM, "none")
    so = app.cell_row("signal_out", sl)
    imgui.text("signal_out: " + "  ".join(f"S{i}={x:.3g}" for i, x in enumerate(so)))


def _insp_sensors(app, sl):
    names = isa.sensor_names(app.sim.state.dims.K)
    vals = app.sim.state.scratch.sensors[sl:sl + 1].numpy()[0]
    if imgui.begin_table("sens", 2, imgui.TableFlags_.row_bg | imgui.TableFlags_.scroll_y):
        for n, v in zip(names, vals):
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text(n)
            imgui.table_next_column()
            imgui.text(f"{v:.4f}")
        imgui.end_table()
    imgui.text_colored(DIM, "values as read by the VM on the last tick (with sensor noise)")


def _insp_energy(app, sl):
    eb = app.sim.state.scratch.eb[sl:sl + 1].numpy()[0]
    gains = sum(eb[:5])
    costs = sum(eb[5:])
    imgui.text_colored(GOOD, f"gains {gains:+.4f}")
    imgui.same_line()
    imgui.text_colored(BAD, f"costs {costs:.4f}")
    imgui.same_line()
    imgui.text(f"net {gains - costs:+.4f}")
    if imgui.begin_table("eb", 2, imgui.TableFlags_.row_bg):
        for i, n in enumerate(EB_NAMES):
            if eb[i] != 0:
                imgui.table_next_row()
                imgui.table_next_column()
                imgui.text_colored(GOOD if i < 5 else BAD, ("+ " if i < 5 else "- ") + n)
                imgui.table_next_column()
                imgui.text(f"{eb[i]:.5f}")
        imgui.end_table()


def _genome_lines(app, words, ip=None):
    d = app.sim.state.dims
    return assembler.disassemble(words, d.K, d.KS, d.MB, exact=True, addresses=True, ip=ip).splitlines()


def _insp_genome(app, sl):
    words, gid = app.genome_words(sl)
    ip = int(app.cell_scalar("ip", sl))
    imgui.text(f"genome {gid}: {len(words)} words;  ip = {ip}")
    lines = _genome_lines(app, words, ip)
    if imgui.begin_child("gen", V2(0, 0)):
        for i, ln in enumerate(lines):
            is_label = "LABEL" in ln
            if i == ip:
                imgui.text_colored(V4(1, 1, 0.3, 1), ln)
                if app.running is False:
                    imgui.set_scroll_here_y(0.5)
            elif is_label:
                imgui.text_colored(V4(0.5, 0.8, 1.0, 1), ln)
            else:
                imgui.text(ln)
        imgui.end_child()


def _insp_trace(app, sl):
    s = app.sim
    if sl not in s.traced:
        imgui.text_colored(WARN, "this cell is not traced (select with a click; up to 8 traced at once)")
        if imgui.button("Trace this cell"):
            s.set_traced([sl])
        return
    rows = s.read_trace(sl)
    d = s.state.dims
    imgui.text(f"last {len(rows)} executed instructions (ring of 64)")
    if imgui.begin_table("trace", 4, imgui.TableFlags_.row_bg | imgui.TableFlags_.scroll_y | imgui.TableFlags_.resizable):
        for h in ("tick", "ip", "instruction", "register changes"):
            imgui.table_setup_column(h)
        imgui.table_headers_row()
        prev = None
        for tick, ip, w, regs in rows:
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text(str(tick))
            imgui.table_next_column()
            imgui.text(str(ip))
            imgui.table_next_column()
            imgui.text(assembler.disassemble_word(w, d.K, d.KS, d.MB))
            imgui.table_next_column()
            if prev is not None:
                ch = [f"r{i}: {prev[i]:.3g}->{regs[i]:.3g}" for i in range(8) if prev[i] != regs[i]]
                imgui.text(", ".join(ch))
            prev = regs
        imgui.end_table()


def _insp_actions(app, sl):
    fl = int(app.cell_scalar("action_flags", sl))
    fa = int(app.cell_scalar("fail_flags", sl))
    imgui.text("Actions this tick")
    shown = False
    for bit, name in isa.ACTION_NAMES.items():
        if fl & bit:
            imgui.text_colored(GOOD, "  " + name)
            shown = True
    if not shown:
        imgui.text_colored(DIM, "  (none)")
    imgui.text("Failures")
    shown = False
    for bit, desc in isa.FAIL_DESCRIPTIONS.items():
        if fa & bit:
            imgui.text_colored(BAD, "  " + desc)
            shown = True
    if not shown:
        imgui.text_colored(DIM, "  (none)")
    nx = int(app.sim.state.scratch.n_exec[sl:sl + 1].numpy()[0])
    imgui.text(f"instructions executed: {nx}")


def _insp_bp(app, sl):
    b = app.bp
    ops = ["(off)"] + isa.MNEMONICS
    ch, i = imgui.combo("opcode", b["opcode"] + 1, ops)
    b["opcode"] = i - 1
    b["addr"] = imgui.input_int("address (-1 off)", b["addr"], 1, 10)[1]
    b["energy"] = imgui.input_float("energy < (0 off)", b["energy"], 0, 0, "%.2f")[1]
    _, b["bitten"] = imgui.checkbox("pause when bitten", b["bitten"])
    _, b["divide"] = imgui.checkbox("pause when cell issues a split", b["divide"])
    if imgui.button("Apply breakpoints"):
        app.apply_breakpoints()
    imgui.text_colored(DIM, "Breakpoints stop the run after the tick in which they occur (ticks are atomic).")
    if app.stop_reason:
        imgui.text_colored(WARN, "last stop: " + app.stop_reason)


def _insp_edit(app, sl):
    s = app.sim
    if imgui.button("Kill"):
        s.kill([sl])
    imgui.same_line()
    if imgui.button("Clone in place"):
        s.clone_cells([sl], np.random.default_rng(), 0.0, "near", offset=2.0)
    imgui.same_line()
    if imgui.button("Mutate now"):
        app.mutate_now(sl)
    app.edit_path = imgui.input_text("genome file", getattr(app, "edit_path", "genome_export.asm"))[1]
    if imgui.button("Export genome (.asm)"):
        words, gid = app.genome_words(sl)
        d = s.state.dims
        assembler.save_asm(app.edit_path, words, header=f"exported from cell uid {int(app.cell_scalar('uid', sl))}",
                           k_chem=d.K, k_sig=d.KS, max_bonds=d.MB)
        app.notify("wrote " + app.edit_path)
    imgui.same_line()
    if imgui.button("Import into this cell"):
        d = s.state.dims
        app.set_cell_genome(sl, assembler.load_genome(app.edit_path, k_chem=d.K, k_sig=d.KS, max_bonds=d.MB))
        app.notify("genome replaced")
    if imgui.button("Inject at cursor"):
        d = s.state.dims
        w = assembler.load_genome(app.edit_path, k_chem=d.K, k_sig=d.KS, max_bonds=d.MB)
        s.inject(1, [w], np.random.default_rng(), placement="near", center=tuple(getattr(app, "cursor_world", app.mouse_world)), radius=5.0)
    imgui.separator()
    imgui.text("Set register / state")
    app.edit_reg = imgui.slider_int("register", getattr(app, "edit_reg", 0), 0, 7)[1]
    app.edit_val = imgui.input_float("value", getattr(app, "edit_val", 0.0))[1]
    if imgui.button("Set register"):
        a = s.state.cells.regs
        arr = a.numpy()
        arr[sl, app.edit_reg] = app.edit_val
        a.assign(arr)
    imgui.same_line()
    if imgui.button("Set state"):
        app.poke("state", sl, int(app.edit_val))
    if imgui.button("Set energy"):
        app.poke("energy", sl, float(app.edit_val))


def _insp_sandbox(app, sl):
    s = app.sim
    d = s.state.dims
    words, gid = app.genome_words(sl)
    if app.sandbox is None or app.sandbox.get("gid") != gid:
        app.sandbox = {"gid": gid, "vm": None, "sensors": np.zeros(d.NS, np.float32), "log": [], "tick": 0}
    sb = app.sandbox
    imgui.text("Runs this genome on the CPU reference interpreter with scripted sensor inputs.")
    if imgui.button("Reset sandbox"):
        sb["vm"] = CellVM(words=words, regs=app.cell_row("regs", sl).copy(), ip=int(app.cell_scalar("ip", sl)),
                          uid=int(app.cell_scalar("uid", sl)), signal_out=np.zeros(d.KS, np.float32))
        sb["tick"], sb["log"] = 0, []
    imgui.same_line()
    if imgui.button("Step 1 tick") and sb["vm"] is not None:
        inp = VMInputs(sb["sensors"], np.zeros(d.KS, np.float32), np.zeros((d.MB, d.KS), np.float32),
                       __import__("evosim.rng", fromlist=["x"]).tick_key(s.cfg.seed, s.cfg.rng_salt, sb["tick"]))
        trace = []
        act = run_tick(sb["vm"], inp, s.cfg.instr_per_tick, s.cfg.max_wait, d.K, d.KS, d.MB, trace=trace, tick=sb["tick"])
        sb["tick"] += 1
        sb["log"].append((sb["tick"], act, trace))
    imgui.same_line()
    if imgui.button("Step 1 instruction") and sb["vm"] is not None:
        trace = []
        run_tick(sb["vm"], VMInputs(sb["sensors"], np.zeros(d.KS, np.float32), np.zeros((d.MB, d.KS), np.float32), 1),
                 1, s.cfg.max_wait, d.K, d.KS, d.MB, trace=trace, tick=sb["tick"])
        sb["log"].append((sb["tick"], None, trace))
    if sb["vm"] is not None:
        vm = sb["vm"]
        imgui.text(f"ip {vm.ip}  state {vm.state}  regs: " + " ".join(f"{x:.3g}" for x in vm.regs))
        for tick, act, tr in sb["log"][-6:][::-1]:
            if act is not None:
                names = isa.describe_flags(act.flags, act.fail)
                imgui.text_colored(GOOD, f"tick {tick}: {act.n_exec} instr; " + ", ".join(names))
            for (_, ip, w, before, after) in tr[-8:]:
                imgui.text_colored(DIM, f"   {ip:3d} {assembler.disassemble_word(w, d.K, d.KS, d.MB)}")
    imgui.separator()
    imgui.text("Scripted sensors")
    names = isa.sensor_names(d.K)
    if imgui.begin_child("sens_in", V2(0, 160)):
        for i, n in enumerate(names):
            sb["sensors"][i] = imgui.drag_float(n, float(sb["sensors"][i]), 0.05)[1]
        imgui.end_child()


def _insp_diff(app, sl):
    g = app.sim.state.genomes
    words, gid = app.genome_words(sl)
    pg = int(g.parent[gid:gid + 1].numpy()[0])
    if pg < 0 or int(g.in_use[pg:pg + 1].numpy()[0]) == 0:
        imgui.text_colored(DIM, "this genome has no recorded (living) parent genome")
        return
    n = int(g.length[pg:pg + 1].numpy()[0])
    pw = [int(x) & 0xFFFFFFFF for x in g.words[pg:pg + 1].numpy()[0][:n]]
    d = app.sim.state.dims
    a = [l.split(": ", 1)[-1] for l in assembler.disassemble(pw, d.K, d.KS, d.MB, exact=True).splitlines()]
    b = [l.split(": ", 1)[-1] for l in assembler.disassemble(words, d.K, d.KS, d.MB, exact=True).splitlines()]
    imgui.text(f"parent genome {pg} ({len(pw)} words)  vs  this genome {gid} ({len(words)} words)")
    nd = 0
    if imgui.begin_child("diff", V2(0, 0)):
        for ln in difflib.unified_diff(a, b, "parent", "daughter", lineterm="", n=2):
            if ln.startswith("+") and not ln.startswith("+++"):
                imgui.text_colored(GOOD, ln)
                nd += 1
            elif ln.startswith("-") and not ln.startswith("---"):
                imgui.text_colored(BAD, ln)
                nd += 1
            else:
                imgui.text_colored(DIM, ln)
        if nd == 0:
            imgui.text_colored(DIM, "identical")
        imgui.end_child()


# ====================================================================== organism view
def organism(app):
    sl = app.selected_slot()
    if sl is None or app.frame is None:
        imgui.text_colored(DIM, "select a cell")
        return
    s = app.sim
    org = int(app.cell_scalar("organism_id", sl))
    f = app.frame
    idx = np.nonzero(f["organism"] == org)[0]
    imgui.text(f"organism {org}: {len(idx)} cell(s)")
    if len(idx) == 0:
        return
    W = np.array(s.state.dims.world)
    ref = f["pos"][idx[0]]
    rel = f["pos"][idx] - ref
    rel -= W * np.round(rel / W)
    avail = imgui.get_content_region_avail()
    origin = imgui.get_cursor_screen_pos()
    size = max(80.0, min(avail.x, avail.y) - 4)
    ext = max(float(np.abs(rel).max()) + float(f["rad"][idx].max()), 1.0)
    k = size / (2 * ext)
    cen = V2(origin.x + size / 2, origin.y + size / 2)
    dl = imgui.get_window_draw_list()
    dl.add_rect_filled(origin, V2(origin.x + size, origin.y + size), imgui.get_color_u32(V4(0.03, 0.03, 0.05, 1)))
    pos_of = {int(f["al"][i]): V2(cen.x + rel[j][0] * k, cen.y - rel[j][1] * k) for j, i in enumerate(idx)}
    b = s.state.bonds
    al = b.alive.numpy() != 0
    ba, bb, st_ = b.a.numpy()[al], b.b.numpy()[al], b.strain.numpy()[al]
    t = time.time()
    sig = s.state.cells.signal_out.numpy()
    for x, y, sr in zip(ba, bb, st_):
        if int(x) in pos_of and int(y) in pos_of:
            u = min(1.0, abs(float(sr)) / max(app.cfg.break_strain, 1e-6))
            dl.add_line(pos_of[int(x)], pos_of[int(y)], imgui.get_color_u32(V4(0.4 + 0.6 * u, 0.85 - 0.6 * u, 0.9 - 0.8 * u, 1)), 2.0)
            act = max(abs(float(sig[int(x), 0])), abs(float(sig[int(y), 0])))
            if act > 1e-3:                                           # animate a pulse travelling along the bond
                ph = (t * 2.0) % 1.0
                p0, p1 = pos_of[int(x)], pos_of[int(y)]
                dl.add_circle_filled(V2(p0.x + (p1.x - p0.x) * ph, p0.y + (p1.y - p0.y) * ph), 3.0,
                                     imgui.get_color_u32(V4(1, 1, 0.3, 1)))
    scol = rnd._STATE_COLORS
    for j, i in enumerate(idx):
        p = pos_of[int(f["al"][i])]
        c = scol[int(f["state"][i]) % len(scol)]
        dl.add_circle_filled(p, max(2.0, float(f["rad"][i]) * k), imgui.get_color_u32(V4(c[0], c[1], c[2], 1)))
        if int(f["uid"][i]) in app.sel:
            dl.add_circle(p, max(3.0, float(f["rad"][i]) * k) + 2, imgui.get_color_u32(V4(1, 1, 0.2, 1)), 24, 2.0)
    imgui.dummy(V2(size, size))
    imgui.text_colored(DIM, "colour = state register; bonds coloured by strain; yellow dots = SIGNAL S0 activity")


# ====================================================================== stats
def _series(app, key):
    xs = np.array([h["tick"] for h in app.history], np.float64)
    ys = np.array([h.get(key, np.nan) if h.get(key) is not None else np.nan for h in app.history], np.float64)
    return xs, ys


def _plot(app, title, keys, h=140):
    if implot.begin_plot(title, V2(-1, h)):
        implot.setup_axes("tick", "")
        for k in keys:
            xs, ys = _series(app, k)
            if len(xs):
                implot.plot_line(k, xs, ys)
        implot.end_plot()


def stats(app):
    if not app.history:
        imgui.text_colored(DIM, "no samples yet - run a few hundred ticks (stats_interval ticks apart)")
    if imgui.begin_tab_bar("stabs"):
        for name, fn in (("Population", _tab_pop), ("Organisms", _tab_org), ("Traits", _tab_traits),
                         ("Genomes", _tab_genomes), ("Resources & ledger", _tab_resources),
                         ("Lineages", _tab_lineages), ("Health", _tab_health)):
            tab(app, name, fn)
        imgui.end_tab_bar()
    if imgui.button("Export stats CSV"):
        os.makedirs("exports", exist_ok=True)
        keys = sorted({k for h in app.history for k in h})
        p = os.path.join("exports", f"stats_{app.sim.tick}.csv")
        with open(p, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for h in app.history:
                f.write(",".join("" if h.get(k) is None else str(h.get(k)) for k in keys) + "\n")
        app.notify("wrote " + p)


def _tab_pop(app):
    _plot(app, "population / organisms", ["population", "organisms", "components"])
    _plot(app, "births & deaths (per tick)", ["births", "deaths"])


def _tab_org(app):
    _plot(app, "organism size (cells)", ["org_size_mean", "org_size_max"])
    _plot(app, "max division depth", ["max_depth"])


def _tab_traits(app):
    _plot(app, "trait means", [f"trait_{n}" for n in isa.TRAIT_NAMES], 200)
    if app.frame is not None and len(app.frame["al"]):
        tr = app.frame["traits"]
        if imgui.collapsing_header("Trait histograms"):
            for t, n in enumerate(isa.TRAIT_NAMES):
                if implot.begin_plot(n, V2(-1, 80)):
                    implot.plot_histogram(n, tr[:, t].astype(np.float64), 20, 1.0, implot.Range(0, 1))
                    implot.end_plot()


def _tab_genomes(app):
    _plot(app, "genome length", ["genome_len_mean", "genome_len_max"])
    _plot(app, "distinct genomes / lineages", ["distinct_genomes", "distinct_lineages", "genome_table_used"])
    if app.frame is not None and len(app.frame["al"]):
        gl = app.sim.state.genomes.length.numpy()[app.frame["gid"]].astype(np.float64)
        if implot.begin_plot("genome length distribution", V2(-1, 120)):
            implot.plot_histogram("length", gl, 30)
            implot.end_plot()


def _tab_resources(app):
    _plot(app, "biomass & field nutrient", ["biomass", "nutrient_total", "energy_total"])
    if app.ledger_hist:
        xs = np.array([t for t, _, _ in app.ledger_hist], np.float64)
        if implot.begin_plot("energy ledger terms (last measured tick)", V2(-1, 190)):
            for i, n in enumerate(EB_NAMES):
                ys = np.array([ct[n] for _, ct, _ in app.ledger_hist], np.float64)
                if np.abs(ys).max() > 0:
                    implot.plot_line(("+" if i < 5 else "-") + n, xs, ys)
            implot.end_plot()
        if implot.begin_plot("mass / nutrient ledger terms", V2(-1, 140)):
            for n in LEDGER_NAMES:
                ys = np.array([gt[n] for _, _, gt in app.ledger_hist], np.float64)
                if np.abs(ys).max() > 0:
                    implot.plot_line(n, xs, ys)
            implot.end_plot()
    _plot(app, "ledger relative error (must stay ~0)", ["ledger_err_energy", "ledger_err_mass", "ledger_err_nutrient"], 100)
    r = app.sim.last_ledger
    if r:
        imgui.text_colored(GOOD if r["ok"] else BAD, f"last ledger check (tick {r['tick']}): {'balanced' if r['ok'] else 'IMBALANCE'}")
    for k in ("interv_inject", "interv_clone", "interv_cull", "origin_natural", "origin_injected", "origin_cloned"):
        if app.history and k in app.history[-1]:
            kv(k, app.history[-1][k])


def _tab_lineages(app):
    if not app.lineage_hist:
        imgui.text_colored(DIM, "no data yet")
        return
    tot = {}
    for _, d in app.lineage_hist:
        for k, v in d.items():
            tot[k] = max(tot.get(k, 0), v)
    top = [k for k, _ in sorted(tot.items(), key=lambda kv_: -kv_[1])[:10]]
    xs = np.array([t for t, _ in app.lineage_hist], np.float64)
    acc = np.zeros(len(xs))
    if implot.begin_plot("top lineages (stacked)", V2(-1, 260)):
        implot.setup_axes("tick", "cells")
        for k in top:
            ys = np.array([d.get(k, 0) for _, d in app.lineage_hist], np.float64)
            implot.plot_shaded(f"lineage {k}", xs, acc.copy(), acc + ys)
            acc = acc + ys
        implot.end_plot()


def _tab_health(app):
    _plot(app, "velocity / spin clamps per sample", ["v_clamps", "w_clamps"])
    _plot(app, "ticks/s", ["tps"])
    _plot(app, "bonds broken / split fails (capacity)", ["bonds_broken", "split_fail_capacity"])


# ====================================================================== profiler
def profiler(app):
    s = app.sim
    if s.cfg.profile_interval == 0:
        imgui.text_colored(WARN, "profiling is off (it synchronises the GPU each phase)")
        if imgui.button("Enable (every 10 ticks)"):
            app.runner.set_param("profile_interval", 10, "user")
        return
    if imgui.button("Disable profiling"):
        app.runner.set_param("profile_interval", 0, "user")
    tot = sum(s.timings.values())
    imgui.text(f"sum {tot:.3f} ms/tick  ->  max ~{1000 / max(tot, 1e-6):.0f} ticks/s")
    for k, v in s.timings.items():
        imgui.progress_bar(v / max(tot, 1e-9), V2(-1, 0), f"{k}  {v:.3f} ms  ({100 * v / max(tot, 1e-9):.0f}%)")
    imgui.separator()
    st = s.state
    imgui.text(f"device: {s.device}")
    imgui.text(f"state memory: {st.vram_bytes() / 2**20:.0f} MiB (capacity {st.dims.N} cells)")
    imgui.text(f"ticks/s (recent): {app.runner.recent_tps() or 0:.1f}")


# ====================================================================== policies
def policies(app):
    eng = app.runner.engine
    imgui.set_next_item_width(100)
    eng.eval_interval = imgui.input_int("eval every N ticks", eng.eval_interval, 0, 0)[1]
    imgui.text("Add built-in preset")
    imgui.set_next_item_width(200)
    app.pol_preset = imgui.combo("##pp", getattr(app, "pol_preset", 1), list(BUILTIN_NAMES))[1]
    imgui.same_line()
    if imgui.button("Add"):
        for p in builtin(BUILTIN_NAMES[app.pol_preset], target=max(100, app.sim.state.dims.N // 4),
                         files=["seeds/seed_replicator.asm"]):
            eng.add(p)
    imgui.same_line()
    if imgui.button("Clear all"):
        eng.set_policies([])
    app.pol_path = imgui.input_text("policy set file", getattr(app, "pol_path", "policies.json"))[1]
    if imgui.button("Save set"):
        eng.save(app.pol_path)
        app.notify("saved " + app.pol_path)
    imgui.same_line()
    if imgui.button("Load set"):
        from ..run.policies import PolicyEngine
        app.runner.engine = PolicyEngine.load(app.pol_path)
        eng = app.runner.engine
    imgui.separator()
    if not eng.policies:
        imgui.text_colored(DIM, "no policies (equivalent to the 'none' preset: clean experiment)")
    if imgui.begin_table("pol", 7, imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_v | imgui.TableFlags_.resizable):
        for h in ("on", "name", "trigger", "live values", "last / count", "", ""):
            imgui.table_setup_column(h)
        imgui.table_headers_row()
        for i, p in enumerate(eng.policies):
            imgui.push_id(i)
            imgui.table_next_row()
            imgui.table_next_column()
            p.enabled = imgui.checkbox("##on", p.enabled)[1]
            imgui.table_next_column()
            if imgui.selectable(p.name, app.policy_edit.get("i") == i)[0]:
                app.policy_edit = {"i": i, "text": json.dumps(p.to_dict(), indent=1)}
            imgui.table_next_column()
            imgui.text_wrapped(json.dumps(p.trigger))
            imgui.table_next_column()
            imgui.text_wrapped(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.last_values.items()}, default=str))
            imgui.table_next_column()
            imgui.text(f"t{p.last_tick if p.last_tick > -10**17 else '-'} x{p.firings}")
            imgui.table_next_column()
            if imgui.small_button("fire now"):
                eng.fire(p, i, app.runner, {"manual": True}, manual=True)
            imgui.table_next_column()
            if imgui.small_button("x"):
                eng.policies.pop(i)
                imgui.pop_id()
                break
            imgui.pop_id()
        imgui.end_table()
    pe = app.policy_edit
    if pe:
        imgui.separator()
        imgui.text(f"edit policy #{pe['i']} (JSON; trigger / actions / cooldown / limits / priority / exclusive)")
        pe["text"] = imgui.input_text_multiline("##pedit", pe["text"], V2(-1, 150))[1]
        if imgui.button("Apply edit"):
            try:
                eng.policies[pe["i"]] = Policy.from_dict(json.loads(pe["text"]))
                app.notify("policy updated")
            except Exception as e:                           # noqa: BLE001
                app.notify(f"invalid policy: {e}")
    if imgui.button("New policy from template"):
        eng.add(Policy("new_policy", {"type": "population_below", "n": 500},
                       [{"type": "log_event", "message": "population low"}]))
    if eng.fire_log:
        imgui.separator()
        imgui.text("recent firings")
        for r in list(eng.fire_log)[-6:][::-1]:
            imgui.text_colored(DIM, f"t{r['tick']} {r['policy']} {r['trigger_values']}")


# ====================================================================== schedule
def schedule(app):
    sc = app.runner.schedule
    specs = [n for n, s in PARAM_SPECS.items() if s.kind in ("float", "int", "bool") and s.live]
    app.sched_i = specs.index(app.sched_param) if app.sched_param in specs else 0
    ch, i = imgui.combo("parameter", app.sched_i, specs)
    app.sched_param = specs[i]
    p = app.sched_param
    ps = sc.items.get(p)
    imgui.text(f"current value: {app.cfg.get(p)}   (scheduled parameters are overwritten by the schedule every tick)")
    imgui.separator()
    imgui.text("Add keyframe")
    k = app.__dict__.setdefault("kf_new", {"tick": 1000, "value": float(app.cfg.get(p)), "ramp": 0, "kind": 0})
    k["tick"] = imgui.input_int("at tick", k["tick"], 0, 0)[1]
    k["value"] = imgui.input_float("value", k["value"], 0, 0, "%.5g")[1]
    k["ramp"] = imgui.input_int("ramp over (ticks)", k["ramp"], 0, 0)[1]
    k["kind"] = imgui.combo("ramp", k["kind"], ["linear", "smooth"])[1]
    if imgui.button("Add keyframe"):
        sc.add_keyframe(p, k["tick"], k["value"], k["ramp"], ("linear", "smooth")[k["kind"]])
    imgui.text("Add periodic modulator (added on top)")
    m = app.__dict__.setdefault("mod_new", {"wave": 0, "period": 5000.0, "amp": 0.1, "phase": 0.0})
    m["wave"] = imgui.combo("wave", m["wave"], ["sine", "square", "triangle"])[1]
    m["period"] = imgui.input_float("period (ticks)", m["period"], 0, 0, "%.0f")[1]
    m["amp"] = imgui.input_float("amplitude", m["amp"], 0, 0, "%.5g")[1]
    m["phase"] = imgui.input_float("phase (fraction)", m["phase"], 0, 0, "%.3f")[1]
    if imgui.button("Add modulator"):
        sc.add_modulator(p, ("sine", "square", "triangle")[m["wave"]], m["period"], m["amp"], m["phase"])
    if ps:
        imgui.separator()
        for j, kf in enumerate(ps.keyframes):
            imgui.text(f"t={kf.tick}  -> {kf.value:g}  ramp {kf.ramp} ({kf.kind})")
            imgui.same_line()
            if imgui.small_button(f"x##k{j}"):
                ps.keyframes.pop(j)
                break
        for j, md in enumerate(ps.modulators):
            imgui.text(f"{md.wave} period {md.period:g} amp {md.amplitude:g} phase {md.phase:g}")
            imgui.same_line()
            if imgui.small_button(f"x##m{j}"):
                ps.modulators.pop(j)
                break
        horizon = max(app.sim.tick * 2, max([kf.tick + kf.ramp for kf in ps.keyframes] + [1000]) * 1.2,
                      3 * max([md.period for md in ps.modulators] + [0]))
        xs = np.linspace(0, horizon, 600)
        init = sc.initial.get(p, float(app.cfg.get(p)))
        ys = np.array([ps.value(float(t), init) for t in xs])
        if implot.begin_plot("timeline", V2(-1, 180)):
            implot.setup_axes("tick", p)
            implot.plot_line("value(t)", xs, ys)
            implot.plot_inf_lines("now", np.array([float(app.sim.tick)]))
            implot.end_plot()
    if imgui.button("Save schedule.json") and app.runner.rd:
        app.runner.save_control_files()
        app.notify("saved")
    imgui.same_line()
    if imgui.button("Clear this parameter") and p in sc.items:
        del sc.items[p]


# ====================================================================== events
def events_panel(app):
    es = app.runner.events
    sim = app.sim
    c = getattr(app, "cursor_world", app.mouse_world)
    imgui.text_colored(DIM, f"cursor in viewport: {c[0]:.0f}, {c[1]:.0f}")
    for t in envmod.EVENT_TYPES:
        cfg = es.configs[t]
        imgui.push_id(t)
        if imgui.collapsing_header(t):
            cfg.enabled = imgui.checkbox("random events enabled", cfg.enabled)[1]
            cfg.mean_interval = imgui.input_float("mean interval (ticks, Poisson)", cfg.mean_interval, 0, 0, "%.0f")[1]
            cfg.mag_lo = imgui.input_float("magnitude min", cfg.mag_lo, 0, 0, "%.3f")[1]
            cfg.mag_hi = imgui.input_float("magnitude max", cfg.mag_hi, 0, 0, "%.3f")[1]
            cfg.size_lo = imgui.input_float("region radius min", cfg.size_lo, 0, 0, "%.0f")[1]
            cfg.size_hi = imgui.input_float("region radius max", cfg.size_hi, 0, 0, "%.0f")[1]
            cfg.duration = imgui.input_int("duration (ticks)", cfg.duration, 0, 0)[1]
            if imgui.button("Trigger now at cursor"):
                es.trigger(sim, t, app.runner.emit, source="manual", at=(float(c[0]), float(c[1])))
        imgui.pop_id()
    imgui.separator()
    imgui.text("Named regions (used by events, policies, brushes)")
    rn = app.region_new
    rn["name"] = imgui.input_text("name", rn["name"])[1]
    rn["shape"] = ["rect", "circle"][imgui.combo("shape", ["rect", "circle"].index(rn["shape"]), ["rect", "circle"])[1]]
    rn["x"] = imgui.input_float("x", rn["x"], 0, 0, "%.0f")[1]
    rn["y"] = imgui.input_float("y", rn["y"], 0, 0, "%.0f")[1]
    rn["w"] = imgui.input_float("w / radius", rn["w"], 0, 0, "%.0f")[1]
    rn["h"] = imgui.input_float("h", rn["h"], 0, 0, "%.0f")[1]
    if imgui.button("Add / replace region"):
        app.runner.regions[rn["name"]] = envmod.Region(**rn)
    for n, r in list(app.runner.regions.items()):
        imgui.text(f"{n}: {r.shape} ({r.x:.0f},{r.y:.0f}) {r.w:.0f}x{r.h:.0f}")
        imgui.same_line()
        if imgui.small_button(f"x##{n}"):
            del app.runner.regions[n]
    if es.active:
        imgui.separator()
        for a in es.active:
            imgui.text_colored(WARN, f"active: {a['type']} until tick {a['end']}")


# ====================================================================== checkpoints
def checkpoints(app):
    r = app.runner
    if not r.ckpt:
        imgui.text_colored(WARN, "not recording: checkpoints need a run directory")
        if imgui.button("Start recording (creates runs/<time>_ui)"):
            app.start_recording()
        app.snap_path = imgui.input_text("snapshot file", getattr(app, "snap_path", "snapshot.evck"))[1]
        if imgui.button("Save snapshot file"):
            snapshot.save_snapshot(app.sim, app.snap_path, extra=r.extra())
            app.notify("saved " + app.snap_path)
        imgui.same_line()
        if imgui.button("Load snapshot file"):
            app.load_checkpoint_inplace(app.snap_path)
        return
    r.ckpt_every_ticks = imgui.input_int("auto every N ticks (0=off)", r.ckpt_every_ticks, 0, 0)[1]
    r.ckpt_every_seconds = imgui.input_float("auto every T seconds (0=off)", float(r.ckpt_every_seconds), 0, 0, "%.0f")[1]
    if imgui.button("Checkpoint now"):
        r.write_checkpoint(reason="manual", block=True)
    imgui.same_line()
    imgui.text_colored(DIM, f"dir {r.rd.size_bytes() / 2**20:.1f} MiB")
    if imgui.begin_table("ck", 6, imgui.TableFlags_.row_bg | imgui.TableFlags_.scroll_y):
        for h in ("tick", "wall", "pop", "status", "size", "actions"):
            imgui.table_setup_column(h)
        imgui.table_headers_row()
        for m in reversed(r.ckpt.list()[-60:]):
            imgui.push_id(m["path"])
            imgui.table_next_row()
            imgui.table_next_column()
            imgui.text(str(m.get("tick")))
            imgui.table_next_column()
            imgui.text(time.strftime("%m-%d %H:%M:%S", time.localtime(m.get("wall_time", 0))))
            imgui.table_next_column()
            imgui.text(str(m.get("population", "?")))
            imgui.table_next_column()
            imgui.text_colored(GOOD if m.get("good") else BAD, ("good" if m.get("good") else "BAD") + (" *keep" if m.get("keep") else ""))
            imgui.table_next_column()
            imgui.text(f"{m.get('size', 0) / 1024:.0f}K")
            imgui.table_next_column()
            if imgui.small_button("load"):
                app.load_checkpoint_inplace(m["path"])
            imgui.same_line()
            if imgui.small_button("fork"):
                app.fork_checkpoint(m["path"])
            imgui.same_line()
            if imgui.small_button("keep"):
                r.ckpt.mark_keep(m["path"], not m.get("keep"))
            imgui.same_line()
            if imgui.small_button("del"):
                r.ckpt.delete(m["path"])
            imgui.pop_id()
        imgui.end_table()


# ====================================================================== archive
def archive(app):
    ar = app.runner.archive
    if ar is None:
        imgui.text_colored(WARN, "the genome archive lives in the run directory: start recording first")
        if imgui.button("Start recording"):
            app.start_recording()
        return
    imgui.text(f"{ar.count()} archived genomes")
    if imgui.button("Archive notable genomes now"):
        ar.add_from_sim(app.sim)
    imgui.same_line()
    app.imp_path = imgui.input_text("import archive", getattr(app, "imp_path", ""))[1]
    imgui.same_line()
    if imgui.button("Import") and os.path.exists(app.imp_path):
        app.notify(f"imported {ar.import_archive(app.imp_path)} genomes")
    rows = ar.list(200)
    if imgui.begin_table("arch", 5, imgui.TableFlags_.row_bg | imgui.TableFlags_.scroll_y, V2(0, 160)):
        for h in ("hash", "len", "peak", "lineage", "tags"):
            imgui.table_setup_column(h)
        imgui.table_headers_row()
        for h, ln, ft, peak, lin, ph, tags, tm in rows:
            imgui.table_next_row()
            imgui.table_next_column()
            if imgui.selectable(h[:10], app.archive_view == h, imgui.SelectableFlags_.span_all_columns)[0]:
                app.archive_view = h
            imgui.table_next_column()
            imgui.text(str(ln))
            imgui.table_next_column()
            imgui.text(str(peak))
            imgui.table_next_column()
            imgui.text(str(lin))
            imgui.table_next_column()
            imgui.text(tags)
        imgui.end_table()
    if app.archive_view:
        w = ar.get(app.archive_view)
        if w:
            d = app.sim.state.dims
            txt = assembler.disassemble(w, d.K, d.KS, d.MB, exact=True)
            if imgui.button("Copy to clipboard"):
                imgui.set_clipboard_text(txt)
                app.notify("copied .asm")
            imgui.same_line()
            if imgui.button("Inject 10 at cursor"):
                app.sim.inject(10, [w], np.random.default_rng(), placement="near",
                               center=tuple(getattr(app, "cursor_world", app.mouse_world)), radius=15.0)
            if imgui.begin_child("archasm", V2(0, 0)):
                for ln in assembler.disassemble(w, d.K, d.KS, d.MB, exact=True, addresses=True).splitlines():
                    imgui.text(ln)
                imgui.end_child()


# ====================================================================== event log
def eventlog(app):
    r = app.runner
    f = app.log_filter
    f["type"] = imgui.input_text("type contains", f["type"])[1]
    f["min_sev"] = imgui.combo("min severity", f["min_sev"], list(SEVERITIES))[1]
    evs = list(r.elog.recent)
    evs = [e for e in evs if f["type"] in e.get("type", "") and SEVERITIES.get(e.get("severity", "info"), 1) >= f["min_sev"]]
    imgui.text(f"{len(evs)} events")
    if imgui.begin_table("ev", 4, imgui.TableFlags_.row_bg | imgui.TableFlags_.scroll_y | imgui.TableFlags_.resizable):
        for h in ("tick", "sev", "type", "detail"):
            imgui.table_setup_column(h)
        imgui.table_headers_row()
        for i, e in enumerate(reversed(evs[-300:])):
            imgui.table_next_row()
            imgui.table_next_column()
            clicked = imgui.selectable(f"{e.get('tick', '')}##{i}", False, imgui.SelectableFlags_.span_all_columns)[0]
            imgui.table_next_column()
            sev = e.get("severity", "info")
            imgui.text_colored(BAD if sev in ("error", "critical") else WARN if sev == "warning" else DIM, sev)
            imgui.table_next_column()
            imgui.text(e.get("type", ""))
            imgui.table_next_column()
            detail = {k: v for k, v in e.items() if k not in ("type", "severity", "wall", "tick")}
            imgui.text(json.dumps(detail, default=str)[:140])
            if clicked and r.ckpt:
                cands = [m for m in r.ckpt.good() if m.get("tick", 0) <= e.get("tick", 0)]
                if cands:
                    app.load_checkpoint_inplace(cands[-1]["path"])
                    app.notify(f"jumped to checkpoint at tick {cands[-1]['tick']} (before event)")
        imgui.end_table()


# ====================================================================== run panel
def run_panel(app):
    r = app.runner
    if app.watch_dir:
        imgui.text_colored(GOOD, app.watch_note)
        if imgui.button("Reload stats from run directory"):
            app.reload_watch_stats()
        imgui.separator()
    if r.rd:
        imgui.text_colored(GOOD, "recording: " + r.rd.path)
        if os.path.exists(r.rd.notes_path):
            app.run_notes = app.run_notes or open(r.rd.notes_path, encoding="utf-8").read()
        app.run_notes = imgui.input_text_multiline("notes.md", app.run_notes, V2(-1, 100))[1]
        if imgui.button("Save notes"):
            open(r.rd.notes_path, "w", encoding="utf-8").write(app.run_notes)
    else:
        imgui.text_colored(WARN, "not recording")
        if imgui.button("Start recording run directory"):
            app.start_recording()
    imgui.separator()
    imgui.text("Stop conditions")
    sc = r.stop_cfg
    sc["max_ticks"] = imgui.input_int("max ticks (0=off)", int(sc.get("max_ticks", 0)), 0, 0)[1]
    sc["max_wall"] = imgui.input_float("max wall seconds (0=off)", float(sc.get("max_wall", 0)), 0, 0, "%.0f")[1]
    sc["stop_on_extinction"] = imgui.checkbox("stop on extinction", bool(sc.get("stop_on_extinction", False)))[1]
    imgui.separator()
    imgui.text("Watchdog responses")
    w = r.watchdog.cfg
    acts = ["log", "pause", "rollback", "stop"]
    for name in ("nan", "ledger", "tps_drop", "mem_growth", "gpu_error"):
        setattr(w, name, acts[imgui.combo(name, acts.index(getattr(w, name)), acts)[1]])
    w.interval = imgui.input_int("check every N ticks", w.interval, 0, 0)[1]
    imgui.separator()
    if imgui.button("Inject a NaN (test the watchdog)"):
        app.poke("pos", int(np.nonzero(app.sim.state.cells.alive.numpy())[0][0]) if app.sim.population() else 0,
                 np.array([np.nan, np.nan], np.float32))
    if r.rd and os.path.exists(r.rd.status_path):
        imgui.separator()
        imgui.text_colored(DIM, "status.json")
        imgui.text_wrapped(json.dumps(r.status(), default=str)[:400])


# ====================================================================== smoke script
def smoke_actions(app, frame):
    """Exercises the UI under software GL in CI: opens every panel, selects cells, runs, uses tools."""
    if frame == 1:
        app.panels = {k: True for k in TITLES}
        app.runner.engine.add(builtin("keep_alive_basic")[0])
        app.runner.schedule.add_keyframe("sun_I", 50, 0.5, 20)
        app.runner.regions["r1"] = envmod.Region("r1", "rect", 10, 10, 50, 50)
    if frame == 4:
        app.find("oldest")
    if frame == 5:
        app.start_mode()
        app.running = True
    if frame == 8:
        app.overlay = 3
    if frame == 12:
        app.overlay = 1
    if frame == 16:
        app.overlay = 5
        app.color_mode = 3
    if frame == 20:
        app.overlay = 0
        app.running = False
        app.step(5)
        app.select_organism()
    if frame == 22:
        app.push_rewind()
        app.tiled = True
    if frame == 26:
        app.screenshot_next = True
    if frame == 30:                                     # every brush once
        c = app.frame["pos"].mean(0) if app.frame is not None and len(app.frame["al"]) else np.array([50.0, 50.0])
        app.mouse_world = np.array(c, np.float64)
        for b in range(1, len(BRUSHES)):
            app.brush = b
            app.apply_brush(True)
        app.brush = 0
        app.sim.step(2)
