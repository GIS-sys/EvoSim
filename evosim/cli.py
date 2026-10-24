"""Command line interface.  python -m evosim <command> ...   (or the `evosim` console script)

  run         headless run (new run directory, or --run-dir to continue one)
  resume      continue a run directory from its latest good checkpoint (or --checkpoint)
  supervise   run under a supervisor that restarts after crashes
  sweep       parameter grid x seeds from a YAML file, summary CSV
  inspect     text summary of a checkpoint
  export-genomes  dump the top genomes of a checkpoint / archive as .asm
  render-video    frames -> video with ffmpeg
  bench       performance measurement of the target scenarios
  asm / disasm    assembler tools
  ui          interactive viewer
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_duration(s):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smhd]?)", str(s).strip())
    if not m:
        raise argparse.ArgumentTypeError(f"bad duration {s!r} (examples: 90s, 30m, 24h, 2d)")
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def _kv(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if not _:
            raise SystemExit(f"--set expects param=value, got {it!r}")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _load_config(args):
    from .config import Config
    cfg = Config()
    src = args.config or (os.path.join(REPO, "presets", f"{args.preset}.json") if args.preset else None)
    if src:
        if src.endswith((".yaml", ".yml")):
            import yaml
            with open(src, encoding="utf-8") as f:
                cfg = Config.from_dict(yaml.safe_load(f))
        else:
            cfg = Config.load(src)
    for k, v in _kv(args.set).items():
        cfg.set(k, v)
    return cfg


def _load_policies(spec, target):
    from .run.policies import PolicyEngine, combine
    if not spec:
        return PolicyEngine(eval_interval=10)
    if os.path.exists(spec):
        return PolicyEngine.load(spec)
    return PolicyEngine(combine(*[s.strip() for s in spec.split(",")], target=target), eval_interval=10)


def _stop_dict(args):
    stop = {}
    if args.ticks:
        stop["max_ticks"] = args.ticks
    if args.wall_time:
        stop["max_wall"] = args.wall_time
    if args.stop_on_extinction:
        stop["stop_on_extinction"] = True
    if args.stop_metric:
        m = re.fullmatch(r"(\w+)\s*(>=|<=|>|<)\s*([\d.eE+-]+)", args.stop_metric)
        if not m:
            raise SystemExit("--stop-metric expects e.g. org_size_max>=50")
        stop["metric"] = {"metric": m.group(1), "op": m.group(2), "value": float(m.group(3))}
    return stop


def _runner_kwargs(args):
    from .run.rundir import Retention
    from .run.runner import Watchdog, WatchdogConfig
    return dict(ckpt_every_ticks=args.checkpoint_ticks, ckpt_every_seconds=args.checkpoint_seconds,
                max_dir_bytes=int(args.max_dir_gb * 2**30) if args.max_dir_gb else None,
                retention=Retention(args.keep_last), stats_format=args.stats_format, webhook=args.webhook,
                stop=_stop_dict(args), frame_every=args.frame_every,
                watchdog=Watchdog(WatchdogConfig(interval=args.watchdog_interval)))


def _add_run_opts(p):
    p.add_argument("--ticks", type=int)
    p.add_argument("--wall-time", type=parse_duration)
    p.add_argument("--stop-on-extinction", action="store_true")
    p.add_argument("--stop-metric", help="e.g. org_size_max>=50")
    p.add_argument("--policies", help="policies.json, or comma-separated built-ins "
                                      "(keep_alive_basic,fill_to_target,diversity_guard,extinction_rollback,overpop_cap)")
    p.add_argument("--policy-target", type=int, default=20000, help="N for fill_to_target")
    p.add_argument("--schedule", help="schedule.json")
    p.add_argument("--device", help="cuda:0 or cpu")
    p.add_argument("--checkpoint-ticks", type=int, default=10000)
    p.add_argument("--checkpoint-seconds", type=float, default=0)
    p.add_argument("--keep-last", type=int, default=10)
    p.add_argument("--max-dir-gb", type=float, default=0)
    p.add_argument("--stats-format", choices=("parquet", "csv"), default="parquet")
    p.add_argument("--frame-every", type=int, default=0, help="save a time-lapse frame every N ticks")
    p.add_argument("--webhook", help="URL receiving JSON POSTs for events >= error")
    p.add_argument("--watchdog-interval", type=int, default=200)
    p.add_argument("--no-sleep-prevention", action="store_true")


def cmd_run(a):
    from .config import Config
    from .run.env import Schedule
    from .run.rundir import RunDir
    from .run.runner import Runner
    import warp as wp
    if a.run_dir and os.path.isdir(os.path.join(a.run_dir, "checkpoints")) and any(
            f.endswith(".evck") for f in os.listdir(os.path.join(a.run_dir, "checkpoints"))):
        r = Runner.resume(a.run_dir, device=a.device, overrides=_kv(a.set), **_runner_kwargs(a))
    else:
        cfg = _load_config(a)
        rd = RunDir(a.run_dir) if a.run_dir else None
        sched = Schedule.load(a.schedule) if a.schedule else None
        r = Runner.create(cfg, root=a.root, name=a.name, device=a.device, rundir=rd, schedule=sched,
                          policies=_load_policies(a.policies, a.policy_target), **_runner_kwargs(a))
    if a.policies and r.engine.policies == [] or (a.policies and not r.engine.policies):
        r.engine = _load_policies(a.policies, a.policy_target)
    print(f"run directory: {r.rd.path}\ndevice: {r.sim.device}  population: {r.sim.population()}", flush=True)
    t0 = time.time()
    r.run(sleep_prevention=not a.no_sleep_prevention)
    dt = time.time() - t0
    print(f"stopped at tick {r.sim.tick} ({r.stop_requested}); {r.sim.tick / max(dt, 1e-9):.1f} ticks/s", flush=True)
    return 0


def cmd_resume(a):
    a.run_dir = a.run_dir
    from .run.runner import Runner
    r = Runner.resume(a.run_dir, checkpoint=a.checkpoint, device=a.device, overrides=_kv(a.set), **_runner_kwargs(a))
    print(f"resumed {a.run_dir} at tick {r.sim.tick}", flush=True)
    r.run(sleep_prevention=not a.no_sleep_prevention)
    return 0


def cmd_supervise(a):
    from .run.rundir import RunDir
    from .run.supervisor import supervise
    if a.target and os.path.isdir(a.target):
        run_dir = a.target
    else:
        rd = RunDir.create(a.root, a.name)
        run_dir = rd.path
    rest = a.child_args[1:] if a.child_args and a.child_args[0] == "--" else a.child_args
    if not (a.target and os.path.isdir(a.target)) and a.target:
        rest = [a.target] + rest if a.target.startswith("-") else rest
    return supervise(run_dir, rest, delay=a.delay, max_restarts_per_hour=a.max_restarts_per_hour)


def cmd_sweep(a):
    import yaml
    from .config import Config
    from .run.policies import PolicyEngine, StatsView
    from .sim import Sim
    with open(a.file, encoding="utf-8") as f:
        sw = yaml.safe_load(f)
    base = Config()
    if sw.get("preset"):
        base = Config.load(os.path.join(REPO, "presets", f"{sw['preset']}.json"))
    elif sw.get("config"):
        base = Config.load(sw["config"])
    for k, v in (sw.get("base") or {}).items():
        base.set(k, v)
    grid = sw.get("grid", {})
    seeds = sw.get("seeds", [1])
    ticks = int(sw.get("ticks", 10000))
    names = list(grid)
    out = sw.get("output", "sweep_summary.csv")
    rows = []
    for combo in itertools.product(*[grid[n] for n in names]):
        for seed in seeds:
            cfg = base.copy()
            for n, v in zip(names, combo):
                cfg.set(n, v)
            cfg.set("seed", seed)
            sim = Sim(cfg, device=a.device)
            t0 = time.time()
            try:
                sim.step(ticks)
                status = "ok"
            except Exception as e:                         # noqa: BLE001
                status = f"error: {e}"
            s = sim.sample_stats() if status == "ok" else {}
            row = dict(zip(names, combo), seed=seed, status=status, ticks=sim.tick, wall_s=round(time.time() - t0, 2))
            row.update({k: v for k, v in s.items() if not k.startswith("ms_") and k not in ("tick", "wall")})
            rows.append(row)
            print(row, flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)
    return 0


def cmd_inspect(a):
    from .io import snapshot
    import numpy as np
    from .genome import isa
    meta, payload = snapshot.read_container(a.checkpoint)
    cap = snapshot.cap_from_bytes(payload)
    m = snapshot._unj(cap["meta_json"])
    print(f"checkpoint: {a.checkpoint}\n  tick {m['tick']}  good={meta.get('good')}  reason={meta.get('reason')}")
    n = len(cap["cell_idx"])
    print(f"  cells {n}  bonds {len(cap['bond_idx'])}  genomes in use {len(cap['gen_idx'])}")
    if n:
        mass, en = cap["c_mass"], cap["c_energy"]
        print(f"  mass: total {mass.sum():.1f} mean {mass.mean():.2f} max {mass.max():.2f};  energy mean {en.mean():.2f}")
        print(f"  distinct lineages {len(np.unique(cap['c_lineage_id']))}  max depth {cap['c_division_depth'].max()}")
        org = cap["c_organism_id"]
        _, c = np.unique(org, return_counts=True)
        print(f"  organisms (>=2 cells): {(c >= 2).sum()}  largest: {c.max()} cells")
        tr = cap["c_traits"].mean(0)
        print("  trait means: " + "  ".join(f"{k}={v:.2f}" for k, v in zip(isa.TRAIT_NAMES, tr)))
        gl = cap["g_length"]
        print(f"  genome length mean {gl.mean():.1f} (min {gl.min()}, max {gl.max()})")
    print(f"  nutrient total {cap['f_nutrient'].sum():.2f}")
    mod = {k: v for k, v in m["config"]["params"].items() if k in __import__("evosim.config", fromlist=["x"]).PARAM_SPECS
           and v != __import__("evosim.config", fromlist=["x"]).PARAM_SPECS[k].default}
    print(f"  parameters changed from default: {len(mod)}")
    for k, v in sorted(mod.items()):
        print(f"    {k} = {v}")
    return 0


def cmd_export(a):
    import numpy as np
    from .genome import assembler, isa
    os.makedirs(a.out, exist_ok=True)
    k = 0
    if a.source.endswith(".sqlite"):
        from .run.archive import GenomeArchive
        ar = GenomeArchive(a.source)
        for h, length, ft, peak, lin, ph, tags, tm in ar.list(a.top):
            w = ar.get(h)
            assembler.save_asm(os.path.join(a.out, f"{k:03d}_{h}.asm"), w,
                               header=f"hash {h}  peak count {peak}  lineage {lin}  tags {tags}")
            k += 1
    else:
        from .io import snapshot
        _, payload = snapshot.read_container(a.source)
        cap = snapshot.cap_from_bytes(payload)
        gid = cap["c_genome_id"]
        u, c = np.unique(gid, return_counts=True)
        pos = {int(g): i for i, g in enumerate(cap["gen_idx"])}
        for g, cnt in sorted(zip(u, c), key=lambda t: -t[1])[:a.top]:
            i = pos[int(g)]
            w = [int(x) & 0xFFFFFFFF for x in cap["g_words"][i, :cap["g_length"][i]]]
            assembler.save_asm(os.path.join(a.out, f"{k:03d}_{assembler.genome_hash(w):016x}.asm"), w,
                               header=f"{cnt} cells carry this genome")
            k += 1
    print(f"wrote {k} genomes to {a.out}")
    return 0


def cmd_video(a):
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found on PATH: install it, then re-run. Frames are in", os.path.join(a.run_dir, "frames"))
        return 1
    frames = os.path.join(a.run_dir, "frames")
    files = sorted(f for f in os.listdir(frames) if f.startswith("frame_") and f.endswith(".png"))
    if not files:
        print("no frames found (run with --frame-every N)")
        return 1
    lst = os.path.join(frames, "list.txt")
    with open(lst, "w") as f:
        for n in files:
            f.write(f"file '{n}'\n")
    out = a.out or os.path.join(a.run_dir, "timelapse.mp4")
    subprocess.run(["ffmpeg", "-y", "-r", str(a.fps), "-f", "concat", "-safe", "0", "-i", lst, "-pix_fmt", "yuv420p", out],
                   check=True)
    print("wrote", out)
    return 0


def cmd_bench(a):
    from .bench import main as bench_main
    return bench_main(a)


def cmd_asm(a):
    from .genome import assembler
    w = assembler.load_asm(a.file)
    out = a.out or os.path.splitext(a.file)[0] + ".gnm"
    assembler.save_binary(out, w)
    print(f"{len(w)} words -> {out}")
    return 0


def cmd_disasm(a):
    from .genome import assembler
    w = assembler.load_genome(a.file)
    sys.stdout.write(assembler.disassemble(w, addresses=a.addresses))
    return 0


def cmd_ui(a):
    try:
        from .ui.app import run_app
    except ImportError as e:
        print("UI dependencies missing:", e, "\nInstall with: pip install glfw moderngl imgui-bundle")
        return 1
    return run_app(a)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="evosim", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cfg_opts(p):
        p.add_argument("--config", help="config JSON/YAML")
        p.add_argument("--preset", help="sunny_pond | dark_vents | predator_world | fast_debug")
        p.add_argument("--set", action="append", help="param=value (repeatable)")

    p = sub.add_parser("run", help="headless run")
    cfg_opts(p)
    p.add_argument("--name")
    p.add_argument("--root", default="runs")
    p.add_argument("--run-dir")
    _add_run_opts(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("resume", help="continue a run directory")
    p.add_argument("run_dir")
    p.add_argument("--checkpoint")
    p.add_argument("--set", action="append", help="param=value override (logged)")
    _add_run_opts(p)
    p.set_defaults(fn=cmd_resume)

    p = sub.add_parser("supervise", help="restart the run after crashes")
    p.add_argument("target", nargs="?", help="existing run directory (resume) or omitted for a new run")
    p.add_argument("--root", default="runs")
    p.add_argument("--name")
    p.add_argument("--delay", type=float, default=5.0)
    p.add_argument("--max-restarts-per-hour", type=int, default=6)
    p.add_argument("child_args", nargs=argparse.REMAINDER, help="-- then the arguments of `evosim run`")
    p.set_defaults(fn=cmd_supervise)

    p = sub.add_parser("sweep")
    p.add_argument("file")
    p.add_argument("--device")
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("inspect")
    p.add_argument("checkpoint")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("export-genomes")
    p.add_argument("source", help="checkpoint .evck or archive .sqlite")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--out", default="exported_genomes")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("render-video")
    p.add_argument("run_dir")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_video)

    p = sub.add_parser("bench", help="performance of the target scenarios")
    p.add_argument("--cells", type=int, default=100000)
    p.add_argument("--ticks", type=int, default=200)
    p.add_argument("--device")
    p.add_argument("--vm-only", action="store_true")
    p.add_argument("--light-only", action="store_true")
    p.add_argument("--preset", default="sunny_pond")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("asm")
    p.add_argument("file")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_asm)
    p = sub.add_parser("disasm")
    p.add_argument("file")
    p.add_argument("--addresses", action="store_true")
    p.set_defaults(fn=cmd_disasm)

    p = sub.add_parser("ui", help="interactive viewer")
    cfg_opts(p)
    p.add_argument("--device")
    p.add_argument("--run-dir", help="open a run directory (tails its stats, loads checkpoints as forks)")
    p.add_argument("--smoke", type=int, default=0, help=argparse.SUPPRESS)
    p.set_defaults(fn=cmd_ui)

    a = ap.parse_args(argv)
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
