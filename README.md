# EvoSim

GPU-accelerated artificial-life simulator in Python. A large population of cells lives in a 2D toroidal world.
Every cell runs a genome (a program for a small register/state machine) and senses **only local physical
quantities**: what touches it, what dissolves in the water at its position, the light reaching its membrane, and
signals passed through its own bonds. Directional vision, smell tracking, multicellular bodies and predation
strategies are meant to *emerge* from morphology and physics, never from "sense other cells" shortcuts.

Simulation kernels: [NVIDIA Warp](https://github.com/NVIDIA/warp). Viewer: glfw + moderngl + Dear ImGui (imgui-bundle).
See `DESIGN_NOTES.md` for decisions, deviations, and what has and has not been verified.

## Install (Windows 10/11, NVIDIA GPU)

1. Install a current NVIDIA driver (**530 or newer** is required: the kernels pass large structs by value and need
   CUDA 12.1+ large kernel-parameter support). You do **not** need a separate CUDA toolkit; Warp bundles what it needs.
2. Install Python 3.11 or newer from python.org.
3. In a terminal, from this folder:

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .          # optional: gives you the `evosim` command; otherwise use `python -m evosim`
```

4. Check that the GPU is seen:

```
python -c "import warp as wp; wp.init(); print(wp.get_devices())"
```

You should see `cuda:0` listed. If only `cpu` appears, update the driver. Everything also runs on the CPU backend
(`--device cpu`), just much slower. Linux works the same way.

The first launch compiles the kernels (about a minute); later launches use Warp's cache.

## Run

```
python -m evosim ui --preset fast_debug            # interactive viewer, small world, 5k cells
python -m evosim ui --preset sunny_pond            # the default pond (2048x2048, 262k capacity)
python -m evosim bench --cells 100000              # measure ticks/s, per-phase timings, memory
python -m evosim run --preset sunny_pond --name pond1 --wall-time 24h --policies keep_alive_basic
python -m evosim supervise --name pond1 -- --preset sunny_pond --wall-time 24h --policies keep_alive_basic
python -m evosim resume runs/<dir>                 # continue from the latest good checkpoint
python -m pytest                                   # test suite (CPU backend unless a GPU is found)
```

Presets: `sunny_pond`, `dark_vents`, `predator_world`, `fast_debug` (in `presets/`, regenerate with
`python scripts/make_presets.py`). Seeds: `seed_replicator`, `seed_colony`, `seed_grazer`, `seed_predator`
(in `seeds/`).

### All commands

| command | purpose |
|---|---|
| `run` | headless run; `--ticks N` / `--wall-time 24h` / `--stop-metric org_size_max>=50` / `--stop-on-extinction`; `--policies`, `--schedule`, `--set param=value`, `--checkpoint-ticks`, `--max-dir-gb`, `--frame-every`, `--webhook URL` |
| `resume <run_dir>` | continue a run (overrides via `--set`, logged to `config_history.jsonl`) |
| `supervise` | run under a supervisor that restarts after crashes (driver reset, native crash, OOM, kill) from the latest good checkpoint |
| `sweep sweep.yaml` | parameter grid x seeds, summary CSV |
| `inspect <ckpt>` | text summary of a checkpoint |
| `export-genomes <ckpt\|archive.sqlite> --top K` | write genomes as `.asm` |
| `render-video <run_dir>` | frames to video (needs ffmpeg) |
| `bench` | performance of the target scenarios |
| `asm` / `disasm` | assembler tools |
| `ui` | interactive viewer (`--run-dir D` opens a *fork* of a run; the original is never written) |

Built-in policy sets for `--policies` (comma-separated): `keep_alive_basic`, `fill_to_target`, `diversity_guard`,
`extinction_rollback`, `overpop_cap`, or a `policies.json` file. Example sweep file:

```yaml
preset: fast_debug
base: {max_cells: 8192}
grid: {nutrient_cap: [0.02, 0.05], mu_point: [0.001, 0.005]}
seeds: [1, 2, 3]
ticks: 5000
output: sweep_summary.csv
```

## Viewer

Left-drag pans, wheel zooms, click selects, shift-click adds to the selection. Keys: `Space` play/pause, `.` step,
`n` step N, `f` follow, `s` screenshot, `[` / `]` parent / daughter in lineage, `r` rewind, `k` kill, `Esc` deselect.

Panels (View menu; dockable; two built-in layouts, *debug* and *monitor*): run controls (event-driven run modes,
epoch mode), parameters (generated from the config metadata: search, modified highlight, reset, pins, preset diff,
restart-time parameters), cell inspector (state, sensors, energy breakdown, disassembly with `ip`, execution trace,
actions and failure reasons, breakpoints, edit actions, CPU sandbox, parent/daughter genome diff), organism view,
stats and plots, profiler, policies, schedule editor, environment events and named regions, checkpoint browser
(load / fork / keep-forever), genome archive, event log, tools (find, brushes, overlays, rewind buffer, layouts).

## Writing genomes

Text `.asm`, one instruction per line, `;` comments. Every 32-bit word is a valid instruction, so mutations never
produce invalid code. Jumps target labels (`JMP k` goes to the next `LABEL k`, wrapping).

```
LABEL 0
  SET r0 1.0
  TRAIT PHOTO r0
LABEL 1
  SENSE r1 ENERGY_FRAC
  SET r2 0.8
  JLT r1 r2 2
  RAND r3
  SET r4 0.5
  SPLIT r3 r4 0
LABEL 2
  YIELD
  JMP 1
```

Operand names: sensors (`ENERGY`, `LIGHT`, `LIGHT_DELTA`, `NUTRIENT`, `TOUCH_F/R/B/L`, `CHEM_0..`, `TRAIT_*`, ...),
traits (`PHOTO`, `ABSORB`, `MOTOR`, `PREDATION`, `ARMOR`, `PIGMENT`, `PHOTORECEPTOR`, `ADHESION`, `RECEPTOR_WIDTH`),
sectors `F R B L`, chemical channels `C0..`, signal channels `S0..`, bond slots `B0..`.
`python -m evosim disasm file.asm --addresses` lists a genome.

## Run directory

`runs/<time>_<n>/`: `config.json`, `config_history.jsonl` (every live parameter change with tick, time and source),
`policies.json`, `schedule.json`, `env.json` (git commit, package versions, GPU, driver, OS), `notes.md`,
`checkpoints/` (`.evck`: atomic writes, SHA-256 checked, tiered retention), `stats/` (Parquet chunks, CSV fallback),
`events.jsonl`, `log.txt`, `archive.sqlite`, `frames/`, `status.json`.

## Windows notes

* **TDR**: Windows resets the GPU if one kernel runs longer than about 2 s, and a laptop GPU also drives the display.
  A tick is about 40-50 short launches (never one giant kernel). If you raise `ray_steps`, `max_cells` or the light
  grid resolution a lot, keep an eye on the per-phase timings in the profiler; if any single phase approaches
  hundreds of ms, lower the cost or the population.
* **Sleep**: headless runs call `SetThreadExecutionState` so the machine does not sleep (disable with
  `--no-sleep-prevention`).
* **Shutdown**: Ctrl+C, Ctrl+Break, closing the console, and SIGTERM write a final checkpoint.
* The viewer needs OpenGL 3.3. If the window cannot be created, update the graphics driver; on hybrid-graphics
  laptops set the Python executable to use the NVIDIA GPU in Windows graphics settings.

## Project layout

```
evosim/
  config.py          typed parameters + metadata (drives the UI, presets, GPU parameter struct)
  dims.py            the only module with 2D-specific simulation math (3D-ready seam)
  state.py           structure-of-arrays GPU state, free lists, ledger term indices
  rng.py             counter-based RNG shared by CPU and GPU
  genome/            isa, assembler, reference_vm (CPU), table (copy-on-write), mutate
  kernels/           spatial_hash, light, sense, vm, metabolism, interactions, lifecycle, physics, fields, organisms
  sim.py spawn.py    tick pipeline, ledger, stats; host-driven inject / clone / cull
  io/                snapshots + checkpoint container, frame rendering
  run/               runner, watchdog, policies, schedule & events, archive, run directory, supervisor
  ui/                app (loop, viewport, run modes), render (moderngl), panels (ImGui)
  cli.py bench.py
seeds/ presets/ scripts/ tests/ DESIGN_NOTES.md
```
