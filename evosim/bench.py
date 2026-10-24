"""Performance measurement for the spec's targets. Reports ticks/s, per-phase timings, VRAM."""
from __future__ import annotations

import os
import time

import numpy as np
import warp as wp

from .config import Config, PARAM_SPECS
from .sim import PHASES, Sim
from .state import estimate_vram


def _pop_to(sim, n, rs):
    """Fill the world with n cells of mixed random + seed genomes without waiting for growth."""
    seed = sim.load_genome_file("seeds/seed_replicator.asm")
    sim.inject(n // 2, [seed], rs)
    gens = [sim.random_genome(rs, 32) for _ in range(min(n - n // 2, 4000))]
    sim.inject(len(gens), gens, rs, one_genome_per_cell=True)


def main(a):
    dev = a.device or str(wp.get_device())
    base = Config.load(os.path.join(os.path.dirname(os.path.dirname(__file__)), "presets", f"{a.preset}.json"))
    n = a.cells
    cap = 262144 if n <= 262144 else int(n * 1.3)
    side = float(np.sqrt(n / 0.025 / 4) * 4)          # ~0.025 cells per unit^2 (default sunny_pond density)
    base.set("max_cells", cap); base.set("initial_count", 0)
    base.set("world_w", max(512.0, side)); base.set("world_h", max(512.0, side))
    base.set("ledger_interval", 0); base.set("stats_interval", 0); base.set("profile_interval", 10)
    print(f"device: {dev}")
    sim = Sim(base, device=dev, seed_population=False)
    rs = np.random.default_rng(0)
    vr = estimate_vram(base)
    print(f"capacity {cap}: estimated state memory {vr / 2**30:.2f} GiB, allocated {sim.state.vram_bytes() / 2**30:.2f} GiB"
          f"  (target < 4 GiB)")
    t0 = time.time()
    _pop_to(sim, n, rs)
    sim.step(3)                                       # compile kernels / warm up
    print(f"population {sim.population()} (setup {time.time() - t0:.1f}s)")
    sim.cfg.set("profile_interval", 0)
    wp.synchronize_device(sim.device)
    t0 = time.perf_counter()
    sim.step(a.ticks)
    wp.synchronize_device(sim.device)
    dt = time.perf_counter() - t0
    tps = a.ticks / dt
    print(f"headless: {tps:.1f} ticks/s  ({dt / a.ticks * 1000:.2f} ms/tick)   target >= 100")
    sim.cfg.set("profile_interval", 10)
    sim.timing_samples = 0
    sim.step(30)
    tot = sum(sim.timings.values())
    print("per-phase (ms, moving average):")
    for k in PHASES:
        print(f"  {k:14s} {sim.timings[k]:8.3f}  {100 * sim.timings[k] / max(tot, 1e-9):5.1f}%")
    worst = max(sim.timings, key=sim.timings.get)
    print(f"bottleneck kernel group: {worst}")
    if tps < 100:
        print("below target: see DESIGN_NOTES.md section 'Performance' for the proposed fixes for the bottleneck above")
    return 0
