# DESIGN NOTES

Status key: **[tested]** has an automated test that passed on the CPU backend (Linux, Warp 1.18.0);
**[untested]** written but not exercised by a test; **[unmeasured]** needs the target hardware.
No GPU was available during development: nothing here has run on CUDA or on Windows.

## 1. Decisions that follow the spec

* Warp kernels, one per-cell interpreter kernel for the VM (no vectorised VM). The tick is an ordered list of ~45 short
  launches, each phase individually timed (`profile_interval`).
* Structure-of-arrays state with fixed capacity and free lists; bonds in their own table with a free list; genome
  table with copy-on-write (identical copies are released and the daughter shares the parent entry).
* Section 5 ISA, label jumps with a per-genome label table (O(occurrences of that label)), 32-bit words with every
  word valid. Opcode / operand decoding is modulo, as specified. [tested: ISA round trip, every word decodes]
* No-cheat sensing: sensors read own state, fields at the cell's own position, contacts by body-frame sector (force
  magnitude only), occlusion-shaped light, and previous-tick bond signals. The per-cell contact list and the
  best-neighbour-per-sector table are used **only** to resolve `BOND` / `EAT` targets and are never readable by the
  VM. No sensor returns another cell's distance, direction, identity or genome.
* Physics: gather formulation. Every pair force comes from an exactly antisymmetric expression, so Newton's third law
  holds without atomics; the bond angular springs apply an equal and opposite force couple, so angular momentum is
  conserved as well. [tested: momentum and COM stay fixed for an internal-forces-only organism, including one that
  spins its cells with TURN; THRUST does translate.]
* Light: opacity density grid built by gather (cloud-in-cell), ray-marched transmittance per source with the cell's own
  footprint subtracted analytically, directional reception `w = RW + (1-RW) max(0,cos)^p`. [tested: numeric thresholds
  for the cup-of-pigment scene, own-footprint exclusion, point-light falloff + absorption, day/night, photosynthesis.]
* Conservation: per-tick energy, mass and nutrient ledgers measured on a configurable interval, with an
  `intervention` term so injections / clones / culls / bloom events do not break balance. A failing ledger raises
  `LedgerError` and the watchdog responds per configuration. [tested: a 250-tick run with all seeds + heavy mutation
  balances every tick; a planted leak is detected.]
* Checkpoints: atomic (temp + fsync + rename), SHA-256 verified, written on a background thread (at most one pending),
  alive cells / in-use genomes only. [tested: kill during replace leaves the previous file loadable; corrupt file is
  skipped with fallback.]

## 2. Deviations and implementer choices (please review)

| # | What | Why |
|---|---|---|
| 1 | **`TANH` is softsign `x/(1+abs(x))`**, not true tanh. | Real tanh is not guaranteed bit-identical between CPU and CUDA libm, and the spec requires bit-for-bit agreement of the GPU and CPU interpreters. Softsign is bounded, smooth, odd and saturating, but its shape differs. If you want true tanh, relax the equivalence test to a tolerance for that opcode. |
| 2 | `NEG` is `0 - x`. | Warp's negation maps -0.0 to +0.0 on both backends; found by the equivalence test. |
| 3 | Default `substeps` = 4 (spec: 2). | With 2, four of five bonds of the colony seed broke during growth; with 4 every link holds at exactly 0.500 rad with zero strain. |
| 4 | Angular bond stiffness is `k_bond_angle * rest_length^2`. | Makes the parameter scale-invariant; a fixed torque constant is unstable for small cells. |
| 5 | RNG is a stateless hash chain (lowbias32 mixer keyed by seed/salt/tick/uid/purpose/counter), not Philox. | Bit-identical on CPU and GPU, no per-cell RNG state to store or snapshot. Adequate for mutation and noise; not a cryptographic or high-dimensional-quality generator. |
| 6 | Injected founders get **one new lineage per distinct genome per injection**. | 2000 copies of one seed are one lineage (so dominance / diversity policies and lineage plots mean something); random genomes are one lineage each. Clones keep their source lineage. |
| 7 | Policy actions run on the host between ticks. Culls are GPU kernels; inject / clone use host-side placement search. | The overlap-avoiding placement search is simplest on the host and these actions are rare. `fill_to_target` (1 clone per tick) is the heaviest user. |
| 8 | Breakpoints are tick-granular. "About to divide" = the cell issued a split this tick. | A tick is atomic; the VM kernel records opcode / address breakpoints for traced cells and the host pauses after that tick. |
| 9 | Genome table capacity equals cell capacity. | Entries can never run out before cells do. Costs 1.3 KB per slot. |
| 10 | UI shows one field overlay at a time with a single opacity slider. | Spec asked for a toggle + slider per overlay. |
| 11 | `ui --run-dir` loads a **fork** of the latest good checkpoint and shows the stats files; it never writes to the run directory. | Avoids colliding with a headless run still using it. Live attach to a running process remains phase 2 as in the spec. |
| 12 | `mutate` of word 0 as evolvable rate: if word 0 decodes to `NOP`, rates scale by 2^((a-128)/32). | The spec left the header-word encoding open. |
| 13 | Event-driven run modes "until depth +1" and "follow lineage" read a few arrays every tick. | Fine for debugging runs; not meant for uncapped speed. |

## 3. Determinism

* Slot allocation always goes through exclusive prefix scans over per-element flags, never atomic counters, so
  request -> slot depends only on element order. Scattered sums (nutrient uptake, bites, deposits) accumulate in
  **int64 fixed point** (order independent). Bond conflicts use `atomic_min` locks on integers. Connected components
  converge to a unique fixed point. Neighbour loops run over a stably sorted spatial hash.
* The only float atomics are the **diagnostic ledger accumulators** (float64); they never feed back into the dynamics.
  Hence no separate "deterministic mode" was needed on this design; it is untested on a real GPU where warp scheduling
  differs. [tested on CPU: same seed + params -> identical state hash; different seed or salt -> different; snapshot
  continuation is bit-identical.]
* Not bit-identical across **different hardware or backends** (CPU vs GPU transcendental functions, FMA contraction).
  Only the VM is guaranteed bit-identical between CPU and GPU.

## 4. Energy / mass accounting and the exploit surface

Mass has sources and sinks that are ledgered: growth converts energy to mass (`growth_eff`), death returns
`detritus_fraction` of the mass to the nutrient field, and nutrient is converted to energy by uptake (`absorb_eff`).
The loop death -> detritus -> uptake -> growth is therefore lossy only while

    detritus_fraction * absorb_eff * growth_eff < 1      (defaults: 0.5 * 1.0 * 0.5 = 0.25)

`Config.exploit_warnings()` checks this (and `bite_eff * bite_mass_gain * growth_eff`, `share_eff <= 1`) and the
runner emits a warning event at start-up. Raise these parameters knowingly: evolution finds free-energy loops quickly.
Other required behaviour is covered by tests: no action pushes energy below zero (thrust, turn, emit, bond, split,
trait change are scaled down or failed with a `fail_flags` bit), a split never increases total mass or energy, thrust
cost is paid before the force is applied.

## 5. Memory

State at the default capacity of 262,144 cells: **0.706 GiB allocated (capacity 262144); estimate 0.708 GiB** (spec target
< 4 GiB). This is the sum of the Warp arrays (including a genome table with one 256-word slot per cell); it does not
include the CUDA context, the radix-sort temporaries or the viewer's buffers. [measured on the CPU backend; the
buffer sizes are the same on the GPU, the context overhead is [unmeasured]]. 2^21 cells needs about 5.4 GiB (estimate).

## 6. Performance  [unmeasured]

The three performance targets (>= 100 ticks/s headless and >= 30 ticks/s rendering at 100k cells, memory above) have
**not** been measured: there was no GPU. Run `python -m evosim bench --cells 100000` on the RTX 4090; it prints
ticks/s, a per-phase breakdown (hash, opacity, light, sense, vm, metabolism, interactions, births, deaths, physics,
fields, organisms) and names the bottleneck phase. Please send the output back; that is the data needed to tune.

The three prototypes the spec asks for in section 16 (VM throughput at 100k x 16 instructions, ray-march cost, the
moderngl + imgui loop on Windows) were **not run**. The moderngl + imgui loop was verified on Linux with software GL
only (`tests/test_ui_smoke.py`). `bench` covers the first two through the per-phase timings (`vm`, `light`).

Likely bottlenecks and the fixes I would try, in order:

1. **`light`**: per cell, per source, `ray_steps` x 4 bilinear taps, each tap evaluating a cloud-in-cell weight with a
   minimum-image. Fixes: precompute a per-node "self" correction only near the cell, lower `ray_steps` (default 16),
   coarser `light_cell`, sample the opacity grid as a texture, or ray-march from the light-grid node instead of per cell.
2. **`vm`**: one thread per cell with register-array access through global memory. Fixes: keep the 8 registers in
   local variables, cache `words[gid]` rows, sort cells by genome id for coherence.
3. **Launch overhead**: ~45 launches + 4 physics substeps per tick. Fix: capture the tick into a CUDA graph
   (`wp.ScopedCapture`) for the host-sync-free portion.
4. **Radix sort of 2N keys every tick** (spatial hash). Fix: sort every k-th tick with a larger bucket margin, or
   incremental re-bucketing.
5. Host copies: `sample_stats` copies ~20 arrays every `stats_interval` ticks (default 100); the viewer copies pos /
   mass / angle / colour inputs every rendered frame (the spec's accepted initial approach).

## 7. CUDA / Windows risks  [unverified]

* **Kernel parameter size.** Several kernels take `Cells` + `Scratch` + `Fields` + `SimParams` by value (~6-7 KB). CUDA
  12.1+ with driver >= 530 allows 32 KB on Volta and later (the 4090 qualifies); older drivers cap at 4 KB and kernel
  launch would fail. If you must support older drivers, split `Scratch` per phase.
* `wp.utils.array_scan` and `radix_sort_pairs` are used on the device arrays; they worked on CPU. Their CUDA behaviour
  with these dtypes is expected to be identical but is [untested].
* Windows: `SetThreadExecutionState`, the console-close handler, and the viewer on a hybrid laptop GPU were written
  from the API documentation and never executed. TDR: every launch is short by construction.
* Warp's CPU backend executes sequentially, so **race conditions that depend on parallel execution have not been
  exercised**. The design avoids races by construction (gathers, scans, fixed point, min-locks), but the first GPU run
  with `ledger_interval=1` (the default preset uses 10 for `fast_debug`) and the determinism test is the real check.

## 8. Verification status against the milestones

| Milestone | Status |
|---|---|
| M1 skeleton & physics | torus math, collisions across the seam and momentum conservation **[tested]**. The "100k passive cells for 100k ticks without NaN" soak was **not run**. |
| M2 VM & lifecycle | GPU == CPU VM bit-for-bit (300 genomes x 40 ticks x 2 seeds by default; `EVOSIM_FULL=1` runs the spec's 10k x 100) **[tested at reduced size on CPU]**. Replicator grows to carrying capacity and the ledger balances **[tested]**. |
| M3 light | numeric cup-eye test **[tested]**. |
| M4 bonds | `seed_colony` forms a 6-cell curved body, only the tip pigmented, holds 0.5 rad per link; a shaken chain keeps curvature with angular springs and loses it without; internal-only organism cannot translate **[tested]**. |
| M5 predation & chemistry | bite / engulf / armor / simultaneous-bite scaling / damage sensor / emit / diffusion / decay **[tested]**. **Predator lineages emerging in `predator_world` within a tick budget was not run**; it needs a long run. `--stop-metric` and the `trait_PREDATION` stat exist to measure it. |
| M6 debugger & run modes | all panels, inspector tabs, breakpoints, run modes, brushes and rewind exercised by a scripted smoke run under software GL **[tested as "does not crash and renders"]**; interactive feel on a real display is **[untested]**. |
| M7 stats, persistence, perf | exact snapshots, exports, headless CLI, profiler **[tested]**; targets **[unmeasured]**. |
| M8 unattended operation | policies (all triggers, hysteresis, cooldowns, priority/exclusive, built-in sets), schedule + events determinism, retention tiers, disk budget, NaN -> rollback with salt change and tweak, rollback rate limit, SIGKILL -> supervisor -> resume **[tested]**. **The 24-hour `sunny_pond` run was not performed.** |

## 9. Known limitations

* Stationary water, as specified: no flow field. `current` events are a uniform drift force.
* `GLOW` (bioluminescence, phase 2) is not implemented; the seam is the light kernels' ray loop.
* Recombination, 3D, multi-GPU, phylogeny tree: out of scope for v1. 2D-specific math is isolated in `dims.py` and
  `ui/render.py`; some kernels still use `.x/.y` through the `vec2` type for positions.
* `can_eat_bonded` / `same_org_eating` rely on `organism_id`, which is recomputed every `organism_label_interval`
  ticks (default 10) and can be stale for up to that long.
* Genome archive pruning uses an approximate size model; it keeps `manual`-tagged and very abundant genomes.
* The tests use the CPU backend by default. `EVOSIM_DEVICE=cuda:0 pytest` runs them on the GPU.
