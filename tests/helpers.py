"""Test helpers: build small scenes with hand-placed cells."""
import math

import numpy as np

from evosim.config import Config
from evosim.genome import assembler, isa
from evosim.sim import Sim

IDLE = "LABEL 0\n YIELD\n JMP 0\n"


def make_sim(device, **kw):
    base = dict(max_cells=2048, world_w=256.0, world_h=256.0, initial_count=0, ledger_interval=1,
                sensor_noise=0.0, basal=0.0, instr_energy_cost=0.0, nutrient_cap=0.0, nutrient_regen=0.0,
                max_energy_per_mass=100.0, growth_rate=0.0, rest_length_tracks_radius=False,
                bond_upkeep=0.0)
    base.update({f"upkeep_{n}": 0.0 for n in isa.TRAIT_NAMES})
    base.update(kw)
    cfg = Config(**base)
    return Sim(cfg, device=device, seed_population=False)


def put(sim, pos, genome=IDLE, angle=None, traits=None, mass=None, energy=None, gids=None, **extra):
    """Place cells. traits: (k, 9) array or dict name->value. Returns slots."""
    from evosim import spawn
    pos = np.asarray(pos, np.float32).reshape(-1, 2)
    k = len(pos)
    if gids is None:
        words = assembler.assemble(genome) if isinstance(genome, str) else genome
        gids = np.repeat(spawn.upload_genomes(sim, [words]), k)
    ent = dict(pos=pos, gid=gids, angle=np.zeros(k) if angle is None else np.asarray(angle, np.float32))
    if mass is not None:
        ent["mass"] = np.broadcast_to(np.asarray(mass, np.float32), (k,)).copy()
    if energy is not None:
        ent["energy"] = np.broadcast_to(np.asarray(energy, np.float32), (k,)).copy()
    if traits is not None:
        if isinstance(traits, dict):
            t = np.zeros((k, isa.NUM_TRAITS), np.float32)
            for n, v in traits.items():
                t[:, isa.TRAIT_NAMES.index(n)] = v
        else:
            t = np.asarray(traits, np.float32).reshape(k, isa.NUM_TRAITS)
        ent["traits"] = t
        ent["targets"] = t
    ent.update(extra)
    return spawn.spawn_cells(sim, ent)


def cellarr(sim, name, slots=None):
    a = getattr(sim.state.cells, name).numpy()
    return a if slots is None else a[slots]


def bond_cells(sim, pairs, slots):
    """Bond cells by (i, j) indices into `slots`, with rest geometry taken from the current layout."""
    from evosim import spawn
    c = sim.state.cells
    pos = c.pos.numpy()
    ang = c.angle.numpy()
    W, H = sim.state.dims.world
    used = {int(s): 0 for s in slots}
    out = []
    for i, j in pairs:
        a, b = int(slots[i]), int(slots[j])
        d = pos[b] - pos[a]
        d[0] -= W * np.round(d[0] / W)
        d[1] -= H * np.round(d[1] / H)
        r = np.linalg.norm(d)
        wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
        out.append((a, b, used[a], used[b], float(r), float(wrap(math.atan2(d[1], d[0]) - ang[a])),
                    float(wrap(math.atan2(-d[1], -d[0]) - ang[b])), 1.0))
        used[a] += 1
        used[b] += 1
    spawn.insert_bonds(sim, out)
