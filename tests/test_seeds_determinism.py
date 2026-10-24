import math
import os

import numpy as np
import pytest

from evosim.config import Config
from evosim.sim import Sim
from helpers import cellarr

ROOT = os.path.dirname(os.path.dirname(__file__))
NOMUT = dict(mu_point=0, mu_insert=0, mu_delete=0, mu_duplicate=0, mu_invert=0, mu_transpose=0)


def preset(name, **over):
    cfg = Config.load(os.path.join(ROOT, "presets", f"{name}.json"))
    for k, v in over.items():
        cfg.set(k, v)
    return cfg


def test_replicator_survives_and_reproduces_in_sunny_pond(device):
    cfg = preset("sunny_pond", world_w=256.0, world_h=256.0, max_cells=4096, initial_count=100, seed=3,
                 ledger_interval=25)
    sim = Sim(cfg, device=device)
    pops = []
    for _ in range(8):
        sim.step(100)
        pops.append(sim.population())
    assert pops[-1] > 5 * 100                                   # grew by an order of magnitude
    assert max(pops) <= 4096                                    # bounded by capacity
    assert sim.last_ledger["ok"] and sim.has_nan() is None
    depth = cellarr(sim, "division_depth")[cellarr(sim, "alive") != 0]
    assert depth.max() >= 3                                     # multiple generations of descendants


def test_seed_colony_forms_stable_curved_shape(device):
    cfg = Config(max_cells=512, world_w=256.0, world_h=256.0, initial_count=1,
                 initial_seed_files="seeds/seed_colony.asm", sun_I=3.0, initial_mass=8.0, initial_energy=14.0,
                 nutrient_cap=0.0, sensor_noise=0.0, ledger_interval=10, **NOMUT)
    sim = Sim(cfg, device=device)
    sim.step(700)
    c = sim.state.cells
    al = np.nonzero(c.alive.numpy())[0]
    assert len(al) == 6 and int(sim.state.bonds.alive.numpy().sum()) == 5
    st = np.sort(c.state.numpy()[al])
    assert list(st) == [1, 1, 1, 1, 1, 2]                       # 5 body cells + 1 differentiated tip
    assert float(np.abs(c.vel.numpy()[al]).max()) < 1e-3        # at rest: a stable shape
    order = sorted(al, key=lambda i: int(c.uid.numpy()[i]))
    pos = c.pos.numpy()
    d = [pos[b] - pos[a] for a, b in zip(order[:-1], order[1:])]
    dirs = [math.atan2(v[1], v[0]) for v in d]
    turns = [(y - x + math.pi) % (2 * math.pi) - math.pi for x, y in zip(dirs[:-1], dirs[1:])]
    assert np.allclose(turns, 0.5, atol=0.03)                   # curvature held at the genome-specified value
    sim.step(300)                                               # and it keeps holding
    assert int(sim.state.bonds.alive.numpy().sum()) == 5
    pigment = c.traits.numpy()[:, 5][al]
    assert pigment.max() > 0.9 and (pigment > 0.5).sum() == 1   # only the tip is pigmented (differentiation)


def test_grazer_absorbs_and_moves(device):
    cfg = Config(max_cells=1024, world_w=256.0, world_h=256.0, initial_count=20,
                 initial_seed_files="seeds/seed_grazer.asm", nutrient_cap=0.2, nutrient_regen=0.01,
                 ledger_interval=20, seed=2, sun_I=0.2)
    sim = Sim(cfg, device=device)
    start = cellarr(sim, "pos")[cellarr(sim, "alive") != 0].copy()
    sim.step(200)
    al = cellarr(sim, "alive") != 0
    assert al.sum() >= 20
    moved = cellarr(sim, "pos")[al][:len(start)]
    assert float(np.abs(cellarr(sim, "vel")[al]).max()) > 0.0
    tr = cellarr(sim, "traits")[al]
    assert tr[:, 1].mean() > 0.5                                # ABSORB trait active
    assert sim.last_ledger["ok"]


def test_predator_seed_kills_prey(device):
    cfg = Config(max_cells=1024, world_w=96.0, world_h=96.0, initial_count=0, ledger_interval=10, seed=5, **NOMUT)
    sim = Sim(cfg, device=device, seed_population=False)
    rs = np.random.default_rng(0)
    prey = sim.load_genome_file("seeds/seed_replicator.asm")
    pred = sim.load_genome_file("seeds/seed_predator.asm")
    sim.inject(60, [prey], rs)
    sim.inject(10, [pred], rs, energy=20.0)
    bites = 0
    for _ in range(300):
        sim.step(1)
        bites += int((cellarr(sim, "action_flags") & 1 << 15 != 0).sum())
    assert bites > 0                                            # predation really happens
    assert sim.last_ledger["ok"]


def _run_hash(device, seed, salt=0, ticks=60):
    cfg = Config(max_cells=2048, world_w=128.0, world_h=128.0, initial_count=40, seed=seed, rng_salt=salt,
                 initial_seed_files="seeds/seed_replicator.asm,seeds/seed_grazer.asm", initial_source="mix",
                 mu_point=0.01, mu_insert=0.1, mu_delete=0.1, ledger_interval=0, organism_label_interval=5)
    sim = Sim(cfg, device=device)
    sim.step(ticks)
    return sim.state_hash(), sim.population()


def test_determinism_same_seed_same_state(device):
    a = _run_hash(device, 11)
    b = _run_hash(device, 11)
    assert a == b


def test_different_seed_or_salt_differs(device):
    assert _run_hash(device, 11) != _run_hash(device, 12)
    assert _run_hash(device, 11, salt=0) != _run_hash(device, 11, salt=1)
