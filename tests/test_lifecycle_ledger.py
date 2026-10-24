import math

import numpy as np
import pytest

from evosim.genome import isa
from evosim.sim import LedgerError
from evosim.state import C_GENOME_FREE, C_SPLIT_FAIL_CAP
from helpers import bond_cells, cellarr, make_sim, put

SPLITTER = "LABEL 0\n SET r3 0.0\n SET r4 0.5\n SPLIT r3 r4 0\n YIELD\n JMP 0\n"
SPLITTERB = "LABEL 0\n SET r3 0.0\n SET r4 0.5\n SPLITB r3 r4 0\n YIELD\n JMP 0\n"
EATER = "LABEL 0\n SET r0 1.0\n EAT r0\n YIELD\n JMP 0\n"
NOMUT = dict(mu_point=0, mu_insert=0, mu_delete=0, mu_duplicate=0, mu_invert=0, mu_transpose=0)


def totals(sim):
    al = cellarr(sim, "alive") != 0
    return float(cellarr(sim, "mass")[al].astype(np.float64).sum()), float(cellarr(sim, "energy")[al].astype(np.float64).sum())


def test_split_never_increases_mass_or_energy(device):
    sim = make_sim(device, split_cost=0.5, min_split_mass=2.0, **NOMUT)
    (p,) = put(sim, [[100.0, 100.0]], genome=SPLITTER, mass=4.0, energy=10.0)
    m0, e0 = totals(sim)
    sim.step(1)
    m1, e1 = totals(sim)
    assert sim.population() == 2
    assert m1 == pytest.approx(m0, abs=1e-6)                       # mass is exactly divided
    assert e1 == pytest.approx(e0 - 0.5, abs=1e-5)                 # only split_cost leaves
    al = np.nonzero(cellarr(sim, "alive"))[0]
    d = [i for i in al if i != p][0]
    assert cellarr(sim, "parent_uid", [d])[0] == cellarr(sim, "uid", [p])[0]
    assert cellarr(sim, "division_depth", [d])[0] == 1
    assert cellarr(sim, "mass", [d])[0] == pytest.approx(2.0)
    assert sim.last_ledger["ok"]


def test_split_failure_flags_and_no_energy_consumed(device):
    sim = make_sim(device, split_cost=0.5, min_split_mass=6.0, **NOMUT)
    a, b = put(sim, [[50.0, 50.0], [150.0, 150.0]], genome=SPLITTER, mass=[3.0, 8.0], energy=[10.0, 0.1])
    sim.step(1)
    f = cellarr(sim, "fail_flags", [a, b])
    assert f[0] & isa.FAIL_SPLIT_MASS and f[1] & isa.FAIL_SPLIT_ENERGY
    assert sim.population() == 2
    assert cellarr(sim, "energy", [a])[0] == pytest.approx(10.0, abs=1e-4)


def test_capacity_full_fails_without_consuming_energy(device):
    sim = make_sim(device, max_cells=1024, world_w=512.0, world_h=512.0, split_cost=0.5, min_split_mass=2.0, **NOMUT)
    xs, ys = np.meshgrid(np.arange(32) * 16.0 + 8, np.arange(32) * 16.0 + 8)
    slots = put(sim, np.stack([xs.ravel(), ys.ravel()], 1), genome=SPLITTER, mass=4.0, energy=10.0)
    assert sim.population() == 1024
    e0 = totals(sim)[1]
    sim.step(1)
    assert sim.population() == 1024
    assert (cellarr(sim, "fail_flags", slots) & isa.FAIL_CAPACITY).all()
    assert totals(sim)[1] == pytest.approx(e0, abs=1e-2)
    assert int(sim.state.counters.numpy()[C_SPLIT_FAIL_CAP]) == 1024


def test_partial_capacity_first_requests_win(device):
    sim = make_sim(device, max_cells=1024, world_w=512.0, world_h=512.0, split_cost=0.5, min_split_mass=2.0, **NOMUT)
    xs, ys = np.meshgrid(np.arange(32) * 16.0 + 8, np.arange(32) * 16.0 + 8)
    slots = put(sim, np.stack([xs.ravel(), ys.ravel()], 1)[:1020], genome=SPLITTER, mass=4.0, energy=10.0)
    sim.step(1)
    assert sim.population() == 1024
    ok = (cellarr(sim, "action_flags", slots) & isa.ACT_SPLIT_OK) != 0
    assert ok.sum() == 4 and ok[:4].all()                            # deterministic: lowest indices win


def test_cow_genome_sharing_and_gc(device):
    sim = make_sim(device, split_cost=0.1, min_split_mass=2.0, **NOMUT)
    (p,) = put(sim, [[100.0, 100.0]], genome=SPLITTER, mass=4.0, energy=10.0)
    gid = cellarr(sim, "genome_id", [p])[0]
    free0 = int(sim.state.counters.numpy()[C_GENOME_FREE])
    sim.step(1)
    al = np.nonzero(cellarr(sim, "alive"))[0]
    gids = cellarr(sim, "genome_id", al)
    assert (gids == gid).all()                                       # no mutation -> shared entry
    assert sim.state.genomes.refcount.numpy()[gid] == 2
    assert int(sim.state.counters.numpy()[C_GENOME_FREE]) == free0   # the tentative slot was returned
    # now with certain mutation: daughters get their own entry
    sim.cfg.mu_point = 0.2
    sim.cfg.mu_insert = 1.0
    sim.step(1)
    gids2 = cellarr(sim, "genome_id", np.nonzero(cellarr(sim, "alive"))[0])
    assert len(set(gids2)) > 1
    # kill everyone with that genome -> refcounts drop, entries are collected
    sim.kill(np.nonzero(cellarr(sim, "alive"))[0])
    sim.step(1)
    assert sim.population() == 0
    assert (sim.state.genomes.refcount.numpy()[sim.state.genomes.in_use.numpy() != 0] > 0).all()
    assert int(sim.state.genomes.in_use.numpy().sum()) == 0


def test_splitb_bonds_parent_and_daughter(device):
    sim = make_sim(device, split_cost=0.1, min_split_mass=2.0, **NOMUT)
    (p,) = put(sim, [[100.0, 100.0]], genome=SPLITTERB, mass=4.0, energy=10.0)
    sim.step(1)
    assert int(sim.state.bonds.alive.numpy().sum()) == 1
    al = np.nonzero(cellarr(sim, "alive"))[0]
    assert (cellarr(sim, "bonds", al) >= 0).sum() == 2
    b = np.nonzero(sim.state.bonds.alive.numpy())[0][0]
    assert sim.state.bonds.rest_len.numpy()[b] == pytest.approx(2 * math.sqrt(2.0 / math.pi), rel=1e-4)


def test_predation_bite_gain_and_ledger(device):
    sim = make_sim(device, bite_rate=1.0, bite_eff=0.7, engulf_ratio=100.0, armor_factor=0.8, dt=1.0)
    pred, prey = put(sim, [[100.0, 100.0], [101.5, 100.0]], genome=[EATER, "LABEL 0\n YIELD\n JMP 0\n"][0],
                     mass=4.0, energy=10.0, traits={"PREDATION": 1.0})
    sim.step(1)
    # both cells run EATER and bite each other (non-bonded, different organisms) -> symmetric trade
    m = cellarr(sim, "mass", [pred, prey])
    assert m[0] == pytest.approx(4.0 - 1.0, abs=1e-3) or m[0] < 4.0
    assert sim.last_ledger["ok"]


def test_predator_one_way_bite_exact(device):
    passive = "LABEL 0\n YIELD\n JMP 0\n"
    sim = make_sim(device, bite_rate=1.0, bite_eff=0.7, engulf_ratio=100.0, armor_factor=0.8)
    from evosim import spawn
    from evosim.genome import assembler
    g_e, g_p = spawn.upload_genomes(sim, [assembler.assemble(EATER), assembler.assemble(passive)])
    pred = put(sim, [[100.0, 100.0]], gids=np.array([g_e]), mass=4.0, energy=10.0, traits={"PREDATION": 1.0})[0]
    prey = put(sim, [[101.5, 100.0]], gids=np.array([g_p]), mass=4.0, energy=8.0)[0]
    sim.step(1)
    assert cellarr(sim, "mass", [prey])[0] == pytest.approx(3.0, abs=1e-4)           # lost bite_rate*PRED*EAT*dt
    assert cellarr(sim, "energy", [prey])[0] == pytest.approx(8.0 * 0.75, abs=1e-3)  # energy lost in proportion
    assert cellarr(sim, "energy", [pred])[0] == pytest.approx(10.0 + 0.7 * 1.0, abs=1e-3)
    assert cellarr(sim, "damage", [prey])[0] == pytest.approx(1.0, abs=1e-4)         # DAMAGE sensor input
    assert cellarr(sim, "action_flags", [prey])[0] & isa.ACT_BITTEN
    assert sim.last_ledger["ok"]


def test_armor_reduces_bite_and_engulf(device):
    passive = "LABEL 0\n YIELD\n JMP 0\n"
    from evosim import spawn
    from evosim.genome import assembler
    sim = make_sim(device, bite_rate=1.0, bite_eff=0.7, engulf_ratio=3.0, armor_factor=0.8)
    g_e, g_p = spawn.upload_genomes(sim, [assembler.assemble(EATER), assembler.assemble(passive)])
    pred = put(sim, [[100.0, 100.0]], gids=np.array([g_e]), mass=4.0, energy=10.0, traits={"PREDATION": 1.0})[0]
    prey = put(sim, [[101.5, 100.0]], gids=np.array([g_p]), mass=3.0, energy=8.0, traits={"ARMOR": 1.0})[0]
    sim.step(1)
    assert cellarr(sim, "mass", [prey])[0] == pytest.approx(3.0 - 0.2, abs=1e-3)      # 80% reduction
    # engulf: predator much heavier than prey -> consumed in one tick
    sim2 = make_sim(device, bite_rate=0.1, bite_eff=0.7, engulf_ratio=3.0)
    g_e, g_p = spawn.upload_genomes(sim2, [assembler.assemble(EATER), assembler.assemble(passive)])
    pred = put(sim2, [[100.0, 100.0]], gids=np.array([g_e]), mass=20.0, energy=10.0, traits={"PREDATION": 1.0})[0]
    prey = put(sim2, [[103.0, 100.0]], gids=np.array([g_p]), mass=3.0, energy=8.0)[0]
    sim2.step(1)
    assert cellarr(sim2, "alive", [prey])[0] == 0
    assert sim2.last_ledger["ok"]


def test_simultaneous_bites_scale_down(device):
    passive = "LABEL 0\n YIELD\n JMP 0\n"
    from evosim import spawn
    from evosim.genome import assembler
    sim = make_sim(device, bite_rate=1.5, bite_eff=1.0, engulf_ratio=100.0, min_mass=0.01, detritus_fraction=0.0)
    g_e, g_p = spawn.upload_genomes(sim, [assembler.assemble(EATER), assembler.assemble(passive)])
    preds = put(sim, [[100.0, 100.0], [102.8, 100.0]], gids=np.array([g_e, g_e]), mass=4.0, energy=10.0,
                traits={"PREDATION": 1.0})
    # prey touches both predators; the predators also touch each other (same-organism ids differ -> they bite too)
    sim.cfg.can_eat_bonded = False
    prey = put(sim, [[101.4, 100.6]], gids=np.array([g_p]), mass=1.0, energy=4.0)[0]
    sim.step(1)
    assert cellarr(sim, "alive", [prey])[0] == 0 or cellarr(sim, "mass", [prey])[0] < 1e-6
    assert sim.last_ledger["ok"]                                       # demand 3.0 > mass 1.0, scaled, still balanced


def test_nutrient_uptake_is_conservative_and_proportional(device):
    sim = make_sim(device, nutrient_cap=0.0, absorb_rate=5.0, absorb_eff=2.0, field_cell=4.0)
    f = sim.state.fields
    arr = np.zeros((sim.state.dims.fnx, sim.state.dims.fny), np.float32)
    arr[25, 25] = 0.02                                   # tiny pool: 0.02 * 16 = 0.32 units
    f.nutrient.assign(arr)
    a, b = put(sim, [[102.0, 102.0], [101.0, 101.0]], traits={"ABSORB": [1.0, 0.5][0] and {"ABSORB": 1.0}["ABSORB"]},
               mass=3.0, energy=1.0)
    sim.step(1)
    e = cellarr(sim, "energy", [a, b]) - 1.0
    left = float(f.nutrient.numpy()[25, 25]) * 16.0
    assert left >= 0.0
    taken = 0.32 - left
    assert e.sum() == pytest.approx(2.0 * taken, rel=1e-3)               # gain == absorb_eff * removed
    assert e[0] == pytest.approx(e[1], rel=1e-3)                         # equal demand -> equal share
    assert sim.last_ledger["ok"]


def test_death_deposits_detritus_and_ledger(device):
    sim = make_sim(device, detritus_fraction=0.5, min_mass=1.0)
    (s,) = put(sim, [[102.0, 102.0]], genome="LABEL 0\n DIE\n YIELD\n JMP 0\n", mass=4.0, energy=3.0)
    n0 = float(sim.state.fields.nutrient.numpy().astype(np.float64).sum()) * 16.0
    sim.step(1)
    assert sim.population() == 0
    n1 = float(sim.state.fields.nutrient.numpy().astype(np.float64).sum()) * 16.0
    assert n1 - n0 == pytest.approx(0.5 * 4.0, rel=1e-4)
    assert sim.last_ledger["ok"]
    assert int(sim.state.counters.numpy()[0]) == sim.state.dims.N        # the slot came back to the free list


def test_apoptosis_passes_energy_to_neighbours(device):
    sim = make_sim(device, share_eff=0.9, break_strain=100.0, apoptosis_share=True)
    a, b = put(sim, [[100.0, 100.0], [106.0, 100.0]], mass=3.0, energy=10.0)
    from evosim import spawn
    from evosim.genome import assembler
    (g,) = spawn.upload_genomes(sim, [assembler.assemble("LABEL 0\n DIE\n YIELD\n JMP 0\n")])
    gi = cellarr(sim, "genome_id").copy()
    gi[a] = g
    sim.state.cells.genome_id.assign(gi)
    bond_cells(sim, [(0, 1)], [a, b])
    sim.step(1)
    assert cellarr(sim, "alive", [a])[0] == 0
    assert cellarr(sim, "energy", [b])[0] == pytest.approx(10.0 + 0.9 * 10.0, abs=1e-3)
    assert sim.last_ledger["ok"]
    assert (cellarr(sim, "bonds", [b]) == -1).all()


@pytest.mark.parametrize("mode,mutual,expect", [("unilateral", False, True), ("mutual", False, False),
                                                ("mutual", True, True)])
def test_bond_creation_consent(device, mode, mutual, expect):
    bond_f = "LABEL 0\n BOND F\n YIELD\n JMP 0\n"
    idle = "LABEL 0\n YIELD\n JMP 0\n"
    from evosim import spawn
    from evosim.genome import assembler
    sim = make_sim(device, bond_consent_mode=mode, bond_create_cost=0.5)
    gb, gi = spawn.upload_genomes(sim, [assembler.assemble(bond_f), assembler.assemble(idle)])
    # a faces +x toward b; b faces -x toward a
    a = put(sim, [[100.0, 100.0]], gids=np.array([gb]), mass=3.0, energy=5.0, angle=[0.0])[0]
    b = put(sim, [[101.5, 100.0]], gids=np.array([gb if mutual else gi]), mass=3.0, energy=5.0, angle=[math.pi])[0]
    sim.step(1)
    n = int(sim.state.bonds.alive.numpy().sum())
    assert (n == 1) == expect
    if not expect:
        assert cellarr(sim, "fail_flags", [a])[0] & isa.FAIL_BOND_CONSENT
    else:
        assert cellarr(sim, "energy", [a])[0] <= 5.0 - 0.5 + 1e-4
    assert sim.last_ledger["ok"]


def test_bond_no_contact_flag(device):
    sim = make_sim(device)
    (a,) = put(sim, [[100.0, 100.0]], genome="LABEL 0\n BOND F\n YIELD\n JMP 0\n", mass=3.0, energy=5.0)
    sim.step(1)
    assert cellarr(sim, "fail_flags", [a])[0] & isa.FAIL_BOND_NOCONTACT


def test_touch_sensor_sectors(device):
    probe = "LABEL 0\n SENSE r0 TOUCH_F\n SIGNAL S0 r0\n SENSE r1 TOUCH_L\n SIGNAL S1 r1\n SENSE r2 TOUCH_B\n SIGNAL S2 r2\n YIELD\n JMP 0\n"
    sim = make_sim(device, signal_cost=0.0)
    a = put(sim, [[100.0, 100.0]], genome=probe, mass=3.0, energy=5.0, angle=[0.0])[0]
    put(sim, [[101.5, 100.0]], mass=3.0, energy=5.0)         # directly in front
    sim.step(1)
    so = cellarr(sim, "signal_out", [a])[0]
    assert so[0] > 0 and so[1] == 0 and so[2] == 0


def test_chemical_emit_diffusion_and_decay_conserve(device):
    emit = "LABEL 0\n SET r0 2.0\n EMIT C1 r0\n YIELD\n JMP 0\n"
    sim = make_sim(device, emit_cost=0.1, D_chem_1=2.0, chem_decay_1=0.0)
    (s,) = put(sim, [[100.0, 100.0]], genome=emit, mass=3.0, energy=10.0)
    sim.step(1)
    ch = sim.state.fields.chem.numpy().astype(np.float64)
    assert ch[1].sum() * 16.0 == pytest.approx(2.0, rel=1e-4)
    assert cellarr(sim, "energy", [s])[0] == pytest.approx(10.0 - 0.2, abs=1e-4)
    sim.kill([s]); sim.step(1)                       # it still emits once more before the death phase
    total = sim.state.fields.chem.numpy().astype(np.float64)[1].sum() * 16.0
    assert total == pytest.approx(4.0, rel=1e-4)
    sim.step(20)
    assert sim.state.fields.chem.numpy().astype(np.float64)[1].sum() * 16.0 == pytest.approx(4.0, rel=1e-4)
    assert sim.state.fields.chem.numpy()[1].max() < 2.0 / 16.0      # it spread out
    sim.cfg.chem_decay_1 = 0.1
    before = sim.state.fields.chem.numpy().astype(np.float64)[1].sum()
    sim.step(10)
    assert sim.state.fields.chem.numpy().astype(np.float64)[1].sum() == pytest.approx(before * 0.9 ** 10, rel=1e-3)


def test_ledger_detects_a_planted_energy_leak(device):
    sim = make_sim(device, ledger_check=True)
    (s,) = put(sim, [[100.0, 100.0]], mass=3.0, energy=5.0)
    sim.step(1)
    e = cellarr(sim, "energy").copy()
    # emulate a buggy kernel creating free energy between ledger snapshots: poke it inside a tick
    orig = sim._check_ledger

    def leaky(before, interv):
        e2 = cellarr(sim, "energy").copy()
        e2[s] += 1.0
        sim.state.cells.energy.assign(e2)
        return orig(before, interv)
    sim._check_ledger = leaky
    with pytest.raises(LedgerError):
        sim.step(1)


def test_long_mixed_run_balances_ledger(device):
    """Replicators + grazers + predators with heavy mutation: every tick's ledger must balance."""
    from evosim.config import Config
    from evosim.sim import Sim
    cfg = Config(max_cells=2048, world_w=192.0, world_h=192.0, initial_count=60, initial_source="seed",
                 initial_seed_files="seeds/seed_replicator.asm,seeds/seed_grazer.asm,seeds/seed_predator.asm,seeds/seed_colony.asm",
                 mu_point=0.01, mu_insert=0.1, mu_delete=0.1, mu_duplicate=0.05, ledger_interval=1,
                 nutrient_cap=0.1, nutrient_pattern="patches", seed=7, organism_label_interval=5)
    sim = Sim(cfg, device=device)
    sim.step(250)                                          # raises LedgerError on any imbalance
    assert sim.ledger_failures == 0 and sim.last_ledger["ok"]
    assert sim.has_nan() is None
