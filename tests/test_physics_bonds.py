import math

import numpy as np
import pytest

from evosim.genome import isa
from helpers import bond_cells, cellarr, make_sim, put


def _ring(sim, n=6, R=6.0, mass=3.0, center=(128.0, 128.0), genome=None):
    th = np.arange(n) * 2 * math.pi / n
    pos = np.array(center) + R * np.stack([np.cos(th), np.sin(th)], 1)
    kw = {} if genome is None else dict(genome=genome)
    slots = put(sim, pos, angle=th + math.pi / 2, mass=mass, energy=50.0, **kw)
    bond_cells(sim, [(i, (i + 1) % n) for i in range(n)], slots)
    return slots


def test_internal_forces_conserve_momentum_and_com(device):
    """Newton's 3rd law: with no drag/thrust a perturbed bonded organism keeps momentum 0 and its COM fixed."""
    sim = make_sim(device, drag_lin=0.0, drag_quad=0.0, ang_drag=0.0, v_max=100.0, w_max=100.0,
                   break_strain=100.0, k_bond=5.0, k_bond_angle=1.0, instr_per_tick=1)
    slots = _ring(sim)
    rs = np.random.default_rng(0)
    v = rs.normal(0, 0.05, (len(slots), 2)).astype(np.float32)
    v -= v.mean(0)                              # zero total momentum (equal masses)
    full = cellarr(sim, "vel").copy()
    full[slots] = v
    sim.state.cells.vel.assign(full)
    com0 = cellarr(sim, "pos", slots).astype(np.float64).mean(0)
    pmax = 0.0
    for _ in range(300):
        sim.step(1)
        P = (cellarr(sim, "vel", slots).astype(np.float64) * cellarr(sim, "mass", slots)[:, None]).sum(0)
        pmax = max(pmax, float(np.abs(P).max()))
    com = cellarr(sim, "pos", slots).astype(np.float64).mean(0)
    assert pmax < 1e-4
    assert np.abs(com - com0).max() < 1e-3


def test_torque_driven_organism_cannot_translate(device):
    """Cells that spin themselves (TURN) back and forth cannot move the COM without THRUST."""
    g = "LABEL 0\n SENSE r0 TOUCH_TOTAL\n RAND r1\n SET r2 0.5\n SUB r1 r1 r2\n SET r2 8.0\n MUL r1 r1 r2\n TURN r1\n YIELD\n JMP 0\n"
    sim = make_sim(device, drag_lin=0.0, ang_drag=0.0, v_max=100.0, w_max=100.0, break_strain=100.0,
                   turn_cost=0.0, max_torque=5.0)
    slots = _ring(sim, genome=g)
    com0 = cellarr(sim, "pos", slots).astype(np.float64).mean(0)
    for _ in range(400):
        sim.step(1)
    assert np.abs(cellarr(sim, "ang_vel", slots)).max() > 1e-3          # it really was rotating
    com = cellarr(sim, "pos", slots).astype(np.float64).mean(0)
    assert np.abs(com - com0).max() < 2e-3


def test_thrust_does_translate(device):
    g = "LABEL 0\n SET r0 1.0\n THRUST r0\n YIELD\n JMP 0\n"
    sim = make_sim(device, drag_lin=0.5, thrust_cost=0.0, max_thrust=2.0, motor_base=1.0)
    (s,) = put(sim, [[100.0, 100.0]], genome=g, mass=3.0, energy=50.0)
    sim.step(30)                                   # (short: the world is a torus)
    assert 105.0 < cellarr(sim, "pos", [s])[0, 0] < 200.0


def test_thrust_cost_formula_and_pay_before_use(device):
    g = "LABEL 0\n SET r0 1.0\n THRUST r0\n YIELD\n JMP 0\n"
    sim = make_sim(device, thrust_cost=0.1, thrust_exp=2.0, max_thrust=2.0, motor_base=1.0, dt=1.0)
    rich, poor = put(sim, [[50.0, 50.0], [150.0, 150.0]], genome=g, mass=3.0, energy=[10.0, 0.001],
                     traits={"MOTOR": 0.0})
    # MOTOR trait is 0 for both: cost = k |F|^p dt / (eps + MOTOR) with eps = 0.05
    sim.step(1)
    e = cellarr(sim, "energy")
    assert 10.0 - e[rich] == pytest.approx(0.1 * 1.0 / 0.05 * 1.0, rel=1e-3)
    fail = cellarr(sim, "fail_flags")
    assert fail[poor] & isa.FAIL_THRUST_ENERGY            # not enough energy -> scaled down, flagged
    assert e[poor] >= 0.0                                  # never pushed below zero


def test_colony_curve_is_held(device):
    """Angular springs keep a curved arc curved: 6 cells bonded with a constant 0.5 rad turn per link."""
    sim = make_sim(device, break_strain=100.0, k_bond_angle=1.0)
    n = 6
    ang = [0.5 * i for i in range(n)]
    pos = [[100.0, 100.0]]
    for i in range(n - 1):                              # each link points along heading_i + 0.5 = heading_{i+1}
        pos.append([pos[-1][0] + 6.0 * math.cos(ang[i + 1]), pos[-1][1] + 6.0 * math.sin(ang[i + 1])])
    slots = put(sim, pos, angle=ang, mass=3.0, energy=50.0)
    bond_cells(sim, [(i, i + 1) for i in range(n - 1)], slots)
    rs = np.random.default_rng(1)
    p = cellarr(sim, "pos").copy()
    p[slots] += rs.normal(0, 0.4, (n, 2)).astype(np.float32)           # shake it
    sim.state.cells.pos.assign(p)
    sim.step(600)
    P = cellarr(sim, "pos", slots)
    dirs = np.arctan2(*(P[1:] - P[:-1]).T[::-1])
    turns = (np.diff(dirs) + math.pi) % (2 * math.pi) - math.pi
    assert np.allclose(turns, 0.5, atol=0.06)
    assert int(sim.state.bonds.alive.numpy().sum()) == n - 1


def test_floppy_without_angular_stiffness(device):
    """Control: with k_bond_angle = 0 the same shaken chain loses its curvature."""
    sim = make_sim(device, break_strain=100.0, k_bond_angle=0.0)
    n = 6
    ang = [0.5 * i for i in range(n)]
    pos = [[100.0, 100.0]]
    for i in range(n - 1):
        pos.append([pos[-1][0] + 6.0 * math.cos(ang[i + 1]), pos[-1][1] + 6.0 * math.sin(ang[i + 1])])
    slots = put(sim, pos, angle=ang, mass=3.0, energy=50.0)
    bond_cells(sim, [(i, i + 1) for i in range(n - 1)], slots)
    p = cellarr(sim, "pos").copy()
    p[slots] += np.random.default_rng(2).normal(0, 1.5, (n, 2)).astype(np.float32)
    sim.state.cells.pos.assign(p)
    sim.step(600)
    P = cellarr(sim, "pos", slots)
    dirs = np.arctan2(*(P[1:] - P[:-1]).T[::-1])
    turns = (np.diff(dirs) + math.pi) % (2 * math.pi) - math.pi
    assert not np.allclose(turns, 0.5, atol=0.06)


def test_bond_breaks_on_strain_and_death(device):
    sim = make_sim(device, break_strain=0.3)
    a, b = put(sim, [[100.0, 100.0], [106.0, 100.0]], mass=3.0, energy=50.0)
    bond_cells(sim, [(0, 1)], [a, b])
    p = cellarr(sim, "pos").copy()
    p[b] = [112.0, 100.0]                                # stretch to strain ~1
    sim.state.cells.pos.assign(p)
    sim.step(3)
    assert int(sim.state.bonds.alive.numpy().sum()) == 0
    assert (cellarr(sim, "bonds", [a, b]) == -1).all()


def test_contacts_conserve_momentum_and_wrap(device):
    """Two cells colliding across the world edge exchange equal and opposite momentum."""
    sim = make_sim(device, drag_lin=0.0, ang_drag=0.0, v_max=100.0)
    a, b = put(sim, [[1.0, 100.0], [255.0, 100.0]], mass=3.0, energy=50.0)    # 2 apart through the seam
    v = cellarr(sim, "vel").copy()
    v[a] = [-0.3, 0]
    v[b] = [0.3, 0]
    sim.state.cells.vel.assign(v)
    p0 = (cellarr(sim, "vel", [a, b]) * 3.0).sum(0)
    sim.step(50)
    P = (cellarr(sim, "vel", [a, b]) * 3.0).sum(0)
    assert np.abs(P - p0).max() < 1e-5
    assert np.all(cellarr(sim, "pos") >= 0) and np.all(cellarr(sim, "pos") < 256)


def test_organism_labeling_and_clone_preserve_geometry(device):
    sim = make_sim(device, break_strain=100.0)
    n = 5
    pos = [[100.0 + 7.0 * i, 100.0 + 2.0 * (i % 2)] for i in range(n)]
    slots = put(sim, pos, angle=np.linspace(0, 1, n), mass=3.0, energy=20.0)
    loner = put(sim, [[40.0, 40.0]], mass=3.0, energy=20.0)
    bond_cells(sim, [(i, i + 1) for i in range(n - 1)], slots)
    from evosim.kernels import organisms
    organisms.label(sim.state)
    org = cellarr(sim, "organism_id")
    assert len(set(org[slots])) == 1 and org[loner[0]] != org[slots[0]]
    rs = np.random.default_rng(0)
    new = sim.clone_organism(int(slots[2]), rs)
    assert len(new) == n
    P0 = cellarr(sim, "pos", slots).astype(np.float64)
    P1 = cellarr(sim, "pos", new).astype(np.float64)
    d0 = P0[:, None] - P0[None]
    d1 = P1[:, None] - P1[None]
    d1 -= 256 * np.round(d1 / 256)
    assert np.allclose(np.linalg.norm(d0, axis=-1), np.linalg.norm(d1, axis=-1), atol=1e-3)
    assert np.allclose(cellarr(sim, "angle", slots), cellarr(sim, "angle", new), atol=1e-6)
    bd = sim.state.bonds
    al = bd.alive.numpy() != 0
    assert al.sum() == 2 * (n - 1)
    ba, bb = bd.a.numpy()[al], bd.b.numpy()[al]
    old_set, new_set = set(map(int, slots)), set(map(int, new))
    for k in range(len(ba)):
        assert (int(ba[k]) in old_set) == (int(bb[k]) in old_set)          # no bond joins original and clone
    organisms.label(sim.state)
    org = cellarr(sim, "organism_id")
    assert len(set(org[new])) == 1 and org[new[0]] != org[slots[0]]
    r_old = sorted(round(float(x), 3) for x in bd.rest_len.numpy()[al][[int(i) in old_set for i in ba]])
    r_new = sorted(round(float(x), 3) for x in bd.rest_len.numpy()[al][[int(i) in new_set for i in ba]])
    assert r_old == r_new


def test_share_and_signal_delay(device):
    send = "LABEL 0\n SET r0 7.0\n SIGNAL S0 r0\n SET r1 0.5\n SHARE r1\n YIELD\n JMP 0\n"
    recv = "LABEL 0\n RECV r0 S0\n SIGNAL S1 r0\n YIELD\n JMP 0\n"
    sim = make_sim(device, share_eff=0.8, signal_cost=0.0, recv_mode="sum", break_strain=100.0, drag_lin=1.0)
    (a,) = put(sim, [[100.0, 100.0]], genome=send, mass=3.0, energy=10.0)
    (b,) = put(sim, [[106.0, 100.0]], genome=recv, mass=3.0, energy=2.0)
    bond_cells(sim, [(0, 1)], [a, b])
    sim.step(1)
    # one-tick delay: during tick 0 the receiver sees the PREVIOUS tick's signal_out (zero)
    assert cellarr(sim, "signal_out", [b])[0, 1] == 0.0
    # SHARE: a gave 50% of its energy, b received 80% of it
    ea, eb = cellarr(sim, "energy", [a, b])
    assert ea == pytest.approx(5.0, abs=0.05) and eb == pytest.approx(2.0 + 0.8 * 5.0, abs=0.05)
    sim.step(1)
    assert cellarr(sim, "signal_out", [b])[0, 1] == pytest.approx(7.0)
