"""Metabolism: nutrient uptake (conservative, two-pass), photosynthesis, basal / trait / instruction
costs, trait relaxation, thrust & turn payment (pay-before-use), growth, energy cap.

Every energy change is written to the per-cell breakdown s.eb (the cell side of the ledger).
"""
import warp as wp

from ..config import SimParams
from ..dims import disc_area, grid_coord, min_image
from ..genome import isa
from ..state import (EB_BASAL, EB_BOND, EB_CAP, EB_GROWTH, EB_INSTR, EB_PHOTO, EB_TCHANGE, EB_THRUST, EB_TURN,
                     EB_UPKEEP, EB_UPTAKE, L_GROWTH_M, L_UPTAKE_F, NUM_EB, Cells, Fields, Scratch)
from .common import fx, unfx
from .light import body_radius

NT = isa.NUM_TRAITS
T_PHOTO, T_ABSORB, T_MOTOR, T_PRED, T_ARMOR = isa.T_PHOTO, isa.T_ABSORB, isa.T_MOTOR, isa.T_PRED, isa.T_ARMOR
FAIL_THRUST_ENERGY = isa.FAIL_THRUST_ENERGY
FAIL_TURN_ENERGY = isa.FAIL_TURN_ENERGY
FAIL_TRAIT_ENERGY = isa.FAIL_TRAIT_ENERGY
THRUST_EPS = 0.05


@wp.kernel
def uptake_demand_kernel(c: Cells, f: Fields, s: Scratch, p: SimParams):
    i = wp.tid()
    s.uptake_q[i] = wp.int64(0)
    if c.alive[i] == 0:
        return
    ab = c.traits[i, T_ABSORB]
    if ab <= 0.0:
        return
    fg = grid_coord(c.pos[i], p.field_cell, p.d_field_nx, p.d_field_ny)
    conc = f.nutrient[fg[0], fg[1]]
    if conc <= 0.0:
        return
    r = body_radius(c, i, p)
    amount = p.absorb_rate * ab * conc * disc_area(r) * p.dt
    q = fx(amount)
    s.uptake_q[i] = q
    wp.atomic_add(f.demand, fg[0], fg[1], q)


@wp.kernel
def uptake_grant_kernel(f: Fields, ledger: wp.array(dtype=wp.float64), p: SimParams):
    ix, iy = wp.tid()
    q = f.demand[ix, iy]
    f.grant[ix, iy] = 1.0
    if q <= wp.int64(0):
        return
    area = p.field_cell * p.field_cell
    avail = f.nutrient[ix, iy] * area
    total = unfx(q)
    frac = float(1.0)
    if total > avail:
        frac = avail / total
    f.grant[ix, iy] = frac
    taken = total * frac
    f.nutrient[ix, iy] = wp.max(0.0, f.nutrient[ix, iy] - taken / area)
    f.demand[ix, iy] = wp.int64(0)
    if p.d_ledger_on != 0:
        wp.atomic_add(ledger, L_UPTAKE_F, wp.float64(taken))


@wp.func
def in_hostile(pos: wp.vec2, p: SimParams) -> float:
    if p.d_hostile_r <= 0.0:
        return 1.0
    d = min_image(pos - wp.vec2(p.d_hostile_cx, p.d_hostile_cy), p.d_world)
    if wp.length(d) < p.d_hostile_r:
        return p.d_hostile_mult
    return 1.0


@wp.func
def upkeep_of(p: SimParams, t: int) -> float:
    if t == 0:
        return p.upkeep_PHOTO
    if t == 1:
        return p.upkeep_ABSORB
    if t == 2:
        return p.upkeep_MOTOR
    if t == 3:
        return p.upkeep_PREDATION
    if t == 4:
        return p.upkeep_ARMOR
    if t == 5:
        return p.upkeep_PIGMENT
    if t == 6:
        return p.upkeep_PHOTORECEPTOR
    if t == 7:
        return p.upkeep_ADHESION
    return p.upkeep_RECEPTOR_WIDTH


@wp.kernel
def metabolism_kernel(c: Cells, f: Fields, s: Scratch, ledger: wp.array(dtype=wp.float64), p: SimParams):
    i = wp.tid()
    for k in range(NUM_EB):
        s.eb[i, k] = 0.0
    if c.alive[i] == 0:
        return
    m = c.mass[i]
    E = c.energy[i]
    r = body_radius(c, i, p)
    area = disc_area(r)
    dt = p.dt
    fail = c.fail_flags[i]
    # ---------------------------------------------------------------- gains
    gu = float(0.0)
    if s.uptake_q[i] > wp.int64(0):
        fg = grid_coord(c.pos[i], p.field_cell, p.d_field_nx, p.d_field_ny)
        gu = unfx(s.uptake_q[i]) * f.grant[fg[0], fg[1]] * p.absorb_eff
    gp = p.photo_eff * c.traits[i, T_PHOTO] * s.light_total[i] * area * dt
    E = E + gu + gp
    s.eb[i, EB_UPTAKE] = gu
    s.eb[i, EB_PHOTO] = gp
    # ---------------------------------------------------------------- passive costs
    basal = p.basal * m * dt * in_hostile(c.pos[i], p)
    upk = float(0.0)
    for t in range(NT):
        upk = upk + upkeep_of(p, t) * c.traits[i, t]
    upk = upk * m * dt
    ins = float(s.n_exec[i]) * p.instr_energy_cost
    nb = int(0)
    for sl in range(p.max_bonds):
        if c.bonds[i, sl] >= 0:
            nb = nb + 1
    bup = float(nb) * p.bond_upkeep * dt
    E = E - basal - upk - ins - bup
    s.eb[i, EB_BASAL] = basal
    s.eb[i, EB_UPKEEP] = upk
    s.eb[i, EB_INSTR] = ins
    s.eb[i, EB_BOND] = bup
    # ---------------------------------------------------------------- trait relaxation (pay before)
    step = p.trait_change_rate * dt
    tcost = float(0.0)
    for t in range(NT):
        dlt = wp.clamp(c.trait_targets[i, t] - c.traits[i, t], -step, step)
        tcost = tcost + wp.abs(dlt) * p.trait_change_cost * m
    tscale = float(1.0)
    if tcost > 0.0:
        if tcost > wp.max(E, 0.0):
            tscale = wp.max(E, 0.0) / tcost
            fail = fail | FAIL_TRAIT_ENERGY
        for t in range(NT):
            dlt = wp.clamp(c.trait_targets[i, t] - c.traits[i, t], -step, step)
            c.traits[i, t] = wp.clamp(c.traits[i, t] + dlt * tscale, 0.0, 1.0)
        E = E - tcost * tscale
        s.eb[i, EB_TCHANGE] = tcost * tscale
    if p.trait_budget_enabled != 0:
        tot = float(0.0)
        for t in range(NT):
            tot = tot + c.traits[i, t]
        if tot > p.trait_budget and tot > 0.0:
            sc = p.trait_budget / tot
            for t in range(NT):
                c.traits[i, t] = c.traits[i, t] * sc
    # ---------------------------------------------------------------- thrust (pay before force)
    motor = c.traits[i, T_MOTOR]
    fmax = p.max_thrust * (p.motor_base + (1.0 - p.motor_base) * motor)
    F = wp.clamp(s.a_thrust[i], -fmax, fmax)
    th_cost = float(0.0)
    if F != 0.0:
        th_cost = p.thrust_cost * wp.pow(wp.abs(F), p.thrust_exp) * dt / (THRUST_EPS + motor)
        avail = wp.max(E, 0.0)
        if th_cost > avail:
            sc = wp.pow(avail / th_cost, 1.0 / p.thrust_exp)
            F = F * sc
            th_cost = avail
            fail = fail | FAIL_THRUST_ENERGY
    E = E - th_cost
    s.thrust_force[i] = F
    s.eb[i, EB_THRUST] = th_cost
    # turn
    Tq = wp.clamp(s.a_turn[i], -p.max_torque, p.max_torque)
    tu_cost = p.turn_cost * wp.abs(Tq) * dt
    avail2 = wp.max(E, 0.0)
    if tu_cost > avail2:
        if tu_cost > 0.0:
            Tq = Tq * avail2 / tu_cost
        tu_cost = avail2
        fail = fail | FAIL_TURN_ENERGY
    E = E - tu_cost
    s.turn_torque[i] = Tq
    s.eb[i, EB_TURN] = tu_cost
    # ---------------------------------------------------------------- growth
    cap = p.max_energy_per_mass * m
    if m < p.max_mass and E > p.growth_threshold * cap:
        dm = wp.min(p.growth_rate * m * dt, p.max_mass - m)
        dE = dm / p.growth_eff
        spare = E - p.growth_threshold * cap
        if dE > spare:
            dE = spare
            dm = dE * p.growth_eff
        if dm > 0.0:
            m = m + dm
            E = E - dE
            s.eb[i, EB_GROWTH] = dE
            if p.d_ledger_on != 0:
                wp.atomic_add(ledger, L_GROWTH_M, wp.float64(dm))
    # ---------------------------------------------------------------- cap
    cap = p.max_energy_per_mass * m
    if E > cap:
        s.eb[i, EB_CAP] = E - cap
        E = cap
    c.mass[i] = m
    c.energy[i] = E
    c.fail_flags[i] = fail


def compute(state, p):
    dev = state.device
    d = state.dims
    wp.launch(uptake_demand_kernel, d.N, [state.cells, state.fields, state.scratch, p], device=dev)
    wp.launch(uptake_grant_kernel, (d.fnx, d.fny), [state.fields, state.ledger, p], device=dev)
    wp.launch(metabolism_kernel, d.N, [state.cells, state.fields, state.scratch, state.ledger, p], device=dev)
