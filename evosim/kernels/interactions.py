"""Interactions: predation, SHARE (and apoptosis energy hand-off), EMIT / passive leak, SIGNAL
costs, UNBOND requests, BOND creation.

Determinism: scattered sums use int64 fixed point (order independent); BOND conflicts are resolved
with atomic_min locks on integer indices (order independent); bond slots come from a prefix scan.

Order matters for conservation: SHARE runs while the bond topology is still the one SHARE amounts
were computed for (bond creation comes last; bond removal happens in the death phase).
"""
import warp as wp

from .. import rng
from ..config import SimParams
from ..dims import direction_angle, grid_coord, min_image, wrap_angle
from ..genome import isa
from ..state import (C_BOND_FREE, C_BOND_NEW_TOTAL, EB_BOND, EB_EMIT, EB_PRED, EB_SHARE_IN, EB_SHARE_OUT, EB_SIGNAL,
                     EB_VICTIM, EB_BASAL, L_EMIT_CHEM, L_LEAK_CHEM, L_PRED_M, L_VICTIM_M, MAX_CONTACTS, Bonds,
                     Cells, Fields, Scratch)
from . import common
from .common import fx, pop_slot, unfx
from .light import body_radius
from .sense import other_end

T_PRED, T_ARMOR, T_ADHESION = isa.T_PRED, isa.T_ARMOR, isa.T_ADHESION
ACT_DIE, ACT_BITTEN, ACT_BOND_OK = isa.ACT_DIE, isa.ACT_BITTEN, isa.ACT_BOND_OK
FAIL_EAT_NOTRAIT, FAIL_EAT_NOCONTACT = isa.FAIL_EAT_NOTRAIT, isa.FAIL_EAT_NOCONTACT
FAIL_SHARE_NOBOND, FAIL_EMIT_ENERGY, FAIL_SIGNAL_ENERGY = isa.FAIL_SHARE_NOBOND, isa.FAIL_EMIT_ENERGY, isa.FAIL_SIGNAL_ENERGY
FAIL_UNBOND_EMPTY = isa.FAIL_UNBOND_EMPTY
FAIL_BOND_NOCONTACT, FAIL_BOND_NOSLOT, FAIL_BOND_CONSENT = isa.FAIL_BOND_NOCONTACT, isa.FAIL_BOND_NOSLOT, isa.FAIL_BOND_CONSENT
FAIL_BOND_ENERGY, FAIL_BOND_CONFLICT, FAIL_BOND_ALREADY = isa.FAIL_BOND_ENERGY, isa.FAIL_BOND_CONFLICT, isa.FAIL_BOND_ALREADY
P_BOND = rng.P_BOND
BIG = 2147483647


@wp.func
def bonded_to(c: Cells, bd: Bonds, i: int, j: int, mb: int) -> int:
    for sl in range(mb):
        b = c.bonds[i, sl]
        if b >= 0:
            if other_end(bd, b, i) == j:
                return 1
    return 0


@wp.func
def bond_count(c: Cells, i: int, mb: int) -> int:
    n = int(0)
    for sl in range(mb):
        if c.bonds[i, sl] >= 0:
            n = n + 1
    return n


@wp.func
def free_slot(c: Cells, i: int, mb: int) -> int:
    for sl in range(mb):
        if c.bonds[i, sl] < 0:
            return sl
    return -1


# ----------------------------------------------------------------------------- predation
@wp.kernel
def pred_demand_kernel(c: Cells, bd: Bonds, s: Scratch, p: SimParams):
    i = wp.tid()
    for k in range(MAX_CONTACTS):
        s.bite_amt[i, k] = 0.0
    if c.alive[i] == 0:
        return
    eat = wp.clamp(s.a_eat[i], 0.0, 1.0)
    if eat <= 0.0:
        return
    pred = c.traits[i, T_PRED]
    if pred <= 0.0 or p.predation_enabled == 0:
        c.fail_flags[i] = c.fail_flags[i] | FAIL_EAT_NOTRAIT
        return
    mi = c.mass[i]
    any_bite = int(0)
    for k in range(s.n_contacts[i]):
        j = s.contacts[i, k]
        ok = int(1)
        if c.alive[j] == 0:
            ok = int(0)
        if ok == 1 and p.can_eat_bonded == 0:
            if bonded_to(c, bd, i, j, p.max_bonds) != 0:
                ok = int(0)
        if ok == 1 and p.same_org_eating == 0:
            if c.organism_id[j] == c.organism_id[i]:
                ok = int(0)
        if ok == 1:
            mj = c.mass[j]
            bite = p.bite_rate * pred * eat * p.dt * (1.0 - p.armor_factor * c.traits[j, T_ARMOR])
            if mi > p.engulf_ratio * mj:
                bite = mj
            bite = wp.clamp(bite, 0.0, mj)
            if bite > 0.0:
                s.bite_amt[i, k] = bite
                wp.atomic_add(s.bite_demand, j, fx(bite))
                any_bite = int(1)
    if any_bite == 0:
        c.fail_flags[i] = c.fail_flags[i] | FAIL_EAT_NOCONTACT


@wp.kernel
def pred_resolve_kernel(c: Cells, s: Scratch, ledger: wp.array(dtype=wp.float64), p: SimParams):
    j = wp.tid()
    s.bite_frac[j] = 0.0
    if c.alive[j] == 0:
        return
    q = s.bite_demand[j]
    s.bite_demand[j] = wp.int64(0)
    if q <= wp.int64(0):
        c.damage[j] = 0.0
        return
    mj = c.mass[j]
    total = unfx(q)
    frac = float(1.0)
    if total > mj:
        frac = mj / total
    loss = total * frac
    if loss > mj:
        loss = mj
    s.bite_frac[j] = frac
    dE = wp.max(c.energy[j], 0.0) * loss / wp.max(mj, 1.0e-12)
    c.mass[j] = mj - loss
    c.energy[j] = c.energy[j] - dE
    c.damage[j] = loss
    s.eb[j, EB_VICTIM] = s.eb[j, EB_VICTIM] + dE
    c.action_flags[j] = c.action_flags[j] | ACT_BITTEN
    if p.d_ledger_on != 0:
        wp.atomic_add(ledger, L_VICTIM_M, wp.float64(loss))


@wp.kernel
def pred_gain_kernel(c: Cells, s: Scratch, ledger: wp.array(dtype=wp.float64), p: SimParams):
    i = wp.tid()
    if c.alive[i] == 0:
        return
    got = float(0.0)
    for k in range(s.n_contacts[i]):
        b = s.bite_amt[i, k]
        if b > 0.0:
            got = got + b * s.bite_frac[s.contacts[i, k]]
    if got > 0.0:
        e = p.bite_eff * got
        c.energy[i] = c.energy[i] + e
        s.eb[i, EB_PRED] = e
        dm = p.bite_mass_gain * got
        c.mass[i] = c.mass[i] + dm
        if p.d_ledger_on != 0:
            wp.atomic_add(ledger, L_PRED_M, wp.float64(dm))


# ----------------------------------------------------------------------------- share
@wp.kernel
def share_out_kernel(c: Cells, s: Scratch, p: SimParams):
    i = wp.tid()
    s.share_out[i] = 0.0
    if c.alive[i] == 0:
        return
    frac = wp.clamp(s.a_share[i], 0.0, 1.0)
    if (c.action_flags[i] & ACT_DIE) != 0 and p.apoptosis_share != 0:
        frac = 1.0
    if frac <= 0.0:
        return
    nb = bond_count(c, i, p.max_bonds)
    if nb == 0:
        if s.a_share[i] > 0.0:
            c.fail_flags[i] = c.fail_flags[i] | FAIL_SHARE_NOBOND
        return
    out = frac * wp.max(c.energy[i], 0.0)
    c.energy[i] = c.energy[i] - out
    s.share_out[i] = out / float(nb)        # per-bond amount
    s.eb[i, EB_SHARE_OUT] = out


@wp.kernel
def share_in_kernel(c: Cells, bd: Bonds, s: Scratch, p: SimParams):
    i = wp.tid()
    if c.alive[i] == 0:
        return
    got = float(0.0)
    for sl in range(p.max_bonds):
        b = c.bonds[i, sl]
        if b >= 0:
            got = got + s.share_out[other_end(bd, b, i)]
    if got > 0.0:
        e = got * p.share_eff
        c.energy[i] = c.energy[i] + e
        s.eb[i, EB_SHARE_IN] = e


# ----------------------------------------------------------------------------- emit / signal / leak
@wp.kernel
def emit_signal_kernel(c: Cells, f: Fields, s: Scratch, ledger: wp.array(dtype=wp.float64), p: SimParams):
    i = wp.tid()
    if c.alive[i] == 0:
        return
    fg = grid_coord(c.pos[i], p.field_cell, p.d_field_nx, p.d_field_ny)
    ch = s.a_emit_ch[i]
    if ch >= 0:
        amt = wp.max(s.a_emit_amt[i], 0.0)
        cost = p.emit_cost * amt
        avail = wp.max(c.energy[i], 0.0)
        if cost > avail:
            amt = amt * avail / cost
            cost = avail
            c.fail_flags[i] = c.fail_flags[i] | FAIL_EMIT_ENERGY
        if amt > 0.0:
            c.energy[i] = c.energy[i] - cost
            s.eb[i, EB_EMIT] = cost
            wp.atomic_add(f.d_chem, ch, fg[0], fg[1], fx(amt))
            if p.d_ledger_on != 0:
                wp.atomic_add(ledger, L_EMIT_CHEM, wp.float64(amt))
    if p.passive_leak > 0.0:
        leak = p.passive_leak * s.eb[i, EB_BASAL]
        if leak > 0.0:
            wp.atomic_add(f.d_chem, 0, fg[0], fg[1], fx(leak))
            if p.d_ledger_on != 0:
                wp.atomic_add(ledger, L_LEAK_CHEM, wp.float64(leak))
    nsig = s.a_signal_count[i]
    if nsig > 0:
        cost = float(nsig) * p.signal_cost
        if cost > wp.max(c.energy[i], 0.0):
            for q in range(p.k_sig):
                c.signal_out[i, q] = c.signal_prev[i, q]
            c.fail_flags[i] = c.fail_flags[i] | FAIL_SIGNAL_ENERGY
        else:
            c.energy[i] = c.energy[i] - cost
            s.eb[i, EB_SIGNAL] = cost


# ----------------------------------------------------------------------------- bonds
@wp.kernel
def unbond_kernel(c: Cells, bd: Bonds, s: Scratch, p: SimParams):
    i = wp.tid()
    if c.alive[i] == 0:
        return
    u = s.a_unbond[i]
    if u == -2:
        for sl in range(p.max_bonds):
            b = c.bonds[i, sl]
            if b >= 0:
                bd.kill[b] = 1
    elif u >= 0:
        b = c.bonds[i, u]
        if b >= 0:
            bd.kill[b] = 1
        else:
            c.fail_flags[i] = c.fail_flags[i] | FAIL_UNBOND_EMPTY


@wp.kernel
def bond_target_kernel(c: Cells, bd: Bonds, s: Scratch, p: SimParams):
    i = wp.tid()
    s.bond_target[i] = -1
    s.bond_lock[i] = BIG
    s.flag[i] = 0
    s.flag2[i] = 0
    if c.alive[i] == 0:
        return
    sec = s.a_bond_sector[i]
    if sec < 0:
        return
    j = s.sector_best[i, sec]
    if j < 0:
        c.fail_flags[i] = c.fail_flags[i] | FAIL_BOND_NOCONTACT
        return
    if bonded_to(c, bd, i, j, p.max_bonds) != 0:
        c.fail_flags[i] = c.fail_flags[i] | FAIL_BOND_ALREADY
        return
    s.bond_target[i] = j


@wp.kernel
def bond_lock_kernel(c: Cells, s: Scratch, p: SimParams):
    i = wp.tid()
    j = s.bond_target[i]
    if j < 0:
        return
    if c.alive[j] == 0:
        return
    if free_slot(c, i, p.max_bonds) < 0 or free_slot(c, j, p.max_bonds) < 0:
        c.fail_flags[i] = c.fail_flags[i] | FAIL_BOND_NOSLOT
        return
    if c.energy[i] < p.bond_create_cost:
        c.fail_flags[i] = c.fail_flags[i] | FAIL_BOND_ENERGY
        return
    ok = int(1)
    if p.bond_consent_mode == 1:
        if s.bond_target[j] != i:
            ok = int(0)
    elif p.bond_consent_mode == 2:
        pr = wp.clamp(c.traits[i, T_ADHESION] - c.traits[j, T_ARMOR], 0.0, 1.0)
        if rng.rng_uniform(p.d_key, c.uid[i], P_BOND, 0) >= pr:
            ok = int(0)
    if ok == 0:
        c.fail_flags[i] = c.fail_flags[i] | FAIL_BOND_CONSENT
        return
    s.flag2[i] = 1
    wp.atomic_min(s.bond_lock, i, i)
    wp.atomic_min(s.bond_lock, j, i)


@wp.kernel
def bond_accept_kernel(c: Cells, s: Scratch, p: SimParams):
    i = wp.tid()
    f = int(0)
    if s.flag2[i] != 0:
        j = s.bond_target[i]
        if s.bond_lock[i] == i and s.bond_lock[j] == i:
            f = int(1)
        elif not (p.bond_consent_mode == 1 and s.bond_lock[i] == j):
            c.fail_flags[i] = c.fail_flags[i] | FAIL_BOND_CONFLICT
    s.flag[i] = f


@wp.kernel
def bond_create_kernel(c: Cells, bd: Bonds, s: Scratch, counters: wp.array(dtype=wp.int64), p: SimParams):
    i = wp.tid()
    if s.flag[i] == 0:
        return
    j = s.bond_target[i]
    b = pop_slot(bd.free_stack, counters, C_BOND_FREE, s.ticket[i])
    if b < 0:
        return
    sa = free_slot(c, i, p.max_bonds)
    sb = free_slot(c, j, p.max_bonds)
    d = min_image(c.pos[j] - c.pos[i], p.d_world)
    bd.a[b] = i
    bd.b[b] = j
    bd.slot_a[b] = sa
    bd.slot_b[b] = sb
    bd.rest_len[b] = body_radius(c, i, p) + body_radius(c, j, p)
    bd.rest_ang_a[b] = wrap_angle(direction_angle(d) - c.angle[i])
    bd.rest_ang_b[b] = wrap_angle(direction_angle(-d) - c.angle[j])
    bd.strength[b] = 1.0 + 0.5 * (c.traits[i, T_ADHESION] + c.traits[j, T_ADHESION])
    bd.strain[b] = 0.0
    bd.alive[b] = 1
    bd.kill[b] = 0
    c.bonds[i, sa] = b
    c.bonds[j, sb] = b
    c.energy[i] = c.energy[i] - p.bond_create_cost
    s.eb[i, EB_BOND] = s.eb[i, EB_BOND] + p.bond_create_cost
    c.action_flags[i] = c.action_flags[i] | ACT_BOND_OK


def compute(state, p):
    dev = state.device
    N = state.dims.N
    c, bd, s, f = state.cells, state.bonds, state.scratch, state.fields
    led = state.ledger
    wp.launch(pred_demand_kernel, N, [c, bd, s, p], device=dev)
    wp.launch(pred_resolve_kernel, N, [c, s, led, p], device=dev)
    wp.launch(pred_gain_kernel, N, [c, s, led, p], device=dev)
    wp.launch(share_out_kernel, N, [c, s, p], device=dev)
    wp.launch(share_in_kernel, N, [c, bd, s, p], device=dev)
    wp.launch(emit_signal_kernel, N, [c, f, s, led, p], device=dev)
    wp.launch(unbond_kernel, N, [c, bd, s, p], device=dev)
    wp.launch(bond_target_kernel, N, [c, bd, s, p], device=dev)
    wp.launch(bond_lock_kernel, N, [c, s, p], device=dev)
    wp.launch(bond_accept_kernel, N, [c, s, p], device=dev)
    common.scan(s.flag, s.ticket, state.counters, C_BOND_NEW_TOTAL)
    wp.launch(bond_create_kernel, N, [c, bd, s, state.counters, p], device=dev)
    wp.launch(common.advance_counter, 1, [state.counters, C_BOND_FREE, C_BOND_NEW_TOTAL, -1], device=dev)
