"""Births (SPLIT / SPLITB) and deaths.

Births: request -> capacity grant (prefix scan, so the first `free` requests in index order win)
-> slot pop -> daughter init -> genome copy+mutate job -> COW resolve (identical copies are released
and the daughter shares the parent's genome) -> daughter ip resolved from its own label table.

Deaths: mark -> detritus deposit (fixed-point, deterministic) -> bond cleanup (+ strain breaking)
-> slots pushed back to free lists (prefix scan) -> genome refcounts / GC.

A split never increases total energy or mass: E' = E - split_cost is divided by the fraction.
"""
import warp as wp

from ..config import SimParams
from ..dims import heading, radius_of, wrap_angle, wrap_pos, grid_coord
from ..genome import isa, mutate, table
from ..genome.table import first_label
from ..state import (C_BIRTHS, C_BOND_DEAD_TOTAL, C_BOND_FREE, C_BONDS_BROKEN, C_CELL_FREE, C_DEATHS, C_DYING_TOTAL,
                     C_GENOME_FREE, C_GFREE_TOTAL, C_RELEASE_TOTAL, C_SPLIT_FAIL_CAP, C_SPLIT_TOTAL, C_SPLITB_TOTAL,
                     C_UID, EB_SPLIT, L_DEATH_E, L_DEATH_M, L_DETRITUS, L_INTERV_E, L_INTERV_M, Bonds, Cells, Fields,
                     Genomes, MutJobs, Scratch)
from . import common
from .common import fx, pop_slot
from .light import body_radius

NT = isa.NUM_TRAITS
T_ARMOR, T_ADHESION = isa.T_ARMOR, isa.T_ADHESION
ACT_SPLIT, ACT_SPLITB, ACT_SPLIT_OK, ACT_BORN, ACT_DIE = (isa.ACT_SPLIT, isa.ACT_SPLITB, isa.ACT_SPLIT_OK,
                                                         isa.ACT_BORN, isa.ACT_DIE)
FAIL_SPLIT_MASS, FAIL_SPLIT_ENERGY, FAIL_CAPACITY = isa.FAIL_SPLIT_MASS, isa.FAIL_SPLIT_ENERGY, isa.FAIL_CAPACITY
FAIL_SPLIT_NOSLOT, FAIL_SPLIT_NOLABEL = isa.FAIL_SPLIT_NOSLOT, isa.FAIL_SPLIT_NOLABEL
ORIGIN_NATURAL = isa.ORIGIN_NATURAL


@wp.func
def _free_slot(c: Cells, i: int, mb: int) -> int:
    for sl in range(mb):
        if c.bonds[i, sl] < 0:
            return sl
    return -1


# ============================================================================= births
@wp.kernel
def split_request_kernel(c: Cells, s: Scratch, p: SimParams):
    i = wp.tid()
    s.flag[i] = 0
    s.daughter[i] = -1
    if c.alive[i] == 0:
        return
    fl = c.action_flags[i]
    if (fl & (ACT_SPLIT | ACT_SPLITB)) == 0 or s.a_split_label[i] < 0:
        return
    if c.mass[i] < p.min_split_mass:
        c.fail_flags[i] = c.fail_flags[i] | FAIL_SPLIT_MASS
        return
    if c.energy[i] < p.split_cost:
        c.fail_flags[i] = c.fail_flags[i] | FAIL_SPLIT_ENERGY
        return
    if (fl & ACT_SPLITB) != 0 and _free_slot(c, i, p.max_bonds) < 0:
        c.fail_flags[i] = c.fail_flags[i] | FAIL_SPLIT_NOSLOT
        return
    s.flag[i] = 1


@wp.kernel
def split_grant_kernel(c: Cells, s: Scratch, counters: wp.array(dtype=wp.int64)):
    i = wp.tid()
    s.flag2[i] = 0
    if s.flag[i] == 0:
        return
    cap = wp.min(counters[C_CELL_FREE], counters[C_GENOME_FREE])
    if wp.int64(s.ticket[i]) >= cap:
        s.flag[i] = 0
        c.fail_flags[i] = c.fail_flags[i] | FAIL_CAPACITY
        wp.atomic_add(counters, C_SPLIT_FAIL_CAP, wp.int64(1))
        return
    if (c.action_flags[i] & ACT_SPLITB) != 0:
        s.flag2[i] = 1


@wp.kernel
def split_apply_kernel(c: Cells, g: Genomes, bd: Bonds, s: Scratch, jobs: MutJobs, cell_stack: wp.array(dtype=wp.int32),
                        counters: wp.array(dtype=wp.int64), p: SimParams):
    i = wp.tid()
    if s.flag[i] == 0:
        return
    t = s.ticket[i]
    gslot = pop_slot(g.free_stack, counters, C_GENOME_FREE, t)
    dc = pop_slot(cell_stack, counters, C_CELL_FREE, t)
    bonded = (c.action_flags[i] & ACT_SPLITB) != 0
    # ---- divide energy and mass (never increases totals)
    m = c.mass[i]
    E = c.energy[i] - p.split_cost
    s.eb[i, EB_SPLIT] = s.eb[i, EB_SPLIT] + p.split_cost
    f = wp.clamp(s.a_split_frac[i], p.min_frac, 1.0 - p.min_frac)
    md = f * m
    Ed = f * E
    mp = m - md
    Ep = E - Ed
    c.mass[i] = mp
    c.energy[i] = Ep
    rho = p.density * (1.0 + p.armor_density * c.traits[i, T_ARMOR])
    rp = radius_of(mp, rho)
    rd = radius_of(md, rho)
    sa = wrap_angle(s.a_split_angle[i])
    ang = wrap_angle(c.angle[i] + sa)
    dirv = heading(ang)
    # ---- daughter
    c.pos[dc] = wrap_pos(c.pos[i] + dirv * ((rp + rd) * 0.5), p.d_world)
    c.vel[dc] = c.vel[i]
    c.angle[dc] = ang
    c.ang_vel[dc] = c.ang_vel[i]
    c.mass[dc] = md
    c.energy[dc] = Ed
    c.age[dc] = 0
    c.genome_id[dc] = gslot
    c.ip[dc] = 0
    c.wait[dc] = 0
    for r in range(8):
        if p.reg_inherit == 0:
            c.regs[dc, r] = c.regs[i, r]
        else:
            c.regs[dc, r] = 0.0
    c.state[dc] = c.state[i]
    for tt in range(NT):
        c.traits[dc, tt] = c.traits[i, tt]
        c.trait_targets[dc, tt] = c.trait_targets[i, tt]
    for q in range(p.k_sig):
        c.signal_out[dc, q] = 0.0
        c.signal_prev[dc, q] = 0.0
    for sl in range(p.max_bonds):
        c.bonds[dc, sl] = -1
    uid = counters[C_UID] + wp.int64(t)
    c.uid[dc] = uid
    c.parent_uid[dc] = c.uid[i]
    c.division_depth[dc] = c.division_depth[i] + 1
    c.lineage_id[dc] = c.lineage_id[i]
    c.organism_id[dc] = c.organism_id[i]
    c.birth_tick[dc] = p.d_tick
    c.action_flags[dc] = ACT_BORN
    c.fail_flags[dc] = 0
    c.origin[dc] = ORIGIN_NATURAL
    c.light_prev[dc] = c.light_prev[i]
    c.damage[dc] = 0.0
    c.traced[dc] = -1
    c.alive[dc] = 1
    s.daughter[i] = dc
    c.action_flags[i] = c.action_flags[i] | ACT_SPLIT_OK
    # ---- genome job
    jobs.src[t] = c.genome_id[i]
    jobs.dst[t] = gslot
    jobs.uid[t] = uid
    mult = float(1.0)
    if bonded:
        mult = p.somatic_rate_multiplier
    jobs.mult[t] = mult
    # ---- bond parent<->daughter
    if bonded:
        b = pop_slot(bd.free_stack, counters, C_BOND_FREE, s.ticket2[i])
        if b >= 0:
            slp = _free_slot(c, i, p.max_bonds)
            bd.a[b] = i
            bd.b[b] = dc
            bd.slot_a[b] = slp
            bd.slot_b[b] = 0
            bd.rest_len[b] = rp + rd
            bd.rest_ang_a[b] = sa
            bd.rest_ang_b[b] = wrap_angle(3.141592653589793)
            bd.strength[b] = 1.0 + c.traits[i, T_ADHESION]
            bd.strain[b] = 0.0
            bd.alive[b] = 1
            bd.kill[b] = 0
            c.bonds[i, slp] = b
            c.bonds[dc, 0] = b


@wp.kernel
def split_counters_kernel(counters: wp.array(dtype=wp.int64)):
    n = counters[C_SPLIT_TOTAL]
    counters[C_CELL_FREE] = counters[C_CELL_FREE] - n
    counters[C_GENOME_FREE] = counters[C_GENOME_FREE] - n
    counters[C_UID] = counters[C_UID] + n
    counters[C_BOND_FREE] = counters[C_BOND_FREE] - counters[C_SPLITB_TOTAL]
    counters[C_BIRTHS] = n


@wp.kernel
def cow_resolve_kernel(c: Cells, g: Genomes, s: Scratch, jobs: MutJobs, p: SimParams):
    i = wp.tid()
    if s.flag[i] == 0:
        return
    t = s.ticket[i]
    dc = s.daughter[i]
    src = jobs.src[t]
    dst = jobs.dst[t]
    gid = dst
    if jobs.same[t] != 0:
        gid = src
        wp.atomic_add(g.refcount, src, 1)
        s.gflag[dst] = 1                     # release the unused slot
    else:
        g.refcount[dst] = 1
        g.in_use[dst] = 1
    c.genome_id[dc] = gid
    st = first_label(g, gid, s.a_split_label[i], g.length[gid])
    if st < 0:
        st = 0
        c.fail_flags[i] = c.fail_flags[i] | FAIL_SPLIT_NOLABEL
    c.ip[dc] = st


def births(state, p):
    dev = state.device
    N = state.dims.N
    c, g, bd, s, j = state.cells, state.genomes, state.bonds, state.scratch, state.jobs
    cnt = state.counters
    wp.launch(split_request_kernel, N, [c, s, p], device=dev)
    common.scan(s.flag, s.ticket, cnt, C_SPLIT_TOTAL)
    wp.launch(split_grant_kernel, N, [c, s, cnt], device=dev)
    common.scan(s.flag, s.ticket, cnt, C_SPLIT_TOTAL)          # rescan granted (same order)
    common.scan(s.flag2, s.ticket2, cnt, C_SPLITB_TOTAL)
    wp.launch(split_apply_kernel, N, [c, g, bd, s, j, state.cell_free_stack, cnt, p], device=dev)
    wp.launch(split_counters_kernel, 1, [cnt], device=dev)
    wp.launch(mutate.mutate_kernel, N, [g, j, cnt[C_SPLIT_TOTAL:C_SPLIT_TOTAL + 1], p], device=dev)
    s.gflag.zero_()
    wp.launch(cow_resolve_kernel, N, [c, g, s, j, p], device=dev)
    common.push_flagged(s.gflag, s.gticket, g.free_stack, cnt, C_GENOME_FREE, C_RELEASE_TOTAL)


# ============================================================================= deaths
@wp.kernel
def death_mark_kernel(c: Cells, g: Genomes, bd: Bonds, f: Fields, s: Scratch, ledger: wp.array(dtype=wp.float64),
                      p: SimParams):
    i = wp.tid()
    cull = s.dying[i] == 2
    s.dying[i] = 0
    s.flag[i] = 0
    if c.alive[i] == 0:
        return
    dead = int(0)
    if cull:
        dead = int(1)
    if c.energy[i] <= 0.0 or c.mass[i] < p.min_mass:
        dead = int(1)
    if p.max_age_enabled != 0 and c.age[i] > p.max_age:
        dead = int(1)
    if (c.action_flags[i] & ACT_DIE) != 0:
        dead = int(1)
    if dead == 0:
        c.age[i] = c.age[i] + 1
        return
    s.dying[i] = 1
    s.flag[i] = 1
    m = wp.max(c.mass[i], 0.0)
    e = c.energy[i]
    if cull:
        if p.d_ledger_on != 0:
            wp.atomic_add(ledger, L_INTERV_M, wp.float64(-m))
            wp.atomic_add(ledger, L_INTERV_E, wp.float64(-e))
    else:
        det = p.detritus_fraction * m
        if det > 0.0:
            fg = grid_coord(c.pos[i], p.field_cell, p.d_field_nx, p.d_field_ny)
            wp.atomic_add(f.d_nutrient, fg[0], fg[1], fx(det))
        if p.d_ledger_on != 0:
            wp.atomic_add(ledger, L_DEATH_E, wp.float64(e))
            wp.atomic_add(ledger, L_DEATH_M, wp.float64(m))
            wp.atomic_add(ledger, L_DETRITUS, wp.float64(det))
    wp.atomic_sub(g.refcount, c.genome_id[i], 1)


@wp.kernel
def bond_cleanup_kernel(c: Cells, bd: Bonds, s: Scratch, counters: wp.array(dtype=wp.int64), p: SimParams):
    b = wp.tid()
    s.bflag[b] = 0
    if bd.alive[b] == 0:
        return
    a = bd.a[b]
    o = bd.b[b]
    remove = int(0)
    if bd.kill[b] != 0 or s.dying[a] != 0 or s.dying[o] != 0 or c.alive[a] == 0 or c.alive[o] == 0:
        remove = int(1)
    else:
        lim = p.break_strain * (1.0 + 0.5 * (c.traits[a, T_ADHESION] + c.traits[o, T_ADHESION]))
        if wp.abs(bd.strain[b]) > lim:
            remove = int(1)
            wp.atomic_add(counters, C_BONDS_BROKEN, wp.int64(1))
    if remove == 0:
        return
    if c.bonds[a, bd.slot_a[b]] == b:
        c.bonds[a, bd.slot_a[b]] = -1
    if c.bonds[o, bd.slot_b[b]] == b:
        c.bonds[o, bd.slot_b[b]] = -1
    bd.alive[b] = 0
    bd.kill[b] = 0
    s.bflag[b] = 1


@wp.kernel
def death_finalize_kernel(c: Cells, s: Scratch):
    i = wp.tid()
    if s.flag[i] != 0:
        c.alive[i] = 0
        c.traced[i] = -1
        c.energy[i] = 0.0
        c.mass[i] = 0.0


@wp.kernel
def deaths_counter_kernel(counters: wp.array(dtype=wp.int64)):
    counters[C_DEATHS] = counters[C_DYING_TOTAL]


def deaths(state, p):
    dev = state.device
    N = state.dims.N
    c, g, bd, s, f = state.cells, state.genomes, state.bonds, state.scratch, state.fields
    cnt = state.counters
    wp.launch(death_mark_kernel, N, [c, g, bd, f, s, state.ledger, p], device=dev)
    wp.launch(bond_cleanup_kernel, state.dims.B, [c, bd, s, cnt, p], device=dev)
    common.push_flagged(s.bflag, s.bticket, bd.free_stack, cnt, C_BOND_FREE, C_BOND_DEAD_TOTAL)
    wp.launch(death_finalize_kernel, N, [c, s], device=dev)
    common.push_flagged(s.flag, s.ticket, state.cell_free_stack, cnt, C_CELL_FREE, C_DYING_TOTAL)
    wp.launch(deaths_counter_kernel, 1, [cnt], device=dev)
    table.gc(state)
