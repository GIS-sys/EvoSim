"""Host-driven population edits: seeding, inject (seed / random / archive / clipboard genomes),
clone (cell or whole organism, optionally mutated), cull. Used by initialisation, policies and UI
brushes. These run between ticks, are rare, and therefore go through the host; slots are taken
from the same free lists (in the same pop order) as births, so the run stays deterministic.

Every intervention is booked to the ledger's intervention terms by the caller (Sim).
"""
from __future__ import annotations

import numpy as np
import warp as wp

from . import rng
from .genome import isa, mutate, table
from .state import (C_BOND_FREE, C_CELL_FREE, C_GENOME_FREE, C_LINEAGE, C_SPLIT_TOTAL, C_UID, Cells, Genomes,
                    Bonds)

NT = isa.NUM_TRAITS


@wp.kernel
def spawn_kernel(c: Cells, g: Genomes, slots: wp.array(dtype=wp.int32), gids: wp.array(dtype=wp.int32),
                 pos: wp.array(dtype=wp.vec2), vel: wp.array(dtype=wp.vec2), ang: wp.array(dtype=float),
                 mass: wp.array(dtype=float), energy: wp.array(dtype=float), traits: wp.array2d(dtype=float),
                 targets: wp.array2d(dtype=float), regs: wp.array2d(dtype=float), state: wp.array(dtype=wp.int32),
                 ip: wp.array(dtype=wp.int32), uid: wp.array(dtype=wp.int64), puid: wp.array(dtype=wp.int64),
                 lineage: wp.array(dtype=wp.int32), depth: wp.array(dtype=wp.int32), origin: wp.array(dtype=wp.int32),
                 tick: wp.int64, k_sig: int, mb: int):
    e = wp.tid()
    i = slots[e]
    gid = gids[e]
    c.pos[i] = pos[e]
    c.vel[i] = vel[e]
    c.angle[i] = ang[e]
    c.ang_vel[i] = 0.0
    c.mass[i] = mass[e]
    c.energy[i] = energy[e]
    c.age[i] = 0
    c.genome_id[i] = gid
    c.ip[i] = ip[e]
    c.wait[i] = 0
    for r in range(8):
        c.regs[i, r] = regs[e, r]
    c.state[i] = state[e]
    for t in range(NT):
        c.traits[i, t] = traits[e, t]
        c.trait_targets[i, t] = targets[e, t]
    for q in range(k_sig):
        c.signal_out[i, q] = 0.0
        c.signal_prev[i, q] = 0.0
    for sl in range(mb):
        c.bonds[i, sl] = -1
    c.uid[i] = uid[e]
    c.parent_uid[i] = puid[e]
    c.division_depth[i] = depth[e]
    c.lineage_id[i] = lineage[e]
    c.organism_id[i] = i
    c.birth_tick[i] = tick
    c.action_flags[i] = 0
    c.fail_flags[i] = 0
    c.origin[i] = origin[e]
    c.light_prev[i] = 0.0
    c.damage[i] = 0.0
    c.traced[i] = -1
    c.alive[i] = 1
    wp.atomic_add(g.refcount, gid, 1)


@wp.kernel
def bond_insert_kernel(c: Cells, bd: Bonds, bslots: wp.array(dtype=wp.int32), a: wp.array(dtype=wp.int32),
                       b: wp.array(dtype=wp.int32), sa: wp.array(dtype=wp.int32), sb: wp.array(dtype=wp.int32),
                       rest: wp.array(dtype=float), ra: wp.array(dtype=float), rb: wp.array(dtype=float),
                       strength: wp.array(dtype=float)):
    e = wp.tid()
    k = bslots[e]
    bd.a[k] = a[e]
    bd.b[k] = b[e]
    bd.slot_a[k] = sa[e]
    bd.slot_b[k] = sb[e]
    bd.rest_len[k] = rest[e]
    bd.rest_ang_a[k] = ra[e]
    bd.rest_ang_b[k] = rb[e]
    bd.strength[k] = strength[e]
    bd.strain[k] = 0.0
    bd.alive[k] = 1
    bd.kill[k] = 0
    c.bonds[a[e], sa[e]] = k
    c.bonds[b[e], sb[e]] = k


@wp.kernel
def mark_kernel(arr: wp.array(dtype=wp.int32), idx: wp.array(dtype=wp.int32), value: int):
    e = wp.tid()
    arr[idx[e]] = value


def _pop_host(stack: wp.array, count: int, k: int) -> np.ndarray:
    """Slots for tickets 0..k-1, i.e. stack[count-1-t]; matches pop_slot on the GPU."""
    if k <= 0:
        return np.zeros(0, np.int32)
    seg = stack[count - k:count].numpy()
    return seg[::-1].copy()


def upload_genomes(sim, genome_list) -> np.ndarray:
    """Upload distinct word lists into free genome slots. Returns their genome ids."""
    st = sim.state
    cnt = st.counters.numpy()
    n = len(genome_list)
    if n == 0:
        return np.zeros(0, np.int32)
    if n > cnt[C_GENOME_FREE]:
        raise RuntimeError("genome table full")
    gslots = _pop_host(st.genomes.free_stack, int(cnt[C_GENOME_FREE]), n)
    L = isa.MAX_GENOME_LEN
    W = np.zeros((n, L), np.int32)
    lens = np.zeros(n, np.int32)
    for i, w in enumerate(genome_list):
        w = list(w)[:L]
        if not w:
            w = [isa.encode(isa.OP_NOP)]
        W[i, :len(w)] = table.words_to_i32(w)
        lens[i] = len(w)
    dev = st.device
    wp.launch(table.upload_kernel, n, [st.genomes, wp.array(gslots, device=dev), wp.array(W, device=dev),
                                       wp.array(lens, device=dev), wp.int64(sim.tick)], device=dev)
    cnt[C_GENOME_FREE] -= n
    st.counters.assign(cnt)
    return gslots


def spawn_cells(sim, entries: dict) -> np.ndarray:
    """entries: pos (k,2), angle (k), mass, energy, gid (k, existing genome ids), optional: vel, traits,
    targets, regs, state, ip, lineage (-1 = new lineage), parent_uid, depth, origin.
    Returns the slots used (may be fewer than k if capacity is short)."""
    st = sim.state
    d = st.dims
    k = len(entries["pos"])
    cnt = st.counters.numpy()
    k = min(k, int(cnt[C_CELL_FREE]))
    if k <= 0:
        return np.zeros(0, np.int32)
    slots = _pop_host(st.cell_free_stack, int(cnt[C_CELL_FREE]), k)

    def get(name, default, shape=None, dtype=np.float32):
        v = entries.get(name)
        if v is None:
            return np.full((k,) + (shape or ()), default, dtype=dtype)
        return np.asarray(v, dtype=dtype)[:k]

    lineage = get("lineage", -1, dtype=np.int32).copy()
    newl = lineage < 0
    nl = int(newl.sum())
    lineage[newl] = int(cnt[C_LINEAGE]) + np.arange(nl, dtype=np.int32)
    uids = int(cnt[C_UID]) + np.arange(k, dtype=np.int64)
    gids = get("gid", 0, dtype=np.int32)
    dev = st.device
    arrs = [
        wp.array(slots.astype(np.int32), device=dev),
        wp.array(gids, device=dev),
        wp.array(get("pos", 0, (2,)).reshape(k, 2), dtype=wp.vec2, device=dev),
        wp.array(get("vel", 0, (2,)).reshape(k, 2), dtype=wp.vec2, device=dev),
        wp.array(get("angle", 0.0), device=dev),
        wp.array(get("mass", sim.cfg.initial_mass), device=dev),
        wp.array(get("energy", sim.cfg.initial_energy), device=dev),
        wp.array(get("traits", 0.0, (NT,)).reshape(k, NT), device=dev),
        wp.array(get("targets", 0.0, (NT,)).reshape(k, NT), device=dev),
        wp.array(get("regs", 0.0, (8,)).reshape(k, 8), device=dev),
        wp.array(get("state", 0, dtype=np.int32), device=dev),
        wp.array(get("ip", 0, dtype=np.int32), device=dev),
        wp.array(uids, device=dev),
        wp.array(get("parent_uid", 0, dtype=np.int64), device=dev),
        wp.array(lineage, device=dev),
        wp.array(get("depth", 0, dtype=np.int32), device=dev),
        wp.array(get("origin", isa.ORIGIN_INJECTED, dtype=np.int32), device=dev),
    ]
    wp.launch(spawn_kernel, k, [st.cells, st.genomes] + arrs + [wp.int64(sim.tick), d.KS, d.MB], device=dev)
    cnt = st.counters.numpy()
    cnt[C_CELL_FREE] -= k
    cnt[C_UID] += k
    cnt[C_LINEAGE] += nl
    st.counters.assign(cnt)
    # ledger: interventions
    sim.book_intervention(mass=float(get("mass", sim.cfg.initial_mass).sum()),
                          energy=float(get("energy", sim.cfg.initial_energy).sum()))
    return slots


def insert_bonds(sim, bonds: list[tuple]):
    """bonds: (a_slot, b_slot, slot_a, slot_b, rest_len, rest_ang_a, rest_ang_b, strength)."""
    st = sim.state
    n = len(bonds)
    if n == 0:
        return
    cnt = st.counters.numpy()
    n = min(n, int(cnt[C_BOND_FREE]))
    bs = _pop_host(st.bonds.free_stack, int(cnt[C_BOND_FREE]), n)
    cols = list(zip(*bonds[:n]))
    dev = st.device
    arrs = [wp.array(bs, device=dev)] + [wp.array(np.asarray(cols[q], np.int32), device=dev) for q in range(4)] + \
           [wp.array(np.asarray(cols[q], np.float32), device=dev) for q in range(4, 8)]
    wp.launch(bond_insert_kernel, n, [st.cells, st.bonds] + arrs, device=dev)
    cnt[C_BOND_FREE] -= n
    st.counters.assign(cnt)


def mutate_genomes(sim, src_gids, mult=1.0) -> np.ndarray:
    """Copy+mutate genomes on the GPU (same kernel as births). Returns new gids; identical copies
    are released and the source id is returned instead (COW)."""
    st = sim.state
    n = len(src_gids)
    if n == 0:
        return np.zeros(0, np.int32)
    cnt = st.counters.numpy()
    gslots = _pop_host(st.genomes.free_stack, int(cnt[C_GENOME_FREE]), n)
    j = st.jobs
    dev = st.device
    src = np.zeros(st.dims.N, np.int32)
    dst = np.zeros(st.dims.N, np.int32)
    src[:n] = src_gids
    dst[:n] = gslots
    j.src.assign(src)
    j.dst.assign(dst)
    j.mult.fill_(float(mult))
    base = 10**12 + sim.tick * 1000003            # distinct RNG stream from births
    j.uid.assign(np.arange(st.dims.N, dtype=np.int64) + base)
    nj = wp.array([n], dtype=wp.int64, device=dev)
    wp.launch(mutate.mutate_kernel, n, [st.genomes, j, nj, sim.params()], device=dev)
    same = j.same.numpy()[:n]
    out = np.where(same != 0, np.asarray(src_gids, np.int32), gslots).astype(np.int32)
    used = gslots[same == 0]
    if len(used):
        g = st.genomes
        inuse = g.in_use.numpy()
        inuse[used] = 1
        g.in_use.assign(inuse)
    # released slots stay on the stack: only pop the used ones. Reorder the stack segment so that
    # used slots are consumed and unused remain available.
    keep = gslots[same != 0]
    seg_n = n
    top = int(cnt[C_GENOME_FREE])
    stack = st.genomes.free_stack.numpy()
    new_seg = list(keep[::-1])          # unused go back on top region
    stack[top - seg_n: top - seg_n + len(new_seg)] = new_seg
    st.genomes.free_stack.assign(stack)
    cnt[C_GENOME_FREE] = top - seg_n + len(new_seg)
    st.counters.assign(cnt)
    return out


def cull(sim, slots):
    """Mark cells for removal; processed in the next tick's death phase (booked as intervention)."""
    slots = np.asarray(slots, np.int32)
    if len(slots) == 0:
        return
    st = sim.state
    wp.launch(mark_kernel, len(slots), [st.scratch.dying, wp.array(slots, device=st.device), 2], device=st.device)


def place_positions(sim, n, rs, mode="uniform", center=None, radius=None, region=None, tries=8, min_gap=None,
                    weights_field=None):
    """Find up to n positions avoiding existing cells. Returns (positions, skipped)."""
    W, H = sim.state.dims.world
    alive = sim.state.cells.alive.numpy() != 0
    P = sim.state.cells.pos.numpy()[alive]
    gap = min_gap if min_gap is not None else 2.0 * np.sqrt(sim.cfg.initial_mass / (np.pi * sim.cfg.density)) + 1.0
    cell = max(gap, 4.0)
    nx, ny = max(1, int(W // cell)), max(1, int(H // cell))
    grid: dict = {}
    for q in P:
        grid.setdefault((int(q[0] / W * nx) % nx, int(q[1] / H * ny) % ny), []).append(q)
    out = []
    skipped = 0
    wcdf = None
    if weights_field is not None:
        w = np.maximum(weights_field.astype(np.float64).ravel(), 0)
        if w.sum() > 0:
            wcdf = np.cumsum(w) / w.sum()
    for _ in range(n):
        ok = False
        for _t in range(tries):
            if mode == "near" and center is not None:
                a = rs.random() * 2 * np.pi
                r = (radius or 20.0) * np.sqrt(rs.random())
                x, y = center[0] + r * np.cos(a), center[1] + r * np.sin(a)
            elif mode == "region" and region is not None:
                x, y = region.sample(rs)
            elif mode == "weighted" and wcdf is not None:
                k = int(np.searchsorted(wcdf, rs.random()))
                fnx, fny = weights_field.shape
                ix, iy = divmod(k, fny)
                x, y = (ix + rs.random()) * W / fnx, (iy + rs.random()) * H / fny
            else:
                x, y = rs.random() * W, rs.random() * H
            x %= W
            y %= H
            gx, gy = int(x / W * nx) % nx, int(y / H * ny) % ny
            clash = False
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for q in grid.get(((gx + dx) % nx, (gy + dy) % ny), ()):
                        ddx = (q[0] - x + W / 2) % W - W / 2
                        ddy = (q[1] - y + H / 2) % H - H / 2
                        if ddx * ddx + ddy * ddy < gap * gap:
                            clash = True
                            break
                    if clash:
                        break
                if clash:
                    break
            if not clash:
                out.append((x, y))
                grid.setdefault((gx, gy), []).append(np.array([x, y]))
                ok = True
                break
        if not ok:
            skipped += 1
    return np.array(out, np.float32).reshape(-1, 2), skipped
