"""GPU genome table with copy-on-write.

Cells reference genome_id. A split without mutation shares the parent's entry (refcount += 1);
a split with an actual change gets a new entry. Entries with refcount 0 are returned to the free
list each tick (deterministic order via scan).

Label table: lab_off[g, k] .. lab_off[g, k+1] index into lab_pos[g, :], which holds positions of
LABEL k in ascending order. A jump scans only the occurrences of its own label (typically 0-3),
so it is O(occurrences of that label), effectively O(1).
"""
import numpy as np
import warp as wp

from ..kernels import common
from ..state import C_GENOME_FREE, C_GFREE_TOTAL, Genomes
from . import isa

NUM_LABELS = isa.NUM_LABELS
NUM_OPCODES = isa.NUM_OPCODES
OP_LABEL = isa.OP_LABEL
U24 = wp.constant(wp.uint32(24))
U16 = wp.constant(wp.uint32(16))
U8 = wp.constant(wp.uint32(8))
UFF = wp.constant(wp.uint32(0xFF))


@wp.func
def w_op(w: int) -> int:
    return int((wp.uint32(w) >> U24) & UFF) % NUM_OPCODES


@wp.func
def w_a(w: int) -> int:
    return int((wp.uint32(w) >> U16) & UFF)


@wp.func
def w_b(w: int) -> int:
    return int((wp.uint32(w) >> U8) & UFF)


@wp.func
def w_c(w: int) -> int:
    return int(wp.uint32(w) & UFF)


@wp.func
def find_label(g: Genomes, gid: int, k: int, ip: int, length: int) -> int:
    s = g.lab_off[gid, k]
    e = g.lab_off[gid, k + 1]
    if s == e:
        return -1
    target = int(g.lab_pos[gid, s])
    for j in range(s, e):
        p = int(g.lab_pos[gid, j])
        if p > ip:
            target = p
            break
    t = target + 1
    if t >= length:
        t = 0
    return t


@wp.func
def first_label(g: Genomes, gid: int, k: int, length: int) -> int:
    s = g.lab_off[gid, k]
    e = g.lab_off[gid, k + 1]
    if s == e:
        return -1
    t = int(g.lab_pos[gid, s]) + 1
    if t >= length:
        t = 0
    return t


@wp.func
def finalize_genome(g: Genomes, gid: int):
    """Rebuild label table and hash of genome gid."""
    n = g.length[gid]
    cnt = int(0)
    for k in range(NUM_LABELS):
        g.lab_off[gid, k] = cnt
        for p in range(n):
            w = g.words[gid, p]
            if w_op(w) == OP_LABEL:
                if w_a(w) % NUM_LABELS == k:
                    g.lab_pos[gid, cnt] = wp.int16(p)
                    cnt = cnt + 1
    g.lab_off[gid, NUM_LABELS] = cnt
    # FNV-1a 64 over (length, words), little-endian bytes
    h = wp.uint64(0xCBF29CE484222325)
    prime = wp.uint64(0x100000001B3)
    for p in range(n + 1):
        uw = wp.uint32(n)
        if p > 0:
            uw = wp.uint32(g.words[gid, p - 1])
        for k in range(4):
            byte = wp.uint64((uw >> wp.uint32(8 * k)) & UFF)
            h = (h ^ byte) * prime
    g.hash[gid] = wp.int64(h)


@wp.kernel
def finalize_list_kernel(g: Genomes, gids: wp.array(dtype=wp.int32)):
    i = wp.tid()
    gid = gids[i]
    if gid >= 0:
        finalize_genome(g, gid)


@wp.kernel
def gc_mark_kernel(g: Genomes, flags: wp.array(dtype=wp.int32)):
    i = wp.tid()
    f = 0
    if g.in_use[i] != 0 and g.refcount[i] <= 0:
        g.in_use[i] = 0
        g.refcount[i] = 0
        f = 1
    flags[i] = f


def gc(state):
    """Free genome entries whose refcount dropped to zero."""
    s, g = state.scratch, state.genomes
    wp.launch(gc_mark_kernel, state.dims.G, [g, s.gflag], device=state.device)
    common.push_flagged(s.gflag, s.gticket, g.free_stack, state.counters, C_GENOME_FREE, C_GFREE_TOTAL)


@wp.kernel
def upload_kernel(g: Genomes, slots: wp.array(dtype=wp.int32), words: wp.array2d(dtype=wp.int32),
                  lengths: wp.array(dtype=wp.int32), tick: wp.int64):
    i = wp.tid()
    gid = slots[i]
    n = lengths[i]
    for p in range(n):
        g.words[gid, p] = words[i, p]
    g.length[gid] = n
    g.in_use[gid] = 1
    g.refcount[gid] = 0
    g.parent[gid] = -1
    g.first_tick[gid] = tick
    finalize_genome(g, gid)


def host_label_table(words):
    """numpy version, for tests."""
    from .reference_vm import build_label_table
    return build_label_table(words)


def words_to_i32(words) -> np.ndarray:
    return np.array([isa.to_signed32(int(w)) for w in words], dtype=np.int32)
