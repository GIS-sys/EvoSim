"""Genome copy + mutation, one thread per job (birth or clone).

Each job copies src -> dst (a freshly popped genome slot), applies point and structural
mutations with rates x job.mult, and records whether the result is identical to the source.
Identical results are released afterwards and the daughter shares the parent's genome (COW).
All randomness: rng(key, job uid, P_MUTATE, counter).
"""
import warp as wp

from .. import rng
from ..config import SimParams
from ..state import Genomes, MutJobs
from . import isa
from .table import finalize_genome, w_a, w_b, w_c, w_op

MAX_LEN = isa.MAX_GENOME_LEN
OP_SET = isa.OP_SET
OP_NOP = isa.OP_NOP
P_MUTATE = rng.P_MUTATE


@wp.func
def _reverse(g: Genomes, gid: int, lo: int, hi: int):
    """Reverse words[lo:hi] in place."""
    i = lo
    j = hi - 1
    while i < j:
        t = g.words[gid, i]
        g.words[gid, i] = g.words[gid, j]
        g.words[gid, j] = t
        i = i + 1
        j = j - 1


@wp.func
def _geom_len(u: float, mean: float) -> int:
    # geometric on {1,2,...} with the given mean
    if mean <= 1.0:
        return 1
    q = 1.0 - 1.0 / mean
    return 1 + int(wp.floor(wp.log(wp.max(u, 1.0e-12)) / wp.log(q)))


@wp.func
def _set_byte(w: int, field: int, v: int) -> int:
    uw = wp.uint32(w)
    shift = wp.uint32(8 * (3 - field))
    mask = ~(wp.uint32(0xFF) << shift)
    uw = (uw & mask) | (wp.uint32(v & 0xFF) << shift)
    return int(uw)


@wp.func
def mutation_scale(g: Genomes, gid: int, p: SimParams) -> float:
    if p.evolvable_mutation_rate != 0 and g.length[gid] > 0:
        w = g.words[gid, 0]
        if w_op(w) == OP_NOP:
            e = (float(w_a(w)) - 128.0) / 32.0
            return wp.pow(2.0, wp.clamp(e, -4.0, 4.0))
    return 1.0


@wp.kernel
def mutate_kernel(g: Genomes, jobs: MutJobs, njobs: wp.array(dtype=wp.int64), p: SimParams):
    i = wp.tid()
    if i >= int(njobs[0]):
        return
    src = jobs.src[i]
    dst = jobs.dst[i]
    uid = jobs.uid[i]
    key = p.d_key
    n = g.length[src]
    for k in range(n):
        g.words[dst, k] = g.words[src, k]
    mult = jobs.mult[i] * mutation_scale(g, src, p)
    cnt = int(0)
    events = int(0)
    # ---------------------------------------------------------------- point mutations
    pm = p.mu_point * mult
    if pm > 0.0:
        for k in range(n):
            u = rng.rng_uniform(key, uid, P_MUTATE, cnt)
            cnt = cnt + 1
            if u < pm:
                events = events + 1
                w = g.words[dst, k]
                v = rng.rng_uniform(key, uid, P_MUTATE, cnt)
                cnt = cnt + 1
                r = int(rng.rng_u32(key, uid, P_MUTATE, cnt) & wp.uint32(0x7FFFFFFF))
                cnt = cnt + 1
                if v < p.p_word_replace:
                    w = int(rng.rng_u32(key, uid, P_MUTATE, cnt))
                    cnt = cnt + 1
                else:
                    field = r % 4
                    if w_op(w) == OP_SET and field >= 2 and p.imm_nudge_sigma > 0.0:
                        imm = (w_b(w) << 8) | w_c(w)
                        if imm >= 32768:
                            imm = imm - 65536
                        z = rng.rng_normal(key, uid, P_MUTATE, cnt)
                        cnt = cnt + 1
                        imm = imm + int(wp.round(z * p.imm_nudge_sigma * 64.0))
                        imm = wp.clamp(imm, -32768, 32767) & 0xFFFF
                        w = _set_byte(w, 2, (imm >> 8) & 0xFF)
                        w = _set_byte(w, 3, imm & 0xFF)
                    else:
                        w = _set_byte(w, field, (r >> 2) & 0xFF)
                g.words[dst, k] = w
    # ---------------------------------------------------------------- structural events
    lo = p.min_genome_len
    # insert
    u = rng.rng_uniform(key, uid, P_MUTATE, 100000)
    if u < p.mu_insert * mult and n < MAX_LEN:
        pos = int(rng.rng_u32(key, uid, P_MUTATE, 100001) % wp.uint32(n + 1))
        k = n
        while k > pos:
            g.words[dst, k] = g.words[dst, k - 1]
            k = k - 1
        g.words[dst, pos] = int(rng.rng_u32(key, uid, P_MUTATE, 100002))
        n = n + 1
        events = events + 1
    # delete
    u = rng.rng_uniform(key, uid, P_MUTATE, 100010)
    if u < p.mu_delete * mult and n > lo and n > 1:
        pos = int(rng.rng_u32(key, uid, P_MUTATE, 100011) % wp.uint32(n))
        for k in range(pos, n - 1):
            g.words[dst, k] = g.words[dst, k + 1]
        n = n - 1
        events = events + 1
    # duplicate
    u = rng.rng_uniform(key, uid, P_MUTATE, 100020)
    if u < p.mu_duplicate * mult:
        sl = wp.min(_geom_len(rng.rng_uniform(key, uid, P_MUTATE, 100021), p.dup_len_mean), n)
        if n + sl <= MAX_LEN and sl > 0:
            st = int(rng.rng_u32(key, uid, P_MUTATE, 100022) % wp.uint32(n - sl + 1))
            k = n - 1
            while k >= st + sl:
                g.words[dst, k + sl] = g.words[dst, k]
                k = k - 1
            for q in range(sl):
                g.words[dst, st + sl + q] = g.words[dst, st + q]
            n = n + sl
            events = events + 1
    # invert
    u = rng.rng_uniform(key, uid, P_MUTATE, 100030)
    if u < p.mu_invert * mult and n > 1:
        sl = wp.min(_geom_len(rng.rng_uniform(key, uid, P_MUTATE, 100031), p.dup_len_mean), n)
        st = int(rng.rng_u32(key, uid, P_MUTATE, 100032) % wp.uint32(n - sl + 1))
        _reverse(g, dst, st, st + sl)
        events = events + 1
    # transpose
    u = rng.rng_uniform(key, uid, P_MUTATE, 100040)
    if u < p.mu_transpose * mult and n > 1:
        sl = wp.min(_geom_len(rng.rng_uniform(key, uid, P_MUTATE, 100041), p.dup_len_mean), n - 1)
        if sl > 0:
            s0 = int(rng.rng_u32(key, uid, P_MUTATE, 100042) % wp.uint32(n - sl + 1))
            t0 = int(rng.rng_u32(key, uid, P_MUTATE, 100043) % wp.uint32(n - sl + 1))
            if t0 < s0:
                _reverse(g, dst, t0, s0)
                _reverse(g, dst, s0, s0 + sl)
                _reverse(g, dst, t0, s0 + sl)
            elif t0 > s0:
                _reverse(g, dst, s0, s0 + sl)
                _reverse(g, dst, s0 + sl, t0 + sl)
                _reverse(g, dst, s0, t0 + sl)
            events = events + 1
    g.length[dst] = n
    # identical to the source?
    same = int(0)
    if n == g.length[src]:
        same = int(1)
        for k in range(n):
            if g.words[dst, k] != g.words[src, k]:
                same = int(0)
                break
    jobs.events[i] = events
    jobs.same[i] = same
    if same == 0:
        finalize_genome(g, dst)
        g.parent[dst] = src
        g.first_tick[dst] = p.d_tick
