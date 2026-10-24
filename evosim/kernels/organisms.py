"""Connected components over bonds by min-label propagation with pointer jumping.

The fixed point (each cell labelled with the minimum slot index in its component) is unique, so
the result is deterministic even though in-place updates race. Converges in O(log diameter)
rounds thanks to pointer jumping; the host checks a change counter every few rounds.
"""
import warp as wp

from ..state import C_ORG_CHANGED, Bonds, Cells, Scratch
from .sense import other_end


@wp.kernel
def init_kernel(c: Cells, s: Scratch):
    i = wp.tid()
    s.org_size[i] = 0
    if c.alive[i] != 0:
        s.org_tmp[i] = i
    else:
        s.org_tmp[i] = -1


@wp.kernel
def propagate_kernel(c: Cells, bd: Bonds, s: Scratch, counters: wp.array(dtype=wp.int64), mb: int):
    i = wp.tid()
    if c.alive[i] == 0:
        return
    cur = s.org_tmp[i]
    m = cur
    for sl in range(mb):
        b = c.bonds[i, sl]
        if b >= 0:
            o = other_end(bd, b, i)
            v = s.org_tmp[o]
            if v >= 0 and v < m:
                m = v
    j = s.org_tmp[m]           # pointer jump
    if j >= 0 and j < m:
        m = j
    if m < cur:
        s.org_tmp[i] = m
        wp.atomic_add(counters, C_ORG_CHANGED, wp.int64(1))


@wp.kernel
def finish_kernel(c: Cells, s: Scratch):
    i = wp.tid()
    if c.alive[i] == 0:
        return
    r = s.org_tmp[i]
    c.organism_id[i] = r
    wp.atomic_add(s.org_size, r, 1)


def label(state, max_rounds=256):
    dev = state.device
    N = state.dims.N
    c, bd, s = state.cells, state.bonds, state.scratch
    wp.launch(init_kernel, N, [c, s], device=dev)
    rounds = 0
    while rounds < max_rounds:
        state.counters[C_ORG_CHANGED:C_ORG_CHANGED + 1].zero_()
        for _ in range(4):
            wp.launch(propagate_kernel, N, [c, bd, s, state.counters, state.dims.MB], device=dev)
        rounds += 4
        if int(state.counters[C_ORG_CHANGED:C_ORG_CHANGED + 1].numpy()[0]) == 0:
            break
    wp.launch(finish_kernel, N, [c, s], device=dev)
    return rounds
