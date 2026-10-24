"""Uniform spatial hash on the torus: key = bucket index, stable radix sort, then bucket ranges.

Bucket size >= max cell diameter (+ motion margin), so every possible contact lies in the 3x3
neighbourhood. Sorted order is deterministic (stable sort on index), which makes every gather
loop over neighbours deterministic too.
"""
import warp as wp

from ..config import SimParams
from ..dims import grid_coord
from ..state import Cells, Hash


@wp.kernel
def keys_kernel(c: Cells, h: Hash, p: SimParams):
    i = wp.tid()
    key = p.d_hash_nx * p.d_hash_ny          # sentinel for dead cells (sorted last)
    if c.alive[i] != 0:
        gc = grid_coord(c.pos[i], p.d_hash_cell, p.d_hash_nx, p.d_hash_ny)
        key = gc[1] * p.d_hash_nx + gc[0]
    h.keys[i] = key
    h.vals[i] = i


@wp.kernel
def ranges_kernel(h: Hash, n: int, nb: int):
    k = wp.tid()
    key = h.keys[k]
    if key >= nb:
        return
    if k == 0 or h.keys[k - 1] != key:
        h.start[key] = k
    if k == n - 1 or h.keys[k + 1] != key:
        h.end[key] = k + 1


def build(state, p):
    h = state.hash
    n = state.dims.N
    nb = p.d_hash_nx * p.d_hash_ny
    dev = state.device
    wp.launch(keys_kernel, n, [state.cells, h, p], device=dev)
    wp.utils.radix_sort_pairs(h.keys, h.vals, n)
    h.start.zero_()
    h.end.zero_()
    wp.launch(ranges_kernel, n, [h, n, nb], device=dev)


def choose_bucket(cfg, world):
    """Bucket size >= max diameter plus a margin for motion within one tick."""
    import math
    r_max = math.sqrt(cfg.max_mass / (math.pi * cfg.density))
    cell = max(4.0, 2.0 * r_max * 1.05 + 2.0 * cfg.v_max * cfg.dt, cfg.light_cell * 1.01)
    nx = max(3, int(world[0] // cell))
    ny = max(3, int(world[1] // cell))
    # actual bucket = world / n (>= cell). Use the larger of the two extents for a square bucket.
    hc = max(world[0] / nx, world[1] / ny)
    nx = max(3, int(world[0] // hc))
    ny = max(3, int(world[1] // hc))
    return hc, nx, ny
