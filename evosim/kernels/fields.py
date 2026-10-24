"""Scalar fields on a toroidal grid: apply fixed-point deposits, explicit diffusion (conservative
flux form, no flux into obstacles, stability-limited substeps), decay, regeneration toward the
capacity pattern.
"""
import math

import numpy as np
import warp as wp

from ..config import SimParams
from ..dims import wrap_idx
from ..state import L_DECAY, L_REGEN, Fields
from .common import unfx


@wp.func
def chem_D(p: SimParams, k: int) -> float:
    if k == 0:
        return p.D_chem_0
    if k == 1:
        return p.D_chem_1
    if k == 2:
        return p.D_chem_2
    if k == 3:
        return p.D_chem_3
    if k == 4:
        return p.D_chem_4
    if k == 5:
        return p.D_chem_5
    if k == 6:
        return p.D_chem_6
    return p.D_chem_7


@wp.func
def chem_decay(p: SimParams, k: int) -> float:
    if k == 0:
        return p.chem_decay_0
    if k == 1:
        return p.chem_decay_1
    if k == 2:
        return p.chem_decay_2
    if k == 3:
        return p.chem_decay_3
    if k == 4:
        return p.chem_decay_4
    if k == 5:
        return p.chem_decay_5
    if k == 6:
        return p.chem_decay_6
    return p.chem_decay_7


@wp.kernel
def deposit_kernel(f: Fields, p: SimParams):
    ix, iy = wp.tid()
    inv = 1.0 / (p.field_cell * p.field_cell)
    q = f.d_nutrient[ix, iy]
    if q != wp.int64(0):
        f.nutrient[ix, iy] = f.nutrient[ix, iy] + unfx(q) * inv
        f.d_nutrient[ix, iy] = wp.int64(0)
    for k in range(p.k_chem):
        qc = f.d_chem[k, ix, iy]
        if qc != wp.int64(0):
            f.chem[k, ix, iy] = f.chem[k, ix, iy] + unfx(qc) * inv
            f.d_chem[k, ix, iy] = wp.int64(0)


@wp.func
def _flux_sum(src: wp.array2d(dtype=float), obs: wp.array2d(dtype=wp.int32), ix: int, iy: int, nx: int, ny: int,
              use_obs: int) -> float:
    v = src[ix, iy]
    acc = float(0.0)
    for q in range(4):
        dx = 0
        dy = 0
        if q == 0:
            dx = 1
        elif q == 1:
            dx = -1
        elif q == 2:
            dy = 1
        else:
            dy = -1
        jx = wrap_idx(ix + dx, nx)
        jy = wrap_idx(iy + dy, ny)
        if use_obs == 0 or obs[jx, jy] == 0:
            acc = acc + (src[jx, jy] - v)
    return acc


@wp.kernel
def diffuse2d_kernel(src: wp.array2d(dtype=float), dst: wp.array2d(dtype=float), obs: wp.array2d(dtype=wp.int32),
                     alpha: float, use_obs: int):
    ix, iy = wp.tid()
    nx = src.shape[0]
    ny = src.shape[1]
    if use_obs != 0 and obs[ix, iy] != 0:
        dst[ix, iy] = src[ix, iy]
        return
    dst[ix, iy] = src[ix, iy] + alpha * _flux_sum(src, obs, ix, iy, nx, ny, use_obs)


@wp.kernel
def diffuse_chem_kernel(src: wp.array3d(dtype=float), dst: wp.array3d(dtype=float), obs: wp.array2d(dtype=wp.int32),
                        alphas: wp.array(dtype=float), use_obs: int):
    k, ix, iy = wp.tid()
    nx = src.shape[1]
    ny = src.shape[2]
    v = src[k, ix, iy]
    if use_obs != 0 and obs[ix, iy] != 0:
        dst[k, ix, iy] = v
        return
    acc = float(0.0)
    for q in range(4):
        dx = 0
        dy = 0
        if q == 0:
            dx = 1
        elif q == 1:
            dx = -1
        elif q == 2:
            dy = 1
        else:
            dy = -1
        jx = wrap_idx(ix + dx, nx)
        jy = wrap_idx(iy + dy, ny)
        if use_obs == 0 or obs[jx, jy] == 0:
            acc = acc + (src[k, jx, jy] - v)
    dst[k, ix, iy] = v + alphas[k] * acc


@wp.kernel
def regen_decay_kernel(f: Fields, ledger: wp.array(dtype=wp.float64), p: SimParams):
    ix, iy = wp.tid()
    nx = p.d_field_nx
    area = p.field_cell * p.field_cell
    if p.obstacles_enabled != 0 and f.obstacle[ix, iy] != 0:
        f.nutrient[ix, iy] = 0.0
        for k in range(p.k_chem):
            f.chem[k, ix, iy] = 0.0
        return
    # pattern drifts along x
    sx = wrap_idx(ix - int(wp.floor(p.d_nutrient_offset / p.field_cell)), nx)
    target = p.nutrient_cap * f.cap_map[sx, iy]
    n = f.nutrient[ix, iy]
    reg = float(0.0)
    if n < target:
        reg = wp.min(p.nutrient_regen * (target - n) * p.dt, target - n)
    dec = p.nutrient_decay * n * p.dt
    f.nutrient[ix, iy] = wp.max(0.0, n + reg - dec)
    if p.d_ledger_on != 0:
        wp.atomic_add(ledger, L_REGEN, wp.float64(reg * area))
        wp.atomic_add(ledger, L_DECAY, wp.float64(dec * area))
    for k in range(p.k_chem):
        f.chem[k, ix, iy] = f.chem[k, ix, iy] * wp.max(0.0, 1.0 - chem_decay(p, k) * p.dt)


def _substeps(D, dt, h):
    a = D * dt / (h * h)
    n = max(1, int(math.ceil(a / 0.2))) if a > 0 else 0
    return n, (a / n if n else 0.0)


def update(state, p, cfg):
    dev = state.device
    d = state.dims
    f = state.fields
    use_obs = int(bool(cfg.obstacles_enabled))
    wp.launch(deposit_kernel, (d.fnx, d.fny), [f, p], device=dev)
    # nutrient diffusion
    n, a = _substeps(cfg.D_nutrient, cfg.dt, d.fh)
    for _ in range(n):
        wp.launch(diffuse2d_kernel, (d.fnx, d.fny), [f.nutrient, f.nutrient_tmp, f.obstacle, a, use_obs], device=dev)
        f.nutrient, f.nutrient_tmp = f.nutrient_tmp, f.nutrient
    # chemical diffusion (all channels share the substep count of the fastest channel)
    Ds = [getattr(cfg, f"D_chem_{k}") for k in range(d.K)]
    n = max(_substeps(D, cfg.dt, d.fh)[0] for D in Ds) if Ds else 0
    if n > 0:
        alphas = np.array([D * cfg.dt / (d.fh * d.fh) / n for D in Ds], np.float32)
        if state.__dict__.get("_chem_alpha") is None or state._chem_alpha.shape[0] != d.K:
            state._chem_alpha = wp.zeros(d.K, dtype=float, device=dev)
        state._chem_alpha.assign(alphas)
        for _ in range(n):
            wp.launch(diffuse_chem_kernel, (d.K, d.fnx, d.fny), [f.chem, f.chem_tmp, f.obstacle, state._chem_alpha,
                                                                 use_obs], device=dev)
            f.chem, f.chem_tmp = f.chem_tmp, f.chem
    wp.launch(regen_decay_kernel, (d.fnx, d.fny), [f, state.ledger, p], device=dev)


# ----------------------------------------------------------------------------- patterns (host)
def value_noise(nx, ny, cell, rs):
    """Smooth periodic value noise in [0,1] (bicubic-ish interpolation of a coarse random lattice)."""
    gx = max(1, int(round(nx / cell)))
    gy = max(1, int(round(ny / cell)))
    lat = rs.random((gx, gy))
    xs = np.arange(nx) / nx * gx
    ys = np.arange(ny) / ny * gy
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    tx = xs - x0
    ty = ys - y0
    sx = tx * tx * (3 - 2 * tx)
    sy = ty * ty * (3 - 2 * ty)
    x1 = (x0 + 1) % gx
    y1 = (y0 + 1) % gy
    x0 %= gx
    y0 %= gy
    a = lat[np.ix_(x0, y0)]
    b = lat[np.ix_(x1, y0)]
    c = lat[np.ix_(x0, y1)]
    d = lat[np.ix_(x1, y1)]
    sxm = sx[:, None]
    sym = sy[None, :]
    v = a * (1 - sxm) * (1 - sym) + b * sxm * (1 - sym) + c * (1 - sxm) * sym + d * sxm * sym
    v = (v - v.min()) / max(1e-9, v.max() - v.min())
    return v.astype(np.float32)


def make_cap_map(cfg, dims, rs):
    nx, ny, h = dims.fnx, dims.fny, dims.fh
    if cfg.nutrient_pattern == "uniform":
        return np.ones((nx, ny), np.float32)
    if cfg.nutrient_pattern == "patches":
        v = value_noise(nx, ny, cfg.pattern_scale / h, rs)
        return np.clip((v - 0.4) / 0.6, 0, 1).astype(np.float32) ** 2
    # vents: gaussian blobs (periodic)
    cap = np.zeros((nx, ny), np.float32)
    X = (np.arange(nx) + 0.5) * h
    Y = (np.arange(ny) + 0.5) * h
    W, H = dims.world
    sig = cfg.pattern_scale * 0.25
    for _ in range(cfg.vent_count):
        cx, cy = rs.random() * W, rs.random() * H
        dx = X - cx
        dx -= W * np.round(dx / W)
        dy = Y - cy
        dy -= H * np.round(dy / H)
        cap += np.exp(-(dx[:, None] ** 2 + dy[None, :] ** 2) / (2 * sig * sig)).astype(np.float32)
    return np.clip(cap / max(1e-9, cap.max()), 0, 1).astype(np.float32)


def make_obstacles(cfg, dims, rs):
    nx, ny, h = dims.fnx, dims.fny, dims.fh
    gen = cfg.obstacle_generator
    m = np.zeros((nx, ny), np.int32)
    X, Y = np.meshgrid((np.arange(nx) + 0.5) * h, (np.arange(ny) + 0.5) * h, indexing="ij")
    W, H = dims.world
    if gen == "rocks":
        for _ in range(max(1, int(W * H / 200000))):
            cx, cy, r = rs.random() * W, rs.random() * H, 10 + rs.random() * 40
            dx = X - cx
            dx -= W * np.round(dx / W)
            dy = Y - cy
            dy -= H * np.round(dy / H)
            m[dx * dx + dy * dy < r * r] = 1
    elif gen == "ring":
        r = np.hypot(X - W / 2, Y - H / 2)
        R = min(W, H) * 0.3
        m[(np.abs(r - R) < 6)] = 1
        m[(np.abs(Y - H / 2) < 30) & (X > W / 2)] = 0        # gap
    elif gen == "channels":
        period = max(64.0, W / 8)
        m[(np.mod(Y, period) < 6) & (np.mod(X, W / 2) > 40)] = 1
    elif gen == "maze":
        cell = 64.0
        gx, gy = int(W // cell), int(H // cell)
        walls_h = rs.random((gx, gy)) < 0.45
        walls_v = rs.random((gx, gy)) < 0.45
        ix = (X // cell).astype(int) % gx
        iy = (Y // cell).astype(int) % gy
        m[(np.mod(Y, cell) < 4) & walls_h[ix, iy]] = 1
        m[(np.mod(X, cell) < 4) & walls_v[ix, iy]] = 1
    return m


def load_mask_png(path, dims):
    """Grayscale PNG -> mask resampled to the field grid (needs Pillow, optional)."""
    from PIL import Image
    im = Image.open(path).convert("L").resize((dims.fnx, dims.fny))
    return (np.asarray(im).T > 127).astype(np.int32)
