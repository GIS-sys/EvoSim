"""Light model.

1. Opacity rasterization (gather, deterministic): every light-grid node sums the cloud-in-cell
   (bilinear) weights of nearby cell bodies, giving an opacity *density* field (1/u):
       rho(node) = sum_i w_i(node) * opacity_i * area_i / h^2 + obstacle term
   opacity_i = base_opacity + pigment_opacity * PIGMENT_i (opacity per unit path length).
2. Per cell, per source: march a ray toward the source, optical depth tau = sum rho(x_k) * step,
   where rho is bilinearly interpolated and the cell's OWN contribution is subtracted analytically
   (own footprint excluded). Transmittance = exp(-tau) (* exp(-k_water*r) for point lights).
3. Directional reception: w(theta) = RW + (1-RW) * max(0, cos theta)^p (RW = RECEPTOR_WIDTH).
   LIGHT (sensor) = sum_s I_s T_s w_s + ambient * avg(w);  photosynthesis uses sum_s I_s T_s + ambient.
"""
import warp as wp

from ..config import SimParams
from ..dims import heading, min_image, radius_of, grid_coord, wrap_idx, disc_area
from ..genome import isa
from ..state import Cells, Fields, Hash, Scratch

T_PIGMENT = isa.T_PIGMENT
T_ARMOR = isa.T_ARMOR
T_RWIDTH = isa.T_RWIDTH


@wp.func
def body_radius(c: Cells, i: int, p: SimParams) -> float:
    rho = p.density * (1.0 + p.armor_density * c.traits[i, T_ARMOR])
    return radius_of(c.mass[i], rho)


@wp.func
def cell_opacity(c: Cells, i: int, p: SimParams) -> float:
    return p.base_opacity + p.pigment_opacity * c.traits[i, T_PIGMENT]


@wp.func
def cic_w(node_pos: wp.vec2, x: wp.vec2, h: float, world: wp.vec2) -> float:
    d = min_image(x - node_pos, world)
    wx = wp.max(0.0, 1.0 - wp.abs(d[0]) / h)
    wy = wp.max(0.0, 1.0 - wp.abs(d[1]) / h)
    return wx * wy


@wp.func
def light_map_value(x: wp.vec2, p: SimParams) -> float:
    if p.light_map_mode == 0:
        return 1.0
    k = 6.283185307179586 / p.light_map_scale
    s = float(0.0)
    if p.light_map_mode == 1:      # bands
        s = wp.sin((x[0] + p.d_light_map_offset) * k)
    else:                          # patches
        s = wp.sin((x[0] + p.d_light_map_offset) * k) * wp.sin(x[1] * k)
    return 1.0 - p.light_map_contrast * 0.5 * (1.0 - s)


@wp.kernel
def rasterize_kernel(c: Cells, h: Hash, f: Fields, p: SimParams):
    """One thread per light-grid node (gather)."""
    tid = wp.tid()
    nx = p.d_light_nx
    ix = tid % nx
    iy = tid / nx
    hl = p.light_cell
    node = wp.vec2((float(ix) + 0.5) * hl, (float(iy) + 0.5) * hl)
    gc = grid_coord(node, p.d_hash_cell, p.d_hash_nx, p.d_hash_ny)
    acc = float(0.0)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            bx = wrap_idx(gc[0] + dx, p.d_hash_nx)
            by = wrap_idx(gc[1] + dy, p.d_hash_ny)
            b = by * p.d_hash_nx + bx
            for k in range(h.start[b], h.end[b]):
                j = h.vals[k]
                w = cic_w(node, c.pos[j], hl, p.d_world)
                if w > 0.0:
                    r = body_radius(c, j, p)
                    acc = acc + w * cell_opacity(c, j, p) * disc_area(r)
    dens = acc / (hl * hl)
    if p.obstacles_enabled != 0:
        fg = grid_coord(node, p.field_cell, p.d_field_nx, p.d_field_ny)
        if f.obstacle[fg[0], fg[1]] != 0:
            dens = dens + p.obstacle_opacity
    f.opacity[ix, iy] = dens
    f.light_map[ix, iy] = light_map_value(node, p)


@wp.func
def sample_density(f: Fields, x: wp.vec2, own_pos: wp.vec2, own_q: float, p: SimParams) -> float:
    """Bilinear opacity density at x, minus the querying cell's own CIC contribution (own_q = opa*area/h^2)."""
    hl = p.light_cell
    fx = x[0] / hl - 0.5
    fy = x[1] / hl - 0.5
    x0 = int(wp.floor(fx))
    y0 = int(wp.floor(fy))
    tx = fx - float(x0)
    ty = fy - float(y0)
    total = float(0.0)
    for ox in range(2):
        for oy in range(2):
            wx = 1.0 - tx
            if ox == 1:
                wx = tx
            wy = 1.0 - ty
            if oy == 1:
                wy = ty
            gx = wrap_idx(x0 + ox, p.d_light_nx)
            gy = wrap_idx(y0 + oy, p.d_light_ny)
            node = wp.vec2((float(x0 + ox) + 0.5) * hl, (float(y0 + oy) + 0.5) * hl)
            own = own_q * cic_w(node, own_pos, hl, p.d_world)
            total = total + wx * wy * (f.opacity[gx, gy] - own)
    return wp.max(total, 0.0)


@wp.func
def receptor_weight(dir_to_src: wp.vec2, angle: float, rw: float, pexp: float) -> float:
    cs = wp.dot(dir_to_src, heading(angle))
    return rw + (1.0 - rw) * wp.pow(wp.max(cs, 0.0), pexp)


@wp.func
def optical_depth(f: Fields, origin: wp.vec2, d: wp.vec2, length: float, steps: int, own_q: float,
                  p: SimParams) -> float:
    if steps <= 0 or length <= 0.0:
        return 0.0
    step = length / float(steps)
    tau = float(0.0)
    for k in range(steps):
        x = origin + d * ((float(k) + 0.5) * step)
        tau = tau + sample_density(f, x, origin, own_q, p) * step
    return tau


@wp.kernel
def cell_light_kernel(c: Cells, f: Fields, s: Scratch, lights: wp.array(dtype=wp.vec3), p: SimParams):
    i = wp.tid()
    if c.alive[i] == 0:
        return
    pos = c.pos[i]
    ang = c.angle[i]
    r = body_radius(c, i, p)
    own_q = cell_opacity(c, i, p) * disc_area(r) / (p.light_cell * p.light_cell)
    rw = c.traits[i, T_RWIDTH]
    total = float(0.0)
    recv = float(0.0)
    # sun
    if p.d_sun_I > 0.0:
        sI = p.d_sun_I * light_map_value(pos, p)
        tau = optical_depth(f, pos, p.d_sun_dir, p.shadow_ray_len, p.ray_steps, own_q, p)
        e = sI * wp.exp(-tau)
        total = total + e
        recv = recv + e * receptor_weight(p.d_sun_dir, ang, rw, p.receptor_p)
    # point lights
    for k in range(p.d_num_point_lights):
        L = lights[k]
        d = min_image(wp.vec2(L[0], L[1]) - pos, p.d_world)
        dist = wp.length(d)
        if dist > 1.0e-4:
            dn = d / dist
            tau = optical_depth(f, pos, dn, dist, p.ray_steps, own_q, p)
            e = L[2] / (1.0 + dist * dist / (p.point_light_r0 * p.point_light_r0)) * wp.exp(-p.k_water * dist - tau)
            total = total + e
            recv = recv + e * receptor_weight(dn, ang, rw, p.receptor_p)
        else:
            total = total + L[2]
            recv = recv + L[2]
    total = total + p.ambient_I
    recv = recv + p.ambient_I * (rw + (1.0 - rw) * p.d_amb_weight_avg)
    s.light_total[i] = total
    s.light_recv[i] = recv


@wp.kernel
def sun_transmittance_map_kernel(f: Fields, out: wp.array2d(dtype=float), p: SimParams):
    """Overlay: sun light reaching each light-grid node (no own-exclusion)."""
    ix, iy = wp.tid()
    hl = p.light_cell
    x = wp.vec2((float(ix) + 0.5) * hl, (float(iy) + 0.5) * hl)
    tau = optical_depth(f, x, p.d_sun_dir, p.shadow_ray_len, p.ray_steps, 0.0, p)
    out[ix, iy] = p.d_sun_I * f.light_map[ix, iy] * wp.exp(-tau) + p.ambient_I


def compute(state, p, lights):
    n_nodes = state.dims.lnx * state.dims.lny
    wp.launch(rasterize_kernel, n_nodes, [state.cells, state.hash, state.fields, p], device=state.device)
    wp.launch(cell_light_kernel, state.dims.N, [state.cells, state.fields, state.scratch, lights, p],
              device=state.device)
