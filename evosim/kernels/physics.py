"""Physics substeps: contacts, bond springs (linear + angular at both ends), thrust, turn, drag,
obstacles, semi-implicit Euler, wrapping.

Gather formulation: each cell sums the forces acting on itself. Every pairwise force is computed
from an antisymmetric expression (min_image(-d) == -min_image(d)), so F_ij == -F_ji and the
internal forces of an organism sum to zero: it cannot move its centre of mass without THRUST.

Angular bond springs, end A with rest angle alpha_A (in A's body frame):
    phi_A = wrap(atan2(d) - theta_A - alpha_A),  E_A = 1/2 k phi_A^2,  d = p_B - p_A
    torque_A = k phi_A ;  F_B = -k phi_A perp(d)/|d|^2 ;  F_A = -F_B
The two transverse forces carry exactly the opposite torque, so angular momentum is conserved too.
"""
import warp as wp

from ..config import SimParams
from ..dims import (cross2, direction_angle, grid_coord, heading, inertia, min_image, perp, wrap_angle, wrap_idx,
                    wrap_pos)
from ..genome import isa
from ..state import C_VCLAMP, C_WCLAMP, Bonds, Cells, Fields, Hash, Scratch
from .light import body_radius
from .sense import contact_force_mag

T_ARMOR = isa.T_ARMOR


@wp.func
def bond_rest_len(c: Cells, bd: Bonds, b: int, p: SimParams) -> float:
    if p.rest_length_tracks_radius != 0:
        return body_radius(c, bd.a[b], p) + body_radius(c, bd.b[b], p)
    return bd.rest_len[b]


@wp.kernel
def forces_kernel(c: Cells, bd: Bonds, h: Hash, f: Fields, s: Scratch, p: SimParams):
    i = wp.tid()
    if c.alive[i] == 0:
        return
    pi = c.pos[i]
    vi = c.vel[i]
    ri = body_radius(c, i, p)
    th = c.angle[i]
    F = wp.vec2(0.0, 0.0)
    tq = float(0.0)
    # ---------------------------------------------------------------- contacts
    gc = grid_coord(pi, p.d_hash_cell, p.d_hash_nx, p.d_hash_ny)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            bx = wrap_idx(gc[0] + dx, p.d_hash_nx)
            by = wrap_idx(gc[1] + dy, p.d_hash_ny)
            b = by * p.d_hash_nx + bx
            for k in range(h.start[b], h.end[b]):
                j = h.vals[k]
                if j != i:
                    d = min_image(c.pos[j] - pi, p.d_world)
                    dist = wp.length(d)
                    ov = ri + body_radius(c, j, p) - dist
                    if ov > 0.0 and dist > 1.0e-6:
                        n = d / dist
                        F = F - n * contact_force_mag(ov, p)
                        F = F + n * (p.contact_damp * wp.dot(c.vel[j] - vi, n))
    # ---------------------------------------------------------------- bonds
    for sl in range(p.max_bonds):
        b = c.bonds[i, sl]
        if b >= 0:
            is_a = bd.a[b] == i
            o = bd.b[b]
            if not is_a:
                o = bd.a[b]
            d = min_image(c.pos[o] - pi, p.d_world)
            L = wp.length(d)
            if L > 1.0e-6:
                u = d / L
                rest = bond_rest_len(c, bd, b, p)
                F = F + u * (p.k_bond * (L - rest) + p.c_bond * wp.dot(c.vel[o] - vi, u))
                # angular springs at both ends
                ang_d = direction_angle(d)
                ang_e = direction_angle(-d)
                ra = bd.rest_ang_a[b]
                rb = bd.rest_ang_b[b]
                if not is_a:
                    ra = bd.rest_ang_b[b]
                    rb = bd.rest_ang_a[b]
                phi_i = wrap_angle(ang_d - th - ra)
                phi_o = wrap_angle(ang_e - c.angle[o] - rb)
                ka = p.k_bond_angle * rest * rest      # scale-invariant angular stiffness
                tq = tq + ka * phi_i
                F = F + perp(d) * (ka * (phi_i + phi_o) / (L * L))
    # ---------------------------------------------------------------- thrust, current, drag
    F = F + heading(th) * s.thrust_force[i]
    F = F + wp.vec2(p.current_x, p.current_y) * c.mass[i]
    arm = 1.0 + p.armor_drag * c.traits[i, T_ARMOR]
    gam = (p.drag_lin + p.drag_quad * wp.length(vi)) * ri * arm
    F = F - vi * gam
    tq = tq + s.turn_torque[i] - p.ang_drag * ri * ri * ri * c.ang_vel[i]
    # ---------------------------------------------------------------- obstacles
    if p.obstacles_enabled != 0:
        fh = p.field_cell
        fg = grid_coord(pi, fh, p.d_field_nx, p.d_field_ny)
        reach = int(wp.ceil(ri / fh)) + 1
        for ox in range(-reach, reach + 1):
            for oy in range(-reach, reach + 1):
                gx = wrap_idx(fg[0] + ox, p.d_field_nx)
                gy = wrap_idx(fg[1] + oy, p.d_field_ny)
                if f.obstacle[gx, gy] != 0:
                    cen = wp.vec2((float(fg[0] + ox) + 0.5) * fh, (float(fg[1] + oy) + 0.5) * fh)
                    rel = min_image(pi - cen, p.d_world)
                    q = wp.vec2(wp.clamp(rel[0], -0.5 * fh, 0.5 * fh), wp.clamp(rel[1], -0.5 * fh, 0.5 * fh))
                    dd = rel - q
                    dist = wp.length(dd)
                    if dist < ri:
                        nrm = wp.vec2(0.0, 0.0)
                        if dist > 1.0e-6:
                            nrm = dd / dist
                        else:
                            ln = wp.length(rel)
                            if ln > 1.0e-6:
                                nrm = rel / ln
                            else:
                                nrm = wp.vec2(1.0, 0.0)
                        F = F + nrm * (p.obstacle_k * (ri - dist))
    s.force[i] = F
    s.torque[i] = tq


@wp.kernel
def integrate_kernel(c: Cells, s: Scratch, counters: wp.array(dtype=wp.int64), dts: float, p: SimParams):
    i = wp.tid()
    if c.alive[i] == 0:
        return
    m = wp.max(c.mass[i], 1.0e-6)
    r = body_radius(c, i, p)
    I = wp.max(inertia(m, r), 1.0e-6)
    v = c.vel[i] + s.force[i] * (dts / m)
    w = c.ang_vel[i] + s.torque[i] * (dts / I)
    sp = wp.length(v)
    if sp > p.v_max:
        v = v * (p.v_max / sp)
        wp.atomic_add(counters, C_VCLAMP, wp.int64(1))
    if wp.abs(w) > p.w_max:
        w = wp.sign(w) * p.w_max
        wp.atomic_add(counters, C_WCLAMP, wp.int64(1))
    c.vel[i] = v
    c.ang_vel[i] = w
    c.pos[i] = wrap_pos(c.pos[i] + v * dts, p.d_world)
    c.angle[i] = wrap_angle(c.angle[i] + w * dts)


@wp.kernel
def bond_strain_kernel(c: Cells, bd: Bonds, p: SimParams):
    b = wp.tid()
    if bd.alive[b] == 0:
        return
    d = min_image(c.pos[bd.b[b]] - c.pos[bd.a[b]], p.d_world)
    rest = wp.max(bond_rest_len(c, bd, b, p), 1.0e-6)
    bd.strain[b] = (wp.length(d) - rest) / rest


def step(state, p, substeps, dt):
    dev = state.device
    N = state.dims.N
    c, bd, h, f, s = state.cells, state.bonds, state.hash, state.fields, state.scratch
    dts = dt / max(1, substeps)
    for _ in range(max(1, substeps)):
        wp.launch(forces_kernel, N, [c, bd, h, f, s, p], device=dev)
        wp.launch(integrate_kernel, N, [c, s, state.counters, dts, p], device=dev)
    wp.launch(bond_strain_kernel, state.dims.B, [c, bd, p], device=dev)
