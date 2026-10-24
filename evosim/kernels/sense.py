"""Sense phase: contacts by body-frame sector, fields at the cell's own position, bond signals of
the previous tick, damage, own state. Only local physical quantities -- no sensor reports the
distance, direction or identity of another cell. (contacts[] / sector_best[] are NOT readable by
the genome; they are used only to resolve BOND / EAT targets.)
"""
import warp as wp

from .. import rng
from ..config import SimParams
from ..dims import grid_coord, min_image, sector_of, to_body, wrap_idx
from ..genome import isa
from ..state import MAX_CONTACTS, Bonds, Cells, Fields, Hash, Scratch
from .light import body_radius

NB = isa.NUM_BASE_SENSORS
NT = isa.NUM_TRAITS
T_PHOTORECEPTOR = isa.T_PHOTORECEPTOR
S_ENERGY, S_ENERGY_FRAC, S_MASS, S_RADIUS, S_AGE = isa.S_ENERGY, isa.S_ENERGY_FRAC, isa.S_MASS, isa.S_RADIUS, isa.S_AGE
S_SPEED, S_ANG_VEL, S_VEL_FWD, S_VEL_SIDE = isa.S_SPEED, isa.S_ANG_VEL, isa.S_VEL_FWD, isa.S_VEL_SIDE
S_LIGHT, S_LIGHT_DELTA, S_NUTRIENT, S_TOUCH_TOTAL = isa.S_LIGHT, isa.S_LIGHT_DELTA, isa.S_NUTRIENT, isa.S_TOUCH_TOTAL
S_TOUCH_F, S_DAMAGE, S_BOND_COUNT, S_BOND_STRAIN, S_RANDOM = (isa.S_TOUCH_F, isa.S_DAMAGE, isa.S_BOND_COUNT,
                                                              isa.S_BOND_STRAIN, isa.S_RANDOM)
AGE_SCALE = isa.AGE_SCALE
P_SENSOR = rng.P_SENSOR


@wp.func
def contact_force_mag(overlap: float, p: SimParams) -> float:
    return p.k_contact * wp.pow(overlap, p.contact_exp)


@wp.func
def other_end(bd: Bonds, b: int, i: int) -> int:
    if bd.a[b] == i:
        return bd.b[b]
    return bd.a[b]


@wp.kernel
def contacts_kernel(c: Cells, h: Hash, s: Scratch, p: SimParams):
    i = wp.tid()
    if c.alive[i] == 0:
        return
    pi = c.pos[i]
    ri = body_radius(c, i, p)
    ang = c.angle[i]
    sec_f = wp.vec4(0.0, 0.0, 0.0, 0.0)
    best_ov = wp.vec4(0.0, 0.0, 0.0, 0.0)
    best = wp.vec4i(-1, -1, -1, -1)
    nc = int(0)
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
                    if ov > 0.0:
                        sec = 0
                        if dist > 1.0e-6:
                            sec = sector_of(d, ang)
                        fm = contact_force_mag(ov, p)
                        sec_f[sec] = sec_f[sec] + fm
                        if ov > best_ov[sec]:
                            best_ov[sec] = ov
                            best[sec] = j
                        if nc < MAX_CONTACTS:
                            s.contacts[i, nc] = j
                            nc = nc + 1
    s.n_contacts[i] = nc
    for q in range(4):
        s.sensors[i, S_TOUCH_F + q] = sec_f[q]
        s.sector_best[i, q] = best[q]
    s.sensors[i, S_TOUCH_TOTAL] = sec_f[0] + sec_f[1] + sec_f[2] + sec_f[3]


@wp.kernel
def sensors_kernel(c: Cells, bd: Bonds, f: Fields, s: Scratch, p: SimParams):
    i = wp.tid()
    if c.alive[i] == 0:
        return
    m = c.mass[i]
    e = c.energy[i]
    cap = p.max_energy_per_mass * m
    r = body_radius(c, i, p)
    v = c.vel[i]
    vb = to_body(v, c.angle[i])
    s.sensors[i, S_ENERGY] = e
    s.sensors[i, S_ENERGY_FRAC] = e / wp.max(cap, 1.0e-6)
    s.sensors[i, S_MASS] = m
    s.sensors[i, S_RADIUS] = r
    s.sensors[i, S_AGE] = float(c.age[i]) * AGE_SCALE
    s.sensors[i, S_SPEED] = wp.length(v)
    s.sensors[i, S_ANG_VEL] = c.ang_vel[i]
    s.sensors[i, S_VEL_FWD] = vb[0]
    s.sensors[i, S_VEL_SIDE] = vb[1]
    light = s.light_recv[i]
    s.sensors[i, S_LIGHT] = light
    s.sensors[i, S_LIGHT_DELTA] = light - c.light_prev[i]
    c.light_prev[i] = light
    fg = grid_coord(c.pos[i], p.field_cell, p.d_field_nx, p.d_field_ny)
    s.sensors[i, S_NUTRIENT] = f.nutrient[fg[0], fg[1]]
    for k in range(p.k_chem):
        s.sensors[i, NB + k] = f.chem[k, fg[0], fg[1]]
    s.sensors[i, S_DAMAGE] = c.damage[i]
    s.sensors[i, S_RANDOM] = rng.rng_uniform(p.d_key, c.uid[i], P_SENSOR, 1000)
    for t in range(NT):
        s.sensors[i, NB + p.k_chem + t] = c.traits[i, t]
    # bonds: previous-tick signals (one-tick delay), count, strain
    ks = p.k_sig
    nb = int(0)
    strain = float(0.0)
    for q in range(ks):
        s.recv_agg[i, q] = 0.0
    for sl in range(p.max_bonds):
        b = c.bonds[i, sl]
        for q in range(ks):
            val = float(0.0)
            if b >= 0:
                o = other_end(bd, b, i)
                val = c.signal_prev[o, q]
            s.recv_slot[i, sl * ks + q] = val
        if b >= 0:
            nb = nb + 1
            strain = wp.max(strain, wp.abs(bd.strain[b]))
            for q in range(ks):
                val = s.recv_slot[i, sl * ks + q]
                if p.recv_mode == 2:
                    if nb == 1 or val > s.recv_agg[i, q]:
                        s.recv_agg[i, q] = val
                else:
                    s.recv_agg[i, q] = s.recv_agg[i, q] + val
    if p.recv_mode == 1 and nb > 0:
        for q in range(ks):
            s.recv_agg[i, q] = s.recv_agg[i, q] / float(nb)
    s.sensors[i, S_BOND_COUNT] = float(nb)
    s.sensors[i, S_BOND_STRAIN] = strain
    # additive Gaussian noise (not on RANDOM). Light noise scaled by 1/(1+gain*PHOTORECEPTOR)
    if p.sensor_noise > 0.0:
        lscale = 1.0 / (1.0 + p.photoreceptor_gain * c.traits[i, T_PHOTORECEPTOR])
        for q in range(p.d_num_sensors):
            if q != S_RANDOM:
                sc = p.sensor_noise
                if q == S_LIGHT or q == S_LIGHT_DELTA:
                    sc = sc * lscale
                s.sensors[i, q] = s.sensors[i, q] + sc * rng.rng_normal(p.d_key, c.uid[i], P_SENSOR, q)


def compute(state, p):
    dev = state.device
    wp.launch(contacts_kernel, state.dims.N, [state.cells, state.hash, state.scratch, p], device=dev)
    wp.launch(sensors_kernel, state.dims.N, [state.cells, state.bonds, state.fields, state.scratch, p], device=dev)
