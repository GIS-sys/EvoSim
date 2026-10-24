"""Dimension-specific math. This is the ONLY simulation module that knows the world is 2D.

Kernels must use these helpers instead of touching .x/.y or atan2 directly, so that a
3D port means rewriting this file (vec3, quaternion orientation, 3D sectors) and the
rendering code, not every kernel.
"""
import math

import numpy as np
import warp as wp

D = 2
vec = wp.vec2            # position / velocity type
NUM_SECTORS = 4          # contact sectors in the body frame: 0 front, 1 right, 2 back, 3 left
SECTOR_NAMES = ("F", "R", "B", "L")
PI = math.pi
TWO_PI = 2.0 * math.pi


# ----------------------------------------------------------------------------- warp side
@wp.func
def wrap_pos(p: wp.vec2, world: wp.vec2) -> wp.vec2:
    x = p[0] - world[0] * wp.floor(p[0] / world[0])
    y = p[1] - world[1] * wp.floor(p[1] / world[1])
    # guard against x == world after rounding
    if x >= world[0]:
        x = x - world[0]
    if y >= world[1]:
        y = y - world[1]
    return wp.vec2(x, y)


@wp.func
def min_image(d: wp.vec2, world: wp.vec2) -> wp.vec2:
    """Minimum-image displacement on the torus. Antisymmetric: min_image(-d) == -min_image(d)."""
    x = d[0] - world[0] * wp.sign(d[0]) * wp.floor(wp.abs(d[0]) / world[0] + 0.5)
    y = d[1] - world[1] * wp.sign(d[1]) * wp.floor(wp.abs(d[1]) / world[1] + 0.5)
    return wp.vec2(x, y)


@wp.func
def heading(angle: float) -> wp.vec2:
    return wp.vec2(wp.cos(angle), wp.sin(angle))


@wp.func
def to_body(v: wp.vec2, angle: float) -> wp.vec2:
    """World vector -> body frame (x = forward, y = left)."""
    c = wp.cos(angle)
    s = wp.sin(angle)
    return wp.vec2(c * v[0] + s * v[1], -s * v[0] + c * v[1])


@wp.func
def from_body(v: wp.vec2, angle: float) -> wp.vec2:
    c = wp.cos(angle)
    s = wp.sin(angle)
    return wp.vec2(c * v[0] - s * v[1], s * v[0] + c * v[1])


@wp.func
def direction_angle(d: wp.vec2) -> float:
    return wp.atan2(d[1], d[0])


@wp.func
def wrap_angle(a: float) -> float:
    """Wrap to [-pi, pi)."""
    return a - 6.283185307179586 * wp.floor((a + 3.141592653589793) / 6.283185307179586)


@wp.func
def sector_of(d_world: wp.vec2, angle: float) -> int:
    """Body-frame sector of a world direction: 0 front, 1 right, 2 back, 3 left."""
    b = to_body(d_world, angle)
    a = wp.atan2(b[1], b[0])          # +: left
    if a >= -0.7853981633974483 and a < 0.7853981633974483:
        return 0
    if a >= 0.7853981633974483 and a < 2.356194490192345:
        return 3
    if a < -0.7853981633974483 and a >= -2.356194490192345:
        return 1
    return 2


@wp.func
def cross2(a: wp.vec2, b: wp.vec2) -> float:
    """z-component of the 3D cross product (torque in 2D)."""
    return a[0] * b[1] - a[1] * b[0]


@wp.func
def perp(d: wp.vec2) -> wp.vec2:
    return wp.vec2(-d[1], d[0])


@wp.func
def disc_area(r: float) -> float:
    return 3.141592653589793 * r * r


@wp.func
def radius_of(mass: float, density: float) -> float:
    return wp.sqrt(mass / (3.141592653589793 * density))


@wp.func
def inertia(mass: float, r: float) -> float:
    return 0.5 * mass * r * r


@wp.func
def grid_coord(p: wp.vec2, h: float, nx: int, ny: int) -> wp.vec2i:
    ix = int(wp.floor(p[0] / h))
    iy = int(wp.floor(p[1] / h))
    ix = ((ix % nx) + nx) % nx
    iy = ((iy % ny) + ny) % ny
    return wp.vec2i(ix, iy)


@wp.func
def wrap_idx(i: int, n: int) -> int:
    return ((i % n) + n) % n


# ----------------------------------------------------------------------------- numpy side
def np_min_image(d, world):
    d = np.asarray(d, dtype=np.float64)
    world = np.asarray(world, dtype=np.float64)
    return d - world * np.sign(d) * np.floor(np.abs(d) / world + 0.5)


def np_wrap_pos(p, world):
    p = np.asarray(p, dtype=np.float64)
    return np.mod(p, world)


def np_heading(angle):
    return np.stack([np.cos(angle), np.sin(angle)], axis=-1)


def np_radius(mass, density):
    return np.sqrt(np.asarray(mass) / (np.pi * density))
