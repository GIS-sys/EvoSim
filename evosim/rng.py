"""Counter-based, stateless RNG.

value = H(key_tick, uid, purpose, counter) where key_tick = H(H(seed, salt), tick) is computed
once per tick on the host. H is a chain of 32-bit integer finalizers (the "lowbias32" mixer),
so the GPU kernels and the numpy reference produce bit-identical streams.

[IMPLEMENTER CHOICE] A hash-chain RNG instead of Philox: it is stateless (no per-cell RNG
state to store or snapshot), deterministic, identical on CPU and GPU, and statistically
adequate for mutation/noise. Documented in DESIGN_NOTES.md.
"""
import numpy as np
import warp as wp

# purposes (distinct streams)
P_VM_RAND = 1
P_SENSOR = 2
P_MUTATE = 3
P_SPLIT = 4
P_BOND = 5
P_EVENT = 6
P_INJECT = 7
P_INIT = 8

M1 = wp.constant(wp.uint32(0x7FEB352D))
M2 = wp.constant(wp.uint32(0x846CA68B))
U16 = wp.constant(wp.uint32(16))
U15 = wp.constant(wp.uint32(15))
U8 = wp.constant(wp.uint32(8))
MASK32 = 0xFFFFFFFF


@wp.func
def mix32(x: wp.uint32) -> wp.uint32:
    h = x
    h = h ^ (h >> U16)
    h = h * M1
    h = h ^ (h >> U15)
    h = h * M2
    h = h ^ (h >> U16)
    return h


@wp.func
def rng_u32(key: wp.uint32, uid: wp.int64, purpose: int, counter: int) -> wp.uint32:
    lo = wp.uint32(uid & wp.int64(0xFFFFFFFF))
    hi = wp.uint32((uid >> wp.int64(32)) & wp.int64(0xFFFFFFFF))
    h = mix32(key ^ lo)
    h = mix32(h ^ hi)
    h = mix32(h ^ (wp.uint32(purpose) << U16) ^ wp.uint32(counter))
    return h


@wp.func
def rng_uniform(key: wp.uint32, uid: wp.int64, purpose: int, counter: int) -> float:
    """Uniform in [0, 1) with 24-bit resolution (exactly representable in float32)."""
    h = rng_u32(key, uid, purpose, counter)
    return float(h >> U8) * 5.9604644775390625e-08


@wp.func
def rng_normal(key: wp.uint32, uid: wp.int64, purpose: int, counter: int) -> float:
    u1 = (float(rng_u32(key, uid, purpose, 2 * counter) >> U8) + 0.5) * 5.9604644775390625e-08
    u2 = float(rng_u32(key, uid, purpose, 2 * counter + 1) >> U8) * 5.9604644775390625e-08
    return wp.sqrt(-2.0 * wp.log(u1)) * wp.cos(6.283185307179586 * u2)


# ----------------------------------------------------------------------------- host side
def np_mix32(x):
    h = np.uint32(x)
    with np.errstate(over="ignore"):
        h = h ^ (h >> np.uint32(16))
        h = np.uint32((int(h) * 0x7FEB352D) & MASK32)
        h = h ^ (h >> np.uint32(15))
        h = np.uint32((int(h) * 0x846CA68B) & MASK32)
        h = h ^ (h >> np.uint32(16))
    return np.uint32(h)


def np_mix32_vec(x):
    h = np.asarray(x, dtype=np.uint64) & MASK32
    h = h ^ (h >> 16)
    h = (h * 0x7FEB352D) & MASK32
    h = h ^ (h >> 15)
    h = (h * 0x846CA68B) & MASK32
    h = h ^ (h >> 16)
    return h.astype(np.uint32)


def run_key(seed: int, salt: int) -> int:
    return int(np_mix32(np_mix32(np.uint32(seed & MASK32) ^ np.uint32(0x9E3779B9)) ^ np.uint32(salt & MASK32)))


def tick_key(seed: int, salt: int, tick: int) -> int:
    k = run_key(seed, salt)
    k = int(np_mix32(np.uint32(k) ^ np.uint32(tick & MASK32)))
    k = int(np_mix32(np.uint32(k) ^ np.uint32((tick >> 32) & MASK32)))
    return k


def np_rng_u32(key, uid, purpose, counter):
    uid = int(uid)
    lo = uid & MASK32
    hi = (uid >> 32) & MASK32
    h = np_mix32(np.uint32(key) ^ np.uint32(lo))
    h = np_mix32(h ^ np.uint32(hi))
    h = np_mix32(h ^ np.uint32(((purpose << 16) & MASK32)) ^ np.uint32(counter & MASK32))
    return int(h)


def np_rng_uniform(key, uid, purpose, counter):
    return np.float32((np_rng_u32(key, uid, purpose, counter) >> 8) * 5.9604644775390625e-08)
