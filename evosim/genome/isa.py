"""Instruction set: single source of truth for opcodes, operand roles, sensors, traits, flags.

Word layout: opcode:8 | a:8 | b:8 | c:8   (opcode in the most significant byte).
Every 32-bit word is a valid instruction: opcode = byte % NUM_OPCODES, registers = x & 7,
labels = x % NUM_LABELS, ids = x % count.
"""
from __future__ import annotations

MAX_GENOME_LEN = 256      # compile-time
NUM_REGS = 8
NUM_LABELS = 32
NUM_TRAITS = 9
NUM_SECTORS = 4

JZ_EPS = 1e-4             # JZ: |R[a]| < JZ_EPS
DIV_EPS = 1e-6            # DIV: |R[c]| < DIV_EPS -> result 0
REG_CLAMP = 1.0e6         # every register write is clamped to [-REG_CLAMP, REG_CLAMP] (no inf/NaN ever)
IMM_SCALE = 64.0          # SET immediate = int16 / 64

# ----------------------------------------------------------------------------- opcodes
# (mnemonic, operand roles for a, b, c)
# roles: r=register, l=label, imm=int16 immediate in b:c, s=sensor, t=trait, ch=chemical channel,
#        sig=signal channel, slot=bond slot, sec=contact sector, byte=raw 0..255
OPCODES = [
    ("NOP", ()),
    ("LABEL", ("l",)),
    ("JMP", ("l",)),
    ("JGT", ("r", "r", "l")),
    ("JLT", ("r", "r", "l")),
    ("JZ", ("r", "l")),
    ("YIELD", ()),
    ("WAIT", ("r",)),
    ("SET", ("r", "imm")),
    ("MOV", ("r", "r")),
    ("ADD", ("r", "r", "r")),
    ("SUB", ("r", "r", "r")),
    ("MUL", ("r", "r", "r")),
    ("DIV", ("r", "r", "r")),
    ("MIN", ("r", "r", "r")),
    ("MAX", ("r", "r", "r")),
    ("ABS", ("r", "r")),
    ("NEG", ("r", "r")),
    ("TANH", ("r", "r")),
    ("RAND", ("r",)),
    ("SETSTATE", ("byte",)),
    ("GETSTATE", ("r",)),
    ("SENSE", ("r", "s")),
    ("RECV", ("r", "sig")),
    ("RECVB", ("r", "sig", "slot")),
    ("THRUST", ("r",)),
    ("TURN", ("r",)),
    ("SPLIT", ("r", "r", "l")),
    ("SPLITB", ("r", "r", "l")),
    ("BOND", ("sec",)),
    ("UNBOND", ("slot",)),
    ("UNBONDALL", ()),
    ("EMIT", ("ch", "r")),
    ("SIGNAL", ("sig", "r")),
    ("TRAIT", ("t", "r")),
    ("SHARE", ("r",)),
    ("EAT", ("r",)),
    ("DIE", ()),
]
NUM_OPCODES = len(OPCODES)
MNEMONICS = [m for m, _ in OPCODES]
OP = {m: i for i, m in enumerate(MNEMONICS)}

# python-level constants for kernels (warp reads module globals as compile-time constants)
OP_NOP, OP_LABEL, OP_JMP, OP_JGT, OP_JLT, OP_JZ, OP_YIELD, OP_WAIT = range(8)
(OP_SET, OP_MOV, OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MIN, OP_MAX, OP_ABS, OP_NEG, OP_TANH, OP_RAND,
 OP_SETSTATE, OP_GETSTATE) = range(8, 22)
OP_SENSE, OP_RECV, OP_RECVB = 22, 23, 24
OP_THRUST, OP_TURN, OP_SPLIT, OP_SPLITB, OP_BOND, OP_UNBOND, OP_UNBONDALL = range(25, 32)
OP_EMIT, OP_SIGNAL, OP_TRAIT, OP_SHARE, OP_EAT, OP_DIE = range(32, 38)
assert MNEMONICS[OP_DIE] == "DIE" and MNEMONICS[OP_SENSE] == "SENSE" and MNEMONICS[OP_EMIT] == "EMIT"

# ----------------------------------------------------------------------------- traits
TRAIT_NAMES = ("PHOTO", "ABSORB", "MOTOR", "PREDATION", "ARMOR", "PIGMENT", "PHOTORECEPTOR", "ADHESION",
               "RECEPTOR_WIDTH")
T_PHOTO, T_ABSORB, T_MOTOR, T_PRED, T_ARMOR, T_PIGMENT, T_PHOTORECEPTOR, T_ADHESION, T_RWIDTH = range(9)
assert len(TRAIT_NAMES) == NUM_TRAITS

# ----------------------------------------------------------------------------- sensors
BASE_SENSORS = ("ENERGY", "ENERGY_FRAC", "MASS", "RADIUS", "AGE", "SPEED", "ANG_VEL", "VEL_FWD", "VEL_SIDE",
                "LIGHT", "LIGHT_DELTA", "NUTRIENT", "TOUCH_TOTAL", "TOUCH_F", "TOUCH_R", "TOUCH_B", "TOUCH_L",
                "DAMAGE", "BOND_COUNT", "BOND_STRAIN", "RANDOM")
(S_ENERGY, S_ENERGY_FRAC, S_MASS, S_RADIUS, S_AGE, S_SPEED, S_ANG_VEL, S_VEL_FWD, S_VEL_SIDE, S_LIGHT,
 S_LIGHT_DELTA, S_NUTRIENT, S_TOUCH_TOTAL, S_TOUCH_F, S_TOUCH_R, S_TOUCH_B, S_TOUCH_L, S_DAMAGE, S_BOND_COUNT,
 S_BOND_STRAIN, S_RANDOM) = range(21)
NUM_BASE_SENSORS = len(BASE_SENSORS)
AGE_SCALE = 1.0 / 1000.0     # AGE sensor = age * AGE_SCALE


def sensor_names(k_chem: int) -> list[str]:
    return list(BASE_SENSORS) + [f"CHEM_{k}" for k in range(k_chem)] + [f"TRAIT_{t}" for t in TRAIT_NAMES]


def num_sensors(k_chem: int) -> int:
    return NUM_BASE_SENSORS + k_chem + NUM_TRAITS


def sensor_chem(k: int) -> int:
    return NUM_BASE_SENSORS + k


def sensor_trait(k_chem: int, t: int) -> int:
    return NUM_BASE_SENSORS + k_chem + t


SECTOR_NAMES = ("F", "R", "B", "L")

# ----------------------------------------------------------------------------- action / fail flags
ACT_THRUST, ACT_TURN, ACT_SPLIT, ACT_SPLITB, ACT_BOND, ACT_UNBOND, ACT_UNBONDALL, ACT_EMIT, ACT_SIGNAL, \
    ACT_SHARE, ACT_EAT, ACT_DIE, ACT_TRAIT, ACT_SPLIT_OK, ACT_BOND_OK, ACT_BITTEN, ACT_BORN = \
    [1 << i for i in range(17)]
ACTION_NAMES = {ACT_THRUST: "THRUST", ACT_TURN: "TURN", ACT_SPLIT: "SPLIT", ACT_SPLITB: "SPLITB",
                ACT_BOND: "BOND", ACT_UNBOND: "UNBOND", ACT_UNBONDALL: "UNBONDALL", ACT_EMIT: "EMIT",
                ACT_SIGNAL: "SIGNAL", ACT_SHARE: "SHARE", ACT_EAT: "EAT", ACT_DIE: "DIE", ACT_TRAIT: "TRAIT",
                ACT_SPLIT_OK: "split succeeded", ACT_BOND_OK: "bond created", ACT_BITTEN: "was bitten",
                ACT_BORN: "born this tick"}

FAIL_NAMES = [
    ("FAIL_JMP_NOLABEL", "JMP: no such label"),
    ("FAIL_SPLIT_MASS", "SPLIT failed: insufficient mass"),
    ("FAIL_SPLIT_ENERGY", "SPLIT failed: insufficient energy"),
    ("FAIL_CAPACITY", "SPLIT failed: cell capacity full"),
    ("FAIL_SPLIT_NOSLOT", "SPLITB failed: no free bond slot"),
    ("FAIL_SPLIT_NOLABEL", "SPLIT: daughter label missing (started at 0)"),
    ("FAIL_BOND_NOCONTACT", "BOND failed: no contact in sector"),
    ("FAIL_BOND_NOSLOT", "BOND failed: no free slot"),
    ("FAIL_BOND_CONSENT", "BOND failed: no consent"),
    ("FAIL_BOND_ENERGY", "BOND failed: insufficient energy"),
    ("FAIL_BOND_CONFLICT", "BOND failed: lost conflict this tick"),
    ("FAIL_UNBOND_EMPTY", "UNBOND: slot empty"),
    ("FAIL_THRUST_ENERGY", "THRUST scaled down: insufficient energy"),
    ("FAIL_TURN_ENERGY", "TURN scaled down: insufficient energy"),
    ("FAIL_EMIT_ENERGY", "EMIT scaled down: insufficient energy"),
    ("FAIL_SIGNAL_ENERGY", "SIGNAL dropped: insufficient energy"),
    ("FAIL_SHARE_NOBOND", "SHARE: no bonds"),
    ("FAIL_EAT_NOTRAIT", "EAT: PREDATION trait is 0 or predation disabled"),
    ("FAIL_TRAIT_ENERGY", "trait change scaled down: insufficient energy"),
    ("FAIL_BOND_ALREADY", "BOND: already bonded to that cell"),
    ("FAIL_EAT_NOCONTACT", "EAT: nothing edible touching"),
]
for _i, (_n, _d) in enumerate(FAIL_NAMES):
    globals()[_n] = 1 << _i
FAIL_DESCRIPTIONS = {1 << i: d for i, (_, d) in enumerate(FAIL_NAMES)}

ORIGIN_NAMES = ("natural", "injected", "cloned", "restored")
ORIGIN_NATURAL, ORIGIN_INJECTED, ORIGIN_CLONED, ORIGIN_RESTORED = range(4)


def describe_flags(action_flags: int, fail_flags: int) -> list[str]:
    out = []
    for bit, name in ACTION_NAMES.items():
        if action_flags & bit:
            out.append(name)
    for bit, d in FAIL_DESCRIPTIONS.items():
        if fail_flags & bit:
            out.append(d)
    return out


# ----------------------------------------------------------------------------- encode / decode
def encode(op: int, a: int = 0, b: int = 0, c: int = 0) -> int:
    return ((op & 0xFF) << 24) | ((a & 0xFF) << 16) | ((b & 0xFF) << 8) | (c & 0xFF)


def decode(word: int):
    word &= 0xFFFFFFFF
    return (word >> 24) % NUM_OPCODES, (word >> 16) & 0xFF, (word >> 8) & 0xFF, word & 0xFF


def imm_from_bytes(b: int, c: int) -> float:
    v = (b << 8) | c
    if v >= 0x8000:
        v -= 0x10000
    return v / IMM_SCALE


def imm_to_bytes(x: float):
    v = int(round(x * IMM_SCALE))
    v = max(-32768, min(32767, v))
    v &= 0xFFFF
    return (v >> 8) & 0xFF, v & 0xFF


def role_count(role: str, k_chem: int, k_sig: int, max_bonds: int) -> int | None:
    return {"r": NUM_REGS, "l": NUM_LABELS, "s": num_sensors(k_chem), "t": NUM_TRAITS, "ch": k_chem,
            "sig": k_sig, "slot": max_bonds, "sec": NUM_SECTORS, "byte": 256}.get(role)


def to_signed32(w: int) -> int:
    w &= 0xFFFFFFFF
    return w - (1 << 32) if w >= (1 << 31) else w


def to_unsigned32(w: int) -> int:
    return int(w) & 0xFFFFFFFF
