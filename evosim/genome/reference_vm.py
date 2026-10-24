"""CPU reference interpreter. Semantics here are normative; kernels/vm.py must match bit-for-bit.

Float discipline: every register value is a numpy float32 and every operation is a single
IEEE float32 operation (no fused multiply-add opportunities), so CPU and GPU agree exactly.
TANH is implemented as softsign x/(1+|x|) for that reason (see DESIGN_NOTES.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import rng
from . import isa

F32 = np.float32
CLAMP = F32(isa.REG_CLAMP)


def clampreg(x) -> np.float32:
    x = F32(x)
    if x > CLAMP:
        return CLAMP
    if x < -CLAMP:
        return -CLAMP
    return x


def build_label_table(words):
    """Returns (offsets[NUM_LABELS+1], positions[]) : positions of LABEL k sorted ascending."""
    buckets = [[] for _ in range(isa.NUM_LABELS)]
    for i, w in enumerate(words):
        op, a, _, _ = isa.decode(int(w))
        if op == isa.OP_LABEL:
            buckets[a % isa.NUM_LABELS].append(i)
    offs = [0]
    pos = []
    for b in buckets:
        pos.extend(b)
        offs.append(len(pos))
    return offs, pos


def find_label(offs, pos, k, ip, length):
    """Position after the next LABEL k strictly after ip (wrapping); -1 if none."""
    s, e = offs[k], offs[k + 1]
    if s == e:
        return -1
    target = pos[s]
    for j in range(s, e):
        if pos[j] > ip:
            target = pos[j]
            break
    t = target + 1
    return 0 if t >= length else t


def first_label(offs, pos, k, length):
    s, e = offs[k], offs[k + 1]
    if s == e:
        return -1
    t = pos[s] + 1
    return 0 if t >= length else t


@dataclass
class Actions:
    flags: int = 0
    fail: int = 0
    thrust: np.float32 = F32(0)
    turn: np.float32 = F32(0)
    split_angle: np.float32 = F32(0)
    split_frac: np.float32 = F32(0)
    split_label: int = 0
    split_bonded: int = 0
    bond_sector: int = -1
    unbond: int = -1          # slot, or -2 = all
    emit_ch: int = -1
    emit_amt: np.float32 = F32(0)
    share: np.float32 = F32(0)
    eat: np.float32 = F32(0)
    signal_count: int = 0
    n_exec: int = 0


@dataclass
class CellVM:
    words: list
    regs: np.ndarray = field(default_factory=lambda: np.zeros(isa.NUM_REGS, np.float32))
    ip: int = 0
    wait: int = 0
    state: int = 0
    trait_targets: np.ndarray = field(default_factory=lambda: np.zeros(isa.NUM_TRAITS, np.float32))
    signal_out: np.ndarray = None
    uid: int = 0

    def __post_init__(self):
        self.words = [int(w) & 0xFFFFFFFF for w in self.words]
        self.offs, self.pos = build_label_table(self.words)


@dataclass
class VMInputs:
    sensors: np.ndarray          # float32[num_sensors]
    recv_agg: np.ndarray         # float32[k_sig]
    recv_slot: np.ndarray        # float32[max_bonds, k_sig]
    key: int                     # tick key


def run_tick(cell: CellVM, inp: VMInputs, instr_per_tick=16, max_wait=64, k_chem=4, k_sig=4, max_bonds=6,
             trace=None, tick=0) -> Actions:
    act = Actions()
    if cell.signal_out is None:
        cell.signal_out = np.zeros(k_sig, np.float32)
    if cell.wait > 0:
        cell.wait -= 1
        return act
    R = cell.regs
    L = len(cell.words)
    ns = isa.num_sensors(k_chem)
    rand_counter = 0
    ip = cell.ip
    for _ in range(instr_per_tick):
        w = cell.words[ip]
        op, a, b, c = isa.decode(w)
        nip = ip + 1
        if nip >= L:
            nip = 0
        act.n_exec += 1
        stop = False
        ra = a & 7
        rb = b & 7
        rc = c & 7
        before = R.copy() if trace is not None else None
        if op == isa.OP_JMP or op == isa.OP_JGT or op == isa.OP_JLT or op == isa.OP_JZ:
            take = True
            lab = a
            if op == isa.OP_JGT:
                take = bool(R[ra] > R[rb])
                lab = c
            elif op == isa.OP_JLT:
                take = bool(R[ra] < R[rb])
                lab = c
            elif op == isa.OP_JZ:
                take = bool(abs(R[ra]) < F32(isa.JZ_EPS))
                lab = b
            if take:
                t = find_label(cell.offs, cell.pos, lab % isa.NUM_LABELS, ip, L)
                if t < 0:
                    act.fail |= isa.FAIL_JMP_NOLABEL
                else:
                    nip = t
        elif op == isa.OP_YIELD:
            stop = True
        elif op == isa.OP_WAIT:
            v = R[ra]
            if v < F32(0):
                v = F32(0)
            mw = F32(max_wait)
            if v > mw:
                v = mw
            cell.wait = int(np.floor(F32(v + F32(0.5))))
            stop = True
        elif op == isa.OP_SET:
            R[ra] = clampreg(F32(isa.imm_from_bytes(b, c)))
        elif op == isa.OP_MOV:
            R[ra] = R[rb]
        elif op == isa.OP_ADD:
            R[ra] = clampreg(R[rb] + R[rc])
        elif op == isa.OP_SUB:
            R[ra] = clampreg(R[rb] - R[rc])
        elif op == isa.OP_MUL:
            R[ra] = clampreg(R[rb] * R[rc])
        elif op == isa.OP_DIV:
            d = R[rc]
            if abs(d) < F32(isa.DIV_EPS):
                R[ra] = F32(0)
            else:
                R[ra] = clampreg(R[rb] / d)
        elif op == isa.OP_MIN:
            R[ra] = R[rb] if R[rb] < R[rc] else R[rc]
        elif op == isa.OP_MAX:
            R[ra] = R[rb] if R[rb] > R[rc] else R[rc]
        elif op == isa.OP_ABS:
            R[ra] = abs(R[rb])
        elif op == isa.OP_NEG:
            R[ra] = F32(0) - R[rb]   # 0-x, never -0.0 (matches GPU)
        elif op == isa.OP_TANH:
            x = R[rb]
            R[ra] = x / (F32(1) + abs(x))
        elif op == isa.OP_RAND:
            R[ra] = rng.np_rng_uniform(inp.key, cell.uid, rng.P_VM_RAND, rand_counter)
            rand_counter += 1
        elif op == isa.OP_SETSTATE:
            cell.state = a
        elif op == isa.OP_GETSTATE:
            R[ra] = F32(cell.state)
        elif op == isa.OP_SENSE:
            R[ra] = clampreg(inp.sensors[b % ns])
        elif op == isa.OP_RECV:
            R[ra] = clampreg(inp.recv_agg[b % k_sig])
        elif op == isa.OP_RECVB:
            R[ra] = clampreg(inp.recv_slot[c % max_bonds, b % k_sig])
        elif op == isa.OP_THRUST:
            act.thrust = R[ra]
            act.flags |= isa.ACT_THRUST
        elif op == isa.OP_TURN:
            act.turn = R[ra]
            act.flags |= isa.ACT_TURN
        elif op == isa.OP_SPLIT or op == isa.OP_SPLITB:
            act.split_angle = R[ra]
            act.split_frac = R[rb]
            act.split_label = c % isa.NUM_LABELS
            act.split_bonded = 1 if op == isa.OP_SPLITB else 0
            act.flags &= ~(isa.ACT_SPLIT | isa.ACT_SPLITB)
            act.flags |= isa.ACT_SPLITB if op == isa.OP_SPLITB else isa.ACT_SPLIT
        elif op == isa.OP_BOND:
            act.bond_sector = a % 4
            act.flags |= isa.ACT_BOND
        elif op == isa.OP_UNBOND:
            act.unbond = a % max_bonds
            act.flags = (act.flags & ~isa.ACT_UNBONDALL) | isa.ACT_UNBOND
        elif op == isa.OP_UNBONDALL:
            act.unbond = -2
            act.flags = (act.flags & ~isa.ACT_UNBOND) | isa.ACT_UNBONDALL
        elif op == isa.OP_EMIT:
            act.emit_ch = a % k_chem
            act.emit_amt = R[rb]
            act.flags |= isa.ACT_EMIT
        elif op == isa.OP_SIGNAL:
            cell.signal_out[a % k_sig] = R[rb]
            act.signal_count += 1
            act.flags |= isa.ACT_SIGNAL
        elif op == isa.OP_TRAIT:
            v = R[rb]
            if v < F32(0):
                v = F32(0)
            if v > F32(1):
                v = F32(1)
            cell.trait_targets[a % isa.NUM_TRAITS] = v
            act.flags |= isa.ACT_TRAIT
        elif op == isa.OP_SHARE:
            act.share = R[ra]
            act.flags |= isa.ACT_SHARE
        elif op == isa.OP_EAT:
            act.eat = R[ra]
            act.flags |= isa.ACT_EAT
        elif op == isa.OP_DIE:
            act.flags |= isa.ACT_DIE
        # NOP / LABEL: nothing
        if trace is not None:
            trace.append((tick, ip, w, before, R.copy()))
        ip = nip
        if stop:
            break
    cell.ip = ip
    return act
