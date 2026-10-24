"""Per-cell genome interpreter. Semantics: evosim/genome/reference_vm.py (normative).

Reads sensors / bond signals from Scratch (filled by the sense phase), writes registers, ip,
state, trait targets and signal_out directly (own state), and queues actions in Scratch.
"""
import warp as wp

from .. import rng
from ..config import SimParams
from ..genome import isa
from ..genome.table import find_label, w_a, w_b, w_c, w_op
from ..state import TRACE_LEN, Cells, Genomes, Scratch, Trace

OP_LABEL, OP_JMP, OP_JGT, OP_JLT, OP_JZ, OP_YIELD, OP_WAIT = (isa.OP_LABEL, isa.OP_JMP, isa.OP_JGT, isa.OP_JLT,
                                                              isa.OP_JZ, isa.OP_YIELD, isa.OP_WAIT)
OP_SET, OP_MOV, OP_ADD, OP_SUB, OP_MUL, OP_DIV = isa.OP_SET, isa.OP_MOV, isa.OP_ADD, isa.OP_SUB, isa.OP_MUL, isa.OP_DIV
OP_MIN, OP_MAX, OP_ABS, OP_NEG, OP_TANH, OP_RAND = isa.OP_MIN, isa.OP_MAX, isa.OP_ABS, isa.OP_NEG, isa.OP_TANH, isa.OP_RAND
OP_SETSTATE, OP_GETSTATE, OP_SENSE, OP_RECV, OP_RECVB = (isa.OP_SETSTATE, isa.OP_GETSTATE, isa.OP_SENSE,
                                                          isa.OP_RECV, isa.OP_RECVB)
OP_THRUST, OP_TURN, OP_SPLIT, OP_SPLITB, OP_BOND, OP_UNBOND, OP_UNBONDALL = (
    isa.OP_THRUST, isa.OP_TURN, isa.OP_SPLIT, isa.OP_SPLITB, isa.OP_BOND, isa.OP_UNBOND, isa.OP_UNBONDALL)
OP_EMIT, OP_SIGNAL, OP_TRAIT, OP_SHARE, OP_EAT, OP_DIE = (isa.OP_EMIT, isa.OP_SIGNAL, isa.OP_TRAIT, isa.OP_SHARE,
                                                          isa.OP_EAT, isa.OP_DIE)
NUM_LABELS = isa.NUM_LABELS
NUM_TRAITS = isa.NUM_TRAITS
JZ_EPS = isa.JZ_EPS
DIV_EPS = isa.DIV_EPS
REG_CLAMP = isa.REG_CLAMP
IMM_SCALE = isa.IMM_SCALE
ACT_THRUST, ACT_TURN, ACT_SPLIT, ACT_SPLITB = isa.ACT_THRUST, isa.ACT_TURN, isa.ACT_SPLIT, isa.ACT_SPLITB
ACT_BOND, ACT_UNBOND, ACT_UNBONDALL, ACT_EMIT = isa.ACT_BOND, isa.ACT_UNBOND, isa.ACT_UNBONDALL, isa.ACT_EMIT
ACT_SIGNAL, ACT_SHARE, ACT_EAT, ACT_DIE, ACT_TRAIT = isa.ACT_SIGNAL, isa.ACT_SHARE, isa.ACT_EAT, isa.ACT_DIE, isa.ACT_TRAIT
ACT_SPLITS = isa.ACT_SPLIT | isa.ACT_SPLITB
ACT_UNBONDS = isa.ACT_UNBOND | isa.ACT_UNBONDALL
FAIL_JMP_NOLABEL = isa.FAIL_JMP_NOLABEL
P_VM_RAND = rng.P_VM_RAND


@wp.func
def clampreg(x: float) -> float:
    if x > REG_CLAMP:
        return REG_CLAMP
    if x < -REG_CLAMP:
        return -REG_CLAMP
    return x


@wp.kernel
def vm_kernel(c: Cells, g: Genomes, s: Scratch, tr: Trace, p: SimParams):
    i = wp.tid()
    if c.alive[i] == 0:
        return
    s.a_thrust[i] = 0.0
    s.a_turn[i] = 0.0
    s.a_split_angle[i] = 0.0
    s.a_split_frac[i] = 0.0
    s.a_split_label[i] = -1
    s.a_bond_sector[i] = -1
    s.a_unbond[i] = -1
    s.a_emit_ch[i] = -1
    s.a_emit_amt[i] = 0.0
    s.a_share[i] = 0.0
    s.a_eat[i] = 0.0
    flags = int(0)
    fail = int(0)
    nexec = int(0)
    sigc = int(0)
    if c.wait[i] > 0:
        c.wait[i] = c.wait[i] - 1
        s.n_exec[i] = 0
        s.a_signal_count[i] = 0
        c.action_flags[i] = 0
        c.fail_flags[i] = 0
        return
    gid = c.genome_id[i]
    L = g.length[gid]
    ip = c.ip[i]
    if ip >= L:
        ip = 0
    uid = c.uid[i]
    key = p.d_key
    ns = p.d_num_sensors
    ks = p.k_sig
    kc = p.k_chem
    mb = p.max_bonds
    rc = int(0)
    tslot = c.traced[i]
    for _n in range(p.instr_per_tick):
        w = g.words[gid, ip]
        op = w_op(w)
        a = w_a(w)
        b = w_b(w)
        cc = w_c(w)
        nip = ip + 1
        if nip >= L:
            nip = 0
        nexec = nexec + 1
        stop = int(0)
        ra = a & 7
        rb = b & 7
        rcc = cc & 7
        if op == OP_JMP or op == OP_JGT or op == OP_JLT or op == OP_JZ:
            take = int(1)
            lab = a
            if op == OP_JGT:
                take = int(0)
                lab = cc
                if c.regs[i, ra] > c.regs[i, rb]:
                    take = int(1)
            elif op == OP_JLT:
                take = int(0)
                lab = cc
                if c.regs[i, ra] < c.regs[i, rb]:
                    take = int(1)
            elif op == OP_JZ:
                take = int(0)
                lab = b
                if wp.abs(c.regs[i, ra]) < JZ_EPS:
                    take = int(1)
            if take != 0:
                t = find_label(g, gid, lab % NUM_LABELS, ip, L)
                if t < 0:
                    fail = fail | FAIL_JMP_NOLABEL
                else:
                    nip = t
        elif op == OP_YIELD:
            stop = int(1)
        elif op == OP_WAIT:
            v = c.regs[i, ra]
            if v < 0.0:
                v = 0.0
            mw = float(p.max_wait)
            if v > mw:
                v = mw
            c.wait[i] = int(wp.floor(v + 0.5))
            stop = int(1)
        elif op == OP_SET:
            imm = (b << 8) | cc
            if imm >= 32768:
                imm = imm - 65536
            c.regs[i, ra] = clampreg(float(imm) / IMM_SCALE)
        elif op == OP_MOV:
            c.regs[i, ra] = c.regs[i, rb]
        elif op == OP_ADD:
            c.regs[i, ra] = clampreg(c.regs[i, rb] + c.regs[i, rcc])
        elif op == OP_SUB:
            c.regs[i, ra] = clampreg(c.regs[i, rb] - c.regs[i, rcc])
        elif op == OP_MUL:
            c.regs[i, ra] = clampreg(c.regs[i, rb] * c.regs[i, rcc])
        elif op == OP_DIV:
            dd = c.regs[i, rcc]
            if wp.abs(dd) < DIV_EPS:
                c.regs[i, ra] = 0.0
            else:
                c.regs[i, ra] = clampreg(c.regs[i, rb] / dd)
        elif op == OP_MIN:
            x = c.regs[i, rb]
            y = c.regs[i, rcc]
            if x < y:
                c.regs[i, ra] = x
            else:
                c.regs[i, ra] = y
        elif op == OP_MAX:
            x = c.regs[i, rb]
            y = c.regs[i, rcc]
            if x > y:
                c.regs[i, ra] = x
            else:
                c.regs[i, ra] = y
        elif op == OP_ABS:
            c.regs[i, ra] = wp.abs(c.regs[i, rb])
        elif op == OP_NEG:
            c.regs[i, ra] = 0.0 - c.regs[i, rb]   # 0-x: IEEE-defined signed zero (+0)
        elif op == OP_TANH:
            x = c.regs[i, rb]
            c.regs[i, ra] = x / (1.0 + wp.abs(x))
        elif op == OP_RAND:
            c.regs[i, ra] = rng.rng_uniform(key, uid, P_VM_RAND, rc)
            rc = rc + 1
        elif op == OP_SETSTATE:
            c.state[i] = a
        elif op == OP_GETSTATE:
            c.regs[i, ra] = float(c.state[i])
        elif op == OP_SENSE:
            c.regs[i, ra] = clampreg(s.sensors[i, b % ns])
        elif op == OP_RECV:
            c.regs[i, ra] = clampreg(s.recv_agg[i, b % ks])
        elif op == OP_RECVB:
            c.regs[i, ra] = clampreg(s.recv_slot[i, (cc % mb) * ks + (b % ks)])
        elif op == OP_THRUST:
            s.a_thrust[i] = c.regs[i, ra]
            flags = flags | ACT_THRUST
        elif op == OP_TURN:
            s.a_turn[i] = c.regs[i, ra]
            flags = flags | ACT_TURN
        elif op == OP_SPLIT or op == OP_SPLITB:
            s.a_split_angle[i] = c.regs[i, ra]
            s.a_split_frac[i] = c.regs[i, rb]
            s.a_split_label[i] = cc % NUM_LABELS
            flags = flags & (~ACT_SPLITS)
            if op == OP_SPLITB:
                flags = flags | ACT_SPLITB
            else:
                flags = flags | ACT_SPLIT
        elif op == OP_BOND:
            s.a_bond_sector[i] = a % 4
            flags = flags | ACT_BOND
        elif op == OP_UNBOND:
            s.a_unbond[i] = a % mb
            flags = (flags & (~ACT_UNBONDS)) | ACT_UNBOND
        elif op == OP_UNBONDALL:
            s.a_unbond[i] = -2
            flags = (flags & (~ACT_UNBONDS)) | ACT_UNBONDALL
        elif op == OP_EMIT:
            s.a_emit_ch[i] = a % kc
            s.a_emit_amt[i] = c.regs[i, rb]
            flags = flags | ACT_EMIT
        elif op == OP_SIGNAL:
            c.signal_out[i, a % ks] = c.regs[i, rb]
            sigc = sigc + 1
            flags = flags | ACT_SIGNAL
        elif op == OP_TRAIT:
            v = c.regs[i, rb]
            if v < 0.0:
                v = 0.0
            if v > 1.0:
                v = 1.0
            c.trait_targets[i, a % NUM_TRAITS] = v
            flags = flags | ACT_TRAIT
        elif op == OP_SHARE:
            s.a_share[i] = c.regs[i, ra]
            flags = flags | ACT_SHARE
        elif op == OP_EAT:
            s.a_eat[i] = c.regs[i, ra]
            flags = flags | ACT_EAT
        elif op == OP_DIE:
            flags = flags | ACT_DIE
        # trace & breakpoints (debug cells only)
        if tslot >= 0:
            h = wp.atomic_add(tr.head, tslot, 1) % TRACE_LEN
            tr.tick[tslot, h] = p.d_tick
            tr.ip[tslot, h] = ip
            tr.word[tslot, h] = w
            for r in range(8):
                tr.regs[tslot, h, r] = c.regs[i, r]
            if tr.bp_op[tslot] == op or tr.bp_addr[tslot] == ip:
                tr.hit[tslot] = 1
        ip = nip
        if stop != 0:
            break
    c.ip[i] = ip
    s.n_exec[i] = nexec
    s.a_signal_count[i] = sigc
    c.action_flags[i] = flags
    c.fail_flags[i] = fail
