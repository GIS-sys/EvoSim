"""Assembler / disassembler.

Text format (one instruction per line, ';' starts a comment, case-insensitive mnemonics):
    LABEL 0
    SET r0 1.0
    TRAIT PHOTO r0
    SENSE r1 ENERGY_FRAC
    JLT r1 r2 2
    BOND F            ; sectors F R B L (or 0..3)
    EMIT C1 r3        ; chemical channel C0..C7 (or plain int)
    SIGNAL S0 r2      ; signal channel S0..S7 (or plain int)
    RECVB r1 S0 B2    ; bond slot B0..B11 (or plain int)
    SETSTATE 3
    .word 0x1b030000  ; raw word

Exact round trip: disassemble(words, exact=True) appends '!0x........' to every line whose raw
word differs from the canonical encoding (e.g. opcode byte >= NUM_OPCODES, register byte > 7),
and the assembler honours it, so assemble(disassemble(w, exact=True)) == w for every word list.

Binary format (.gnm): b"EVGN" | u32 version | u32 count | count * u32 little-endian words.
"""
from __future__ import annotations

import re
import struct

from . import isa


class AsmError(ValueError):
    pass


def _ctx_counts(k_chem=4, k_sig=4, max_bonds=6):
    return dict(k_chem=k_chem, k_sig=k_sig, max_bonds=max_bonds)


def _parse_operand(role, tok, sensors, line_no):
    t = tok.strip()
    tu = t.upper()
    try:
        if role == "r":
            if not tu.startswith("R"):
                raise ValueError
            return int(tu[1:])
        if role in ("l", "byte"):
            return int(t, 0)
        if role == "s":
            if tu in sensors:
                return sensors.index(tu)
            return int(t, 0)
        if role == "t":
            if tu in isa.TRAIT_NAMES:
                return isa.TRAIT_NAMES.index(tu)
            return int(t, 0)
        if role == "ch":
            return int(tu[1:]) if tu.startswith("C") and tu[1:].isdigit() else int(t, 0)
        if role == "sig":
            return int(tu[1:]) if tu.startswith("S") and tu[1:].isdigit() else int(t, 0)
        if role == "slot":
            return int(tu[1:]) if tu.startswith("B") and tu[1:].isdigit() else int(t, 0)
        if role == "sec":
            if tu in isa.SECTOR_NAMES:
                return isa.SECTOR_NAMES.index(tu)
            return int(t, 0)
    except ValueError:
        pass
    raise AsmError(f"line {line_no}: bad operand {tok!r} for role {role!r}")


def assemble(text: str, k_chem=4, k_sig=4, max_bonds=6) -> list[int]:
    sensors = isa.sensor_names(k_chem)
    words: list[int] = []
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        exact = None
        m = re.search(r"!\s*(0x[0-9a-fA-F]+)\s*$", line)
        if m:
            exact = int(m.group(1), 16)
            line = line[: m.start()].strip()
        toks = line.replace(",", " ").split()
        mnem = toks[0].upper()
        if mnem == ".WORD":
            if len(toks) != 2:
                raise AsmError(f"line {line_no}: .word needs one value")
            words.append(int(toks[1], 0) & 0xFFFFFFFF)
            continue
        if mnem not in isa.OP:
            raise AsmError(f"line {line_no}: unknown mnemonic {toks[0]!r}")
        op = isa.OP[mnem]
        roles = isa.OPCODES[op][1]
        args = toks[1:]
        if len(args) != len(roles):
            raise AsmError(f"line {line_no}: {mnem} expects {len(roles)} operands, got {len(args)}")
        fields = [0, 0, 0]
        for i, (role, tok) in enumerate(zip(roles, args)):
            if role == "imm":
                try:
                    v = float(tok)
                except ValueError:
                    raise AsmError(f"line {line_no}: bad immediate {tok!r}") from None
                b, c = isa.imm_to_bytes(v)
                fields[1], fields[2] = b, c
            else:
                fields[i] = _parse_operand(role, tok, sensors, line_no) & 0xFF
        w = isa.encode(op, *fields)
        if exact is not None:
            if isa.decode(exact)[0] != op:
                raise AsmError(f"line {line_no}: raw word {exact:#010x} does not decode to {mnem}")
            w = exact
        words.append(w)
    return words


def _fmt_operand(role, v, sensors, k_chem, k_sig, max_bonds):
    if role == "r":
        return f"r{v & 7}"
    if role == "l":
        return str(v % isa.NUM_LABELS)
    if role == "byte":
        return str(v)
    if role == "s":
        return sensors[v % len(sensors)]
    if role == "t":
        return isa.TRAIT_NAMES[v % isa.NUM_TRAITS]
    if role == "ch":
        return f"C{v % k_chem}"
    if role == "sig":
        return f"S{v % k_sig}"
    if role == "slot":
        return f"B{v % max_bonds}"
    if role == "sec":
        return isa.SECTOR_NAMES[v % 4]
    raise ValueError(role)


def disassemble_word(word: int, k_chem=4, k_sig=4, max_bonds=6, exact=False) -> str:
    sensors = isa.sensor_names(k_chem)
    op, a, b, c = isa.decode(word)
    mnem, roles = isa.OPCODES[op]
    fields = (a, b, c)
    parts = [mnem]
    canon = [0, 0, 0]
    for i, role in enumerate(roles):
        if role == "imm":
            v = isa.imm_from_bytes(b, c)
            parts.append(repr(v) if v != int(v) else f"{v:.1f}")
            canon[1], canon[2] = b, c
        else:
            v = fields[i]
            parts.append(_fmt_operand(role, v, sensors, k_chem, k_sig, max_bonds))
            n = isa.role_count(role, k_chem, k_sig, max_bonds)
            canon[i] = (v & 7) if role == "r" else (v % n)
    s = " ".join(parts)
    if exact:
        if isa.encode(op, *canon) != (word & 0xFFFFFFFF):
            s += f" !{word & 0xFFFFFFFF:#010x}"
    return s


def disassemble(words, k_chem=4, k_sig=4, max_bonds=6, exact=False, addresses=False, ip=None) -> str:
    lines = []
    for i, w in enumerate(words):
        txt = disassemble_word(int(w), k_chem, k_sig, max_bonds, exact)
        if isa.decode(int(w))[0] != isa.OP_LABEL:
            txt = "  " + txt
        if addresses:
            mark = ">" if ip == i else " "
            txt = f"{mark}{i:3d}: {txt}"
        lines.append(txt)
    return "\n".join(lines) + ("\n" if lines else "")


# ----------------------------------------------------------------------------- files
def load_asm(path, **counts) -> list[int]:
    with open(path, encoding="utf-8") as f:
        return assemble(f.read(), **counts)


def save_asm(path, words, header="", **counts):
    with open(path, "w", encoding="utf-8") as f:
        if header:
            for line in header.splitlines():
                f.write(f"; {line}\n")
        f.write(disassemble(words, exact=True, **counts))


def save_binary(path, words):
    with open(path, "wb") as f:
        f.write(b"EVGN" + struct.pack("<II", 1, len(words)))
        f.write(struct.pack(f"<{len(words)}I", *[int(w) & 0xFFFFFFFF for w in words]))


def load_binary(path) -> list[int]:
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"EVGN":
        raise AsmError("not an EvoSim genome file")
    _, n = struct.unpack("<II", data[4:12])
    return list(struct.unpack(f"<{n}I", data[12:12 + 4 * n]))


def load_genome(path, **counts) -> list[int]:
    return load_binary(path) if str(path).endswith(".gnm") else load_asm(path, **counts)


def genome_hash(words) -> int:
    """64-bit FNV-1a over (length, words). Must match genome/table.py's GPU version."""
    h = 0xCBF29CE484222325
    prime = 0x100000001B3
    mask = (1 << 64) - 1
    seq = [len(words)] + [int(w) & 0xFFFFFFFF for w in words]
    for w in seq:
        for k in range(4):
            h ^= (w >> (8 * k)) & 0xFF
            h = (h * prime) & mask
    return h
