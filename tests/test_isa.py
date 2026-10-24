import numpy as np

from evosim.genome import assembler, isa


def test_every_word_decodes_to_valid_instruction():
    rs = np.random.default_rng(0)
    words = rs.integers(0, 2**32, size=200000, dtype=np.uint64)
    for w in words[:20000]:
        op, a, b, c = isa.decode(int(w))
        assert 0 <= op < isa.NUM_OPCODES
        assert 0 <= a < 256 and 0 <= b < 256 and 0 <= c < 256
        assembler.disassemble_word(int(w))        # must never raise
    # the numpy-free exhaustive check on the opcode byte
    for byte in range(256):
        assert 0 <= isa.decode(byte << 24)[0] < isa.NUM_OPCODES


def test_assemble_disassemble_roundtrip_exact():
    rs = np.random.default_rng(1)
    for _ in range(300):
        n = int(rs.integers(1, 64))
        w = [int(x) for x in rs.integers(0, 2**32, size=n, dtype=np.uint64)]
        txt = assembler.disassemble(w, exact=True)
        assert assembler.assemble(txt) == w


def test_canonical_roundtrip_is_human_readable():
    src = "LABEL 0\n  SET r0 1.0\n  TRAIT PHOTO r0\n  SENSE r1 ENERGY_FRAC\n  JLT r1 r2 2\n  BOND F\n  EMIT C1 r3\n"
    w = assembler.assemble(src)
    txt = assembler.disassemble(w)
    assert assembler.assemble(txt) == w
    assert "PHOTO" in txt and "ENERGY_FRAC" in txt and "BOND F" in txt
    assert "!" not in txt            # canonical words need no exact marker


def test_set_immediate_encoding():
    for v in (0.0, 1.0, -1.0, 0.5, 100.0, -511.0, 3.140625):
        b, c = isa.imm_to_bytes(v)
        assert abs(isa.imm_from_bytes(b, c) - v) <= 1 / 128
    assert isa.imm_from_bytes(0x80, 0x00) == -512.0


def test_binary_format(tmp_path):
    w = [0xDEADBEEF, 1, 2, 0xFFFFFFFF]
    p = tmp_path / "g.gnm"
    assembler.save_binary(p, w)
    assert assembler.load_binary(p) == w


def test_seed_files_assemble_and_roundtrip():
    import glob
    for f in glob.glob("seeds/*.asm"):
        w = assembler.load_asm(f)
        assert w and assembler.assemble(assembler.disassemble(w, exact=True)) == w
