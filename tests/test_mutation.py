import numpy as np
import warp as wp

from evosim.config import Config, fill_struct
from evosim.genome import isa, mutate, table
from evosim.genome.assembler import genome_hash
from evosim.genome.reference_vm import build_label_table
from evosim.state import SimState


def _run_jobs(device, words, lens, **over):
    n = len(lens)
    cfg = Config(max_cells=1024, world_w=256, world_h=256, **over)
    st = SimState(cfg, device=device)
    wp.launch(table.upload_kernel, n, [st.genomes, wp.array(np.arange(n, dtype=np.int32), device=device),
                                       wp.array(words, device=device), wp.array(lens, device=device), wp.int64(0)],
              device=device)
    p = fill_struct(cfg)
    p.d_key = wp.uint32(99)
    j = st.jobs
    src = np.zeros(1024, np.int32); src[:n] = np.arange(n)
    dst = np.zeros(1024, np.int32); dst[:n] = np.arange(n) + n
    j.src.assign(src); j.dst.assign(dst); j.mult.fill_(1.0); j.uid.assign(np.arange(1024, dtype=np.int64))
    wp.launch(mutate.mutate_kernel, 1024, [st.genomes, j, wp.array([n], dtype=wp.int64, device=device), p],
              device=device)
    return st, j


def test_mutation_bounds_and_tables(device):
    rs = np.random.default_rng(3)
    n = 200
    words = rs.integers(0, 2**32, (n, 256), dtype=np.uint64).astype(np.uint32).view(np.int32)
    lens = rs.integers(4, 257, n).astype(np.int32)
    lens[:20] = 256      # at the maximum: insertions / duplications must be skipped
    lens[20:40] = 4      # at the minimum: deletions must be skipped
    st, j = _run_jobs(device, words, lens, min_genome_len=4, mu_point=0.05, mu_insert=0.6, mu_delete=0.6,
                      mu_duplicate=0.6, mu_invert=0.6, mu_transpose=0.6)
    L = st.genomes.length.numpy()[n:2 * n]
    assert L.min() >= 4 and L.max() <= isa.MAX_GENOME_LEN
    assert (j.events.numpy()[:n] > 0).all()
    W = st.genomes.words.numpy()
    same = j.same.numpy()[:n]
    for i in range(n, 2 * n):
        if same[i - n]:
            continue
        w = [int(x) & 0xFFFFFFFF for x in W[i, :L[i - n]]]
        o, _ = build_label_table(w)
        assert list(st.genomes.lab_off.numpy()[i]) == o
        assert int(st.genomes.hash.numpy()[i]) & (2**64 - 1) == genome_hash(w)


def test_zero_rates_copy_exactly_and_report_same(device):
    rs = np.random.default_rng(4)
    n = 50
    words = rs.integers(0, 2**32, (n, 256), dtype=np.uint64).astype(np.uint32).view(np.int32)
    lens = rs.integers(4, 100, n).astype(np.int32)
    st, j = _run_jobs(device, words, lens, mu_point=0, mu_insert=0, mu_delete=0, mu_duplicate=0, mu_invert=0,
                      mu_transpose=0)
    assert (j.same.numpy()[:n] == 1).all() and (j.events.numpy()[:n] == 0).all()


def test_mutation_is_deterministic_per_uid(device):
    rs = np.random.default_rng(5)
    n = 30
    words = rs.integers(0, 2**32, (n, 256), dtype=np.uint64).astype(np.uint32).view(np.int32)
    lens = rs.integers(4, 100, n).astype(np.int32)
    kw = dict(mu_point=0.05, mu_insert=0.5)
    a, _ = _run_jobs(device, words, lens, **kw)
    b, _ = _run_jobs(device, words, lens, **kw)
    assert np.array_equal(a.genomes.words.numpy()[n:2 * n], b.genomes.words.numpy()[n:2 * n])
