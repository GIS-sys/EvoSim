"""GPU (Warp) VM vs CPU reference VM: bit-for-bit agreement on registers, ip, wait, state, trait
targets, signal_out and queued actions over many ticks with fixed sensor inputs.

Default size is reduced for CI speed on the CPU backend; set EVOSIM_FULL=1 for the
spec size (10k genomes x 100 ticks).
"""
import os

import numpy as np
import pytest
import warp as wp

from evosim import rng
from evosim.config import Config, fill_struct
from evosim.genome import isa, table
from evosim.genome.reference_vm import CellVM, VMInputs, run_tick
from evosim.kernels.vm import vm_kernel
from evosim.state import SimState

FULL = os.environ.get("EVOSIM_FULL") == "1"
N_GENOMES = 10000 if FULL else 300
N_TICKS = 100 if FULL else 40


def _random_genomes(rs, n):
    out = []
    for _ in range(n):
        L = int(rs.integers(1, 64))
        mode = rs.random()
        if mode < 0.5:
            w = rs.integers(0, 2**32, size=L, dtype=np.uint64)
        else:   # biased toward valid-ish canonical code (more control flow / labels)
            ops = rs.integers(0, isa.NUM_OPCODES, size=L)
            w = (ops.astype(np.uint64) << 24) | rs.integers(0, 2**24, size=L, dtype=np.uint64)
        out.append([int(x) for x in w])
    return out


@pytest.mark.parametrize("seed", [0, 1])
def test_vm_gpu_matches_cpu(device, seed):
    rs = np.random.default_rng(seed)
    n = N_GENOMES
    cfg = Config(max_cells=max(1024, n), world_w=256, world_h=256, instr_per_tick=16, max_wait=5)
    st = SimState(cfg, device=device)
    d = st.dims
    genomes = _random_genomes(rs, n)
    W = np.zeros((n, isa.MAX_GENOME_LEN), np.int32)
    lens = np.zeros(n, np.int32)
    for i, g in enumerate(genomes):
        W[i, :len(g)] = table.words_to_i32(g)
        lens[i] = len(g)
    wp.launch(table.upload_kernel, n, [st.genomes, wp.array(np.arange(n, dtype=np.int32), device=device),
                                       wp.array(W, device=device), wp.array(lens, device=device), wp.int64(0)],
              device=device)
    N = d.N
    alive = np.zeros(N, np.int32)
    alive[:n] = 1
    st.cells.alive.assign(alive)
    gid = np.zeros(N, np.int32)
    gid[:n] = np.arange(n)
    st.cells.genome_id.assign(gid)
    uids = np.arange(N, dtype=np.int64) * 7919 + 13
    st.cells.uid.assign(uids)
    init_regs = rs.normal(0, 3, size=(N, 8)).astype(np.float32)
    st.cells.regs.assign(init_regs)
    ips = np.array([rs.integers(0, l) for l in lens] + [0] * (N - n), dtype=np.int32)
    st.cells.ip.assign(ips)

    sensors = rs.normal(0, 2, size=(N, d.NS)).astype(np.float32)
    recv_agg = rs.normal(0, 1, size=(N, d.KS)).astype(np.float32)
    recv_slot = rs.normal(0, 1, size=(N, d.MB * d.KS)).astype(np.float32)
    st.scratch.sensors.assign(sensors)
    st.scratch.recv_agg.assign(recv_agg)
    st.scratch.recv_slot.assign(recv_slot)

    cpu = [CellVM(words=genomes[i], regs=init_regs[i].copy(), ip=int(ips[i]), uid=int(uids[i]),
                  signal_out=np.zeros(d.KS, np.float32)) for i in range(n)]
    p = fill_struct(cfg)
    p.d_num_sensors = d.NS
    for t in range(N_TICKS):
        key = rng.tick_key(cfg.seed, cfg.rng_salt, t)
        p.d_key = wp.uint32(key)
        p.d_tick = wp.int64(t)
        wp.launch(vm_kernel, N, [st.cells, st.genomes, st.scratch, st.trace, p], device=device)
        acts = []
        for i in range(n):
            inp = VMInputs(sensors[i], recv_agg[i], recv_slot[i].reshape(d.MB, d.KS), key)
            acts.append(run_tick(cpu[i], inp, cfg.instr_per_tick, cfg.max_wait, d.K, d.KS, d.MB))
        regs = st.cells.regs.numpy()[:n]
        ip = st.cells.ip.numpy()[:n]
        wait = st.cells.wait.numpy()[:n]
        state = st.cells.state.numpy()[:n]
        tt = st.cells.trait_targets.numpy()[:n]
        so = st.cells.signal_out.numpy()[:n]
        flags = st.cells.action_flags.numpy()[:n]
        fails = st.cells.fail_flags.numpy()[:n]
        thrust = st.scratch.a_thrust.numpy()[:n]
        lab = st.scratch.a_split_label.numpy()[:n]
        nex = st.scratch.n_exec.numpy()[:n]
        for i in range(n):
            c = cpu[i]
            ctx = f"tick {t} cell {i}"
            assert np.array_equal(regs[i].view(np.uint32), c.regs.view(np.uint32)), ctx + " regs"
            assert ip[i] == c.ip, ctx + " ip"
            assert wait[i] == c.wait, ctx + " wait"
            assert state[i] == c.state, ctx + " state"
            assert np.array_equal(tt[i].view(np.uint32), c.trait_targets.view(np.uint32)), ctx + " traits"
            assert np.array_equal(so[i].view(np.uint32), c.signal_out.view(np.uint32)), ctx + " signal"
            a = acts[i]
            if a.n_exec:   # sleeping cells don't reset the action scratch
                assert flags[i] == a.flags, ctx + " flags"
                assert fails[i] == a.fail, ctx + " fails"
                assert nex[i] == a.n_exec, ctx + " n_exec"
                assert np.float32(thrust[i]).view(np.uint32) == np.float32(a.thrust).view(np.uint32), ctx
                if a.flags & (isa.ACT_SPLIT | isa.ACT_SPLITB):
                    assert lab[i] == a.split_label, ctx
