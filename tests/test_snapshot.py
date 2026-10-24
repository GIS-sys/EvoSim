import os

import numpy as np
import pytest

from evosim.config import Config
from evosim.io import snapshot
from evosim.sim import Sim


def _mk(device):
    cfg = Config(max_cells=2048, world_w=160.0, world_h=160.0, initial_count=30, seed=21,
                 initial_seed_files="seeds/seed_replicator.asm,seeds/seed_colony.asm,seeds/seed_grazer.asm,seeds/seed_predator.asm",
                 initial_source="seed", mu_point=0.01, mu_insert=0.05, mu_delete=0.05, ledger_interval=0,
                 sun_I=2.0, organism_label_interval=5, k_chem=4, passive_leak=0.5)
    return Sim(cfg, device=device)


def test_snapshot_roundtrip_is_exact_continuation(device, tmp_path):
    a = _mk(device)
    a.step(120)
    path = str(tmp_path / "s.evck")
    snapshot.save_snapshot(a, path)
    a.step(60)
    ha, pa = a.state_hash(), a.population()
    b, _ = snapshot.load_snapshot(path, device=device)
    assert b.tick == 120
    b.step(60)
    assert b.population() == pa and b.state_hash() == ha          # bit-identical continuation
    assert a.tick == b.tick == 180


def test_load_restores_state_exactly_before_continuing(device, tmp_path):
    a = _mk(device)
    a.step(80)
    path = str(tmp_path / "s.evck")
    snapshot.save_snapshot(a, path)
    b, _ = snapshot.load_snapshot(path, device=device)
    assert a.state_hash() == b.state_hash()
    assert int(a.state.bonds.alive.numpy().sum()) == int(b.state.bonds.alive.numpy().sum())
    assert np.array_equal(a.state.fields.nutrient.numpy(), b.state.fields.nutrient.numpy())
    assert np.array_equal(a.state.counters.numpy(), b.state.counters.numpy())


def test_inplace_restore_for_rollback(device, tmp_path):
    a = _mk(device)
    a.step(50)
    path = str(tmp_path / "s.evck")
    snapshot.save_snapshot(a, path)
    h = a.state_hash()
    a.step(40)
    assert a.state_hash() != h
    snapshot.load_snapshot(path, sim=a)
    assert a.tick == 50 and a.state_hash() == h


def test_corrupt_file_detected(device, tmp_path):
    a = _mk(device)
    a.step(5)
    path = str(tmp_path / "s.evck")
    snapshot.save_snapshot(a, path)
    data = bytearray(open(path, "rb").read())
    data[-20] ^= 0xFF
    open(path, "wb").write(bytes(data))
    with pytest.raises(snapshot.CheckpointError):
        snapshot.load_snapshot(path, device=device)
    open(path, "wb").write(bytes(data[:100]))                       # truncated
    with pytest.raises(snapshot.CheckpointError):
        snapshot.read_container(path)


def test_atomic_write_killed_midway_leaves_previous_loadable(device, tmp_path, monkeypatch):
    a = _mk(device)
    a.step(5)
    path = str(tmp_path / "s.evck")
    snapshot.save_snapshot(a, path)
    h = a.state_hash()
    a.step(5)

    def boom(src, dst):
        raise KeyboardInterrupt("killed while replacing")
    monkeypatch.setattr(os, "replace", boom)                       # the process "dies" before the rename
    with pytest.raises(KeyboardInterrupt):
        snapshot.save_snapshot(a, path)
    monkeypatch.undo()
    assert not [f for f in os.listdir(tmp_path) if f.startswith(".tmp_")]    # temp file cleaned
    b, _ = snapshot.load_snapshot(path, device=device)
    assert b.tick == 5 and b.state_hash() == h
    # a torn temp file from a hard kill (never renamed) does not disturb the real file either
    open(tmp_path / ".tmp_dead.evck", "wb").write(b"EVCK1garbage")
    b2, _ = snapshot.load_snapshot(path, device=device)
    assert b2.tick == 5
