"""Exact snapshots and the checkpoint container format.

capture(sim) -> dict[str, ndarray]   GPU -> host copy (synchronous, fast)
restore(sim, cap)                     load it back so the run continues bit-identically
Only alive cells, in-use genomes and alive bonds are stored, but at their ORIGINAL slot indices and
together with the free-list stacks, so slot allocation order (hence neighbour order, hence float
summation order) after a restore is identical to the uninterrupted run.

Container (.evck):  b"EVCK1" | u32 meta_len | meta_json | sha256(meta_json + payload) | payload
payload = zstd(npz). Writes are atomic (temp file + fsync + os.replace). The loader verifies the checksum.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import tempfile

import numpy as np
import warp as wp

from ..config import Config
from ..genome import table
from ..state import CELL_FIELDS_1D, CELL_FIELDS_2D, C_BOND_FREE, C_CELL_FREE, C_GENOME_FREE

try:
    import zstandard as zstd
except ImportError:           # pragma: no cover
    zstd = None

MAGIC = b"EVCK1"


class CheckpointError(RuntimeError):
    pass


def _j(obj) -> np.ndarray:
    return np.frombuffer(json.dumps(obj, sort_keys=True).encode("utf-8"), dtype=np.uint8).copy()


def _unj(arr):
    return json.loads(bytes(np.asarray(arr, np.uint8)).decode("utf-8"))


# ----------------------------------------------------------------------------- capture / restore
def capture(sim, extra: dict | None = None) -> dict:
    st = sim.state
    wp.synchronize_device(st.device)
    cap: dict = {}
    c = st.cells
    alive = c.alive.numpy() != 0
    idx = np.nonzero(alive)[0].astype(np.int32)
    cap["cell_idx"] = idx
    for name in CELL_FIELDS_1D:
        if name != "alive":
            cap[f"c_{name}"] = getattr(c, name).numpy()[idx]
    for name in CELL_FIELDS_2D:
        cap[f"c_{name}"] = getattr(c, name).numpy()[idx]
    g = st.genomes
    gin = np.nonzero(g.in_use.numpy() != 0)[0].astype(np.int32)
    cap["gen_idx"] = gin
    gl = g.length.numpy()[gin]
    cap["g_length"] = gl
    W = g.words.numpy()[gin]
    cap["g_words"] = W[:, :int(gl.max())] if len(gin) else W[:, :1]
    for name in ("refcount", "parent", "first_tick"):
        cap[f"g_{name}"] = getattr(g, name).numpy()[gin]
    b = st.bonds
    bin_ = np.nonzero(b.alive.numpy() != 0)[0].astype(np.int32)
    cap["bond_idx"] = bin_
    for name in ("a", "b", "slot_a", "slot_b", "rest_len", "rest_ang_a", "rest_ang_b", "strength", "strain"):
        cap[f"b_{name}"] = getattr(b, name).numpy()[bin_]
    cnt = st.counters.numpy().copy()
    cap["counters"] = cnt
    cap["cell_free"] = st.cell_free_stack.numpy()[:int(cnt[C_CELL_FREE])]
    cap["genome_free"] = g.free_stack.numpy()[:int(cnt[C_GENOME_FREE])]
    cap["bond_free"] = b.free_stack.numpy()[:int(cnt[C_BOND_FREE])]
    f = st.fields
    for name in ("nutrient", "chem", "cap_map", "obstacle"):
        cap[f"f_{name}"] = getattr(f, name).numpy()
    cap["pending_cull"] = np.nonzero(st.scratch.dying.numpy() == 2)[0].astype(np.int32)
    meta = {"tick": int(sim.tick), "config": sim.cfg.to_dict(), "env": _clean_env(sim.env),
            "pending_interv": sim.pending_interv, "traced": {str(k): v for k, v in sim.traced.items()},
            "extra": extra or {}}
    cap["meta_json"] = _j(meta)
    return cap


def _clean_env(env):
    return json.loads(json.dumps(env))


def restore(sim, cap: dict):
    st = sim.state
    d = st.dims
    dev = st.device
    meta = _unj(cap["meta_json"])
    cfg_new = Config.from_dict(meta["config"])
    for k in ("max_cells", "max_bonds", "k_chem", "k_sig", "world_w", "world_h", "field_cell", "light_cell"):
        if getattr(cfg_new, k) != getattr(sim.cfg, k):
            raise CheckpointError(f"snapshot parameter {k}={getattr(cfg_new, k)} != running {getattr(sim.cfg, k)}")
    for k, v in cfg_new._v.items():
        sim.cfg.set(k, v)
    sim.cfg.point_lights = list(cfg_new.point_lights)
    N, G, B = d.N, d.G, d.B
    # ---- cells
    idx = cap["cell_idx"]
    c = st.cells
    alive = np.zeros(N, np.int32)
    alive[idx] = 1
    c.alive.assign(alive)
    for name in CELL_FIELDS_1D:
        if name == "alive":
            continue
        arr = getattr(c, name)
        full = np.zeros_like(arr.numpy())
        if name == "traced":
            full[:] = -1
        full[idx] = cap[f"c_{name}"]
        arr.assign(full)
    for name in CELL_FIELDS_2D:
        arr = getattr(c, name)
        full = np.zeros_like(arr.numpy())
        if name == "bonds":
            full[:] = -1
        full[idx] = cap[f"c_{name}"]
        arr.assign(full)
    # ---- genomes
    g = st.genomes
    gin = cap["gen_idx"]
    words = np.zeros((G, d.L), np.int32)
    if len(gin):
        words[gin, :cap["g_words"].shape[1]] = cap["g_words"]
    g.words.assign(words)
    for name, dt in (("length", np.int32), ("refcount", np.int32), ("in_use", np.int32)):
        full = np.zeros(G, dt)
        if name == "in_use":
            full[gin] = 1
        else:
            full[gin] = cap[f"g_{name}" if name != "length" else "g_length"]
        getattr(g, name).assign(full)
    par = np.full(G, -1, np.int32)
    par[gin] = cap["g_parent"]
    g.parent.assign(par)
    ft = np.zeros(G, np.int64)
    ft[gin] = cap["g_first_tick"]
    g.first_tick.assign(ft)
    if len(gin):
        wp.launch(table.finalize_list_kernel, len(gin), [g, wp.array(gin, dtype=wp.int32, device=dev)], device=dev)
    # ---- bonds
    b = st.bonds
    bi = cap["bond_idx"]
    for name in ("a", "b", "slot_a", "slot_b"):
        full = np.zeros(B, np.int32)
        full[bi] = cap[f"b_{name}"]
        getattr(b, name).assign(full)
    for name in ("rest_len", "rest_ang_a", "rest_ang_b", "strength", "strain"):
        full = np.zeros(B, np.float32)
        full[bi] = cap[f"b_{name}"]
        getattr(b, name).assign(full)
    al = np.zeros(B, np.int32)
    al[bi] = 1
    b.alive.assign(al)
    b.kill.zero_()
    # ---- free lists, counters
    for arr, key, n in ((st.cell_free_stack, "cell_free", N), (g.free_stack, "genome_free", G),
                        (b.free_stack, "bond_free", B)):
        full = arr.numpy().copy()
        seg = cap[key]
        full[:len(seg)] = seg
        arr.assign(full)
    st.counters.assign(cap["counters"])
    # ---- fields
    f = st.fields
    for name in ("nutrient", "chem", "cap_map", "obstacle"):
        getattr(f, name).assign(cap[f"f_{name}"])
    st.scratch.dying.zero_()
    pc = cap["pending_cull"]
    if len(pc):
        dy = np.zeros(N, np.int32)
        dy[pc] = 2
        st.scratch.dying.assign(dy)
    # ---- host state
    sim.tick = int(meta["tick"])
    sim.env = meta["env"]
    sim.pending_interv = meta["pending_interv"]
    sim.traced = {int(k): v for k, v in meta["traced"].items()}
    sim._p_version = -1
    sim.last_ledger = None
    return meta.get("extra", {})


# ----------------------------------------------------------------------------- serialisation
def cap_to_bytes(cap: dict, level=3) -> bytes:
    bio = io.BytesIO()
    np.savez(bio, **cap)
    raw = bio.getvalue()
    if zstd is None:
        import zlib
        return b"Z" + zlib.compress(raw, 3)
    return b"S" + zstd.ZstdCompressor(level=level, threads=0).compress(raw)


def cap_from_bytes(payload: bytes) -> dict:
    kind, body = payload[:1], payload[1:]
    if kind == b"S":
        if zstd is None:
            raise CheckpointError("zstandard is required to read this file")
        raw = zstd.ZstdDecompressor().decompress(body, max_output_size=1 << 33)
    else:
        import zlib
        raw = zlib.decompress(body)
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_container(path: str, payload: bytes, meta: dict):
    """Atomic: temp file in the same directory, fsync, os.replace."""
    mj = json.dumps(meta, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(mj + payload).digest()
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".evck", dir=d)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(MAGIC + struct.pack("<I", len(mj)) + mj + digest + payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_meta(path: str) -> dict:
    with open(path, "rb") as f:
        head = f.read(len(MAGIC) + 4)
        if head[:len(MAGIC)] != MAGIC:
            raise CheckpointError(f"{path}: not an EvoSim checkpoint")
        (n,) = struct.unpack("<I", head[len(MAGIC):])
        return json.loads(f.read(n).decode("utf-8"))


def read_container(path: str, verify=True):
    with open(path, "rb") as f:
        data = f.read()
    if data[:len(MAGIC)] != MAGIC or len(data) < len(MAGIC) + 4 + 32:
        raise CheckpointError(f"{path}: not an EvoSim checkpoint (or truncated)")
    (n,) = struct.unpack("<I", data[len(MAGIC):len(MAGIC) + 4])
    o = len(MAGIC) + 4
    mj = data[o:o + n]
    digest = data[o + n:o + n + 32]
    payload = data[o + n + 32:]
    if len(mj) != n or verify and hashlib.sha256(mj + payload).digest() != digest:
        raise CheckpointError(f"{path}: checksum mismatch (corrupt file)")
    return json.loads(mj.decode("utf-8")), payload


def save_snapshot(sim, path: str, meta: dict | None = None, extra: dict | None = None):
    cap = capture(sim, extra)
    m = {"tick": int(sim.tick), "population": sim.population(), "good": True}
    m.update(meta or {})
    write_container(path, cap_to_bytes(cap), m)


def load_snapshot(path: str, device=None, sim=None):
    """Load into a new Sim (or an existing compatible one). Returns (sim, extra)."""
    from ..sim import Sim
    meta, payload = read_container(path)
    cap = cap_from_bytes(payload)
    if sim is None:
        cfg = Config.from_dict(_unj(cap["meta_json"])["config"])
        sim = Sim(cfg, device=device, seed_population=False)
    extra = restore(sim, cap)
    return sim, extra
