"""Structure-of-arrays GPU state, free lists, counters.

Everything the simulation needs to continue exactly lives in Cells, Genomes, Bonds, Fields and the
counters array (plus the host-side tick / config / policy state). Scratch and Work are rebuilt
every tick.
"""
from __future__ import annotations

import math

import numpy as np
import warp as wp

from .genome import isa

MAX_CONTACTS = 8
TRACE_SLOTS = 8
TRACE_LEN = 64
FX = 4294967296.0          # fixed-point scale (2^32) for deterministic int64 scatter accumulation

# counters (int64 device array)
(C_CELL_FREE, C_GENOME_FREE, C_BOND_FREE, C_UID, C_LINEAGE, C_BIRTHS, C_DEATHS, C_SPLIT_TOTAL, C_SPLITB_TOTAL,
 C_RELEASE_TOTAL, C_DYING_TOTAL, C_BOND_NEW_TOTAL, C_BOND_DEAD_TOTAL, C_ORG_CHANGED, C_VCLAMP, C_WCLAMP,
 C_GFREE_TOTAL, C_SPLIT_FAIL_CAP, C_BITES, C_ENGULFS, C_BONDS_BROKEN) = range(21)
NUM_COUNTERS = 32

# per-cell energy breakdown columns (this tick) -- also the cell-side ledger terms
EB_NAMES = ("photo", "uptake", "pred_gain", "share_in", "intervention",       # inflows
            "basal", "upkeep", "trait_change", "instr", "thrust", "turn", "split", "bond", "signal",
            "emit", "share_out", "cap_loss", "growth", "victim_loss")            # outflows
NUM_EB = len(EB_NAMES)
(EB_PHOTO, EB_UPTAKE, EB_PRED, EB_SHARE_IN, EB_INTERV, EB_BASAL, EB_UPKEEP, EB_TCHANGE, EB_INSTR, EB_THRUST,
 EB_TURN, EB_SPLIT, EB_BOND, EB_SIGNAL, EB_EMIT, EB_SHARE_OUT, EB_CAP, EB_GROWTH, EB_VICTIM) = range(NUM_EB)
EB_INFLOWS = 5

# global ledger terms (float64 device array, accumulated by kernels when ledger is on)
LEDGER_NAMES = ("death_energy", "death_mass", "detritus", "growth_mass", "victim_mass", "pred_mass_gain",
                "uptake_field", "regen", "decay", "interv_mass", "interv_energy", "interv_field",
                "emit_chem", "leak_chem")
NUM_LEDGER = len(LEDGER_NAMES)
(L_DEATH_E, L_DEATH_M, L_DETRITUS, L_GROWTH_M, L_VICTIM_M, L_PRED_M, L_UPTAKE_F, L_REGEN, L_DECAY, L_INTERV_M,
 L_INTERV_E, L_INTERV_F, L_EMIT_CHEM, L_LEAK_CHEM) = range(NUM_LEDGER)


@wp.struct
class Cells:
    alive: wp.array(dtype=wp.int32)
    pos: wp.array(dtype=wp.vec2)
    vel: wp.array(dtype=wp.vec2)
    angle: wp.array(dtype=float)
    ang_vel: wp.array(dtype=float)
    mass: wp.array(dtype=float)
    energy: wp.array(dtype=float)
    age: wp.array(dtype=wp.int32)
    genome_id: wp.array(dtype=wp.int32)
    ip: wp.array(dtype=wp.int32)
    wait: wp.array(dtype=wp.int32)
    regs: wp.array2d(dtype=float)
    state: wp.array(dtype=wp.int32)
    traits: wp.array2d(dtype=float)
    trait_targets: wp.array2d(dtype=float)
    signal_out: wp.array2d(dtype=float)
    signal_prev: wp.array2d(dtype=float)
    bonds: wp.array2d(dtype=wp.int32)
    uid: wp.array(dtype=wp.int64)
    parent_uid: wp.array(dtype=wp.int64)
    division_depth: wp.array(dtype=wp.int32)
    lineage_id: wp.array(dtype=wp.int32)
    organism_id: wp.array(dtype=wp.int32)
    birth_tick: wp.array(dtype=wp.int64)
    action_flags: wp.array(dtype=wp.int32)
    fail_flags: wp.array(dtype=wp.int32)
    origin: wp.array(dtype=wp.int32)
    light_prev: wp.array(dtype=float)
    damage: wp.array(dtype=float)
    traced: wp.array(dtype=wp.int32)


CELL_FIELDS_1D = ("alive", "pos", "vel", "angle", "ang_vel", "mass", "energy", "age", "genome_id", "ip", "wait",
                  "state", "uid", "parent_uid", "division_depth", "lineage_id", "organism_id", "birth_tick",
                  "action_flags", "fail_flags", "origin", "light_prev", "damage", "traced")
CELL_FIELDS_2D = ("regs", "traits", "trait_targets", "signal_out", "signal_prev", "bonds")


@wp.struct
class Genomes:
    words: wp.array2d(dtype=wp.int32)
    length: wp.array(dtype=wp.int32)
    lab_off: wp.array2d(dtype=wp.int32)
    lab_pos: wp.array2d(dtype=wp.int16)
    refcount: wp.array(dtype=wp.int32)
    in_use: wp.array(dtype=wp.int32)
    parent: wp.array(dtype=wp.int32)
    hash: wp.array(dtype=wp.int64)
    first_tick: wp.array(dtype=wp.int64)
    free_stack: wp.array(dtype=wp.int32)


@wp.struct
class Bonds:
    a: wp.array(dtype=wp.int32)
    b: wp.array(dtype=wp.int32)
    slot_a: wp.array(dtype=wp.int32)
    slot_b: wp.array(dtype=wp.int32)
    rest_len: wp.array(dtype=float)
    rest_ang_a: wp.array(dtype=float)
    rest_ang_b: wp.array(dtype=float)
    strength: wp.array(dtype=float)
    strain: wp.array(dtype=float)
    alive: wp.array(dtype=wp.int32)
    kill: wp.array(dtype=wp.int32)
    free_stack: wp.array(dtype=wp.int32)


@wp.struct
class Fields:
    nutrient: wp.array2d(dtype=float)         # concentration (mass / u^2)
    nutrient_tmp: wp.array2d(dtype=float)
    cap_map: wp.array2d(dtype=float)          # pattern in [0,1] scaling nutrient_cap
    chem: wp.array3d(dtype=float)             # [k, x, y]
    chem_tmp: wp.array3d(dtype=float)
    obstacle: wp.array2d(dtype=wp.int32)
    d_nutrient: wp.array2d(dtype=wp.int64)    # fixed-point deposits (amount, not concentration)
    d_chem: wp.array3d(dtype=wp.int64)
    demand: wp.array2d(dtype=wp.int64)        # nutrient uptake demand (fixed-point amount)
    grant: wp.array2d(dtype=float)            # granted fraction of demand
    opacity: wp.array2d(dtype=float)          # light grid, opacity density (1/u)
    light_map: wp.array2d(dtype=float)        # sun modulation on the light grid (for overlay)


@wp.struct
class Hash:
    keys: wp.array(dtype=wp.int32)
    vals: wp.array(dtype=wp.int32)
    start: wp.array(dtype=wp.int32)
    end: wp.array(dtype=wp.int32)


@wp.struct
class Scratch:
    sensors: wp.array2d(dtype=float)
    recv_agg: wp.array2d(dtype=float)
    recv_slot: wp.array2d(dtype=float)        # [N, max_bonds * k_sig]
    contacts: wp.array2d(dtype=wp.int32)      # [N, MAX_CONTACTS]
    n_contacts: wp.array(dtype=wp.int32)
    sector_best: wp.array2d(dtype=wp.int32)   # [N, 4] neighbour with max overlap per sector
    light_total: wp.array(dtype=float)
    light_recv: wp.array(dtype=float)
    # actions
    a_thrust: wp.array(dtype=float)
    a_turn: wp.array(dtype=float)
    a_split_angle: wp.array(dtype=float)
    a_split_frac: wp.array(dtype=float)
    a_split_label: wp.array(dtype=wp.int32)
    a_bond_sector: wp.array(dtype=wp.int32)
    a_unbond: wp.array(dtype=wp.int32)
    a_emit_ch: wp.array(dtype=wp.int32)
    a_emit_amt: wp.array(dtype=float)
    a_share: wp.array(dtype=float)
    a_eat: wp.array(dtype=float)
    a_signal_count: wp.array(dtype=wp.int32)
    n_exec: wp.array(dtype=wp.int32)
    eb: wp.array2d(dtype=float)               # energy breakdown [N, NUM_EB]
    # physics
    thrust_force: wp.array(dtype=float)
    turn_torque: wp.array(dtype=float)
    force: wp.array(dtype=wp.vec2)
    torque: wp.array(dtype=float)
    # interactions
    uptake_q: wp.array(dtype=wp.int64)
    bite_demand: wp.array(dtype=wp.int64)
    bite_amt: wp.array2d(dtype=float)         # [N, MAX_CONTACTS]
    bite_frac: wp.array(dtype=float)
    share_out: wp.array(dtype=float)
    bond_target: wp.array(dtype=wp.int32)
    bond_lock: wp.array(dtype=wp.int32)
    # lifecycle
    flag: wp.array(dtype=wp.int32)            # generic per-cell flag for scans
    ticket: wp.array(dtype=wp.int32)
    flag2: wp.array(dtype=wp.int32)
    ticket2: wp.array(dtype=wp.int32)
    daughter: wp.array(dtype=wp.int32)
    dying: wp.array(dtype=wp.int32)
    bflag: wp.array(dtype=wp.int32)           # per-bond flag
    bticket: wp.array(dtype=wp.int32)
    gflag: wp.array(dtype=wp.int32)           # per-genome flag
    gticket: wp.array(dtype=wp.int32)
    org_tmp: wp.array(dtype=wp.int32)
    org_size: wp.array(dtype=wp.int32)


@wp.struct
class MutJobs:
    """Genome copy+mutate jobs (births, clones)."""
    src: wp.array(dtype=wp.int32)
    dst: wp.array(dtype=wp.int32)
    mult: wp.array(dtype=float)
    uid: wp.array(dtype=wp.int64)
    events: wp.array(dtype=wp.int32)
    same: wp.array(dtype=wp.int32)


@wp.struct
class Trace:
    tick: wp.array2d(dtype=wp.int64)          # [TRACE_SLOTS, TRACE_LEN]
    ip: wp.array2d(dtype=wp.int32)
    word: wp.array2d(dtype=wp.int32)
    regs: wp.array3d(dtype=float)             # [TRACE_SLOTS, TRACE_LEN, 8]
    head: wp.array(dtype=wp.int32)            # next write position per slot (monotonic)
    bp_op: wp.array(dtype=wp.int32)           # breakpoint opcode per slot (-1 off)
    bp_addr: wp.array(dtype=wp.int32)         # breakpoint address per slot (-1 off)
    hit: wp.array(dtype=wp.int32)             # breakpoint hit flags per slot


class Dims:
    """Sizes derived from config (restart-time)."""

    def __init__(self, cfg):
        self.N = int(cfg.max_cells)
        self.G = self.N                       # genome capacity (>= live cells, see DESIGN_NOTES)
        self.MB = int(cfg.max_bonds)
        self.B = max(1, self.N * self.MB // 2)
        self.K = int(cfg.k_chem)
        self.KS = int(cfg.k_sig)
        self.T = isa.NUM_TRAITS
        self.L = isa.MAX_GENOME_LEN
        self.NS = isa.num_sensors(self.K)
        self.world = (float(cfg.world_w), float(cfg.world_h))
        self.fh = float(cfg.field_cell)
        self.fnx = max(3, int(round(self.world[0] / self.fh)))
        self.fny = max(3, int(round(self.world[1] / self.fh)))
        self.lh = float(cfg.light_cell)
        self.lnx = max(3, int(round(self.world[0] / self.lh)))
        self.lny = max(3, int(round(self.world[1] / self.lh)))
        # hash grid sized for the smallest allowed bucket (4 u); actual bucket chosen per tick
        self.hmax = max(9, int(self.world[0] // 4) * int(self.world[1] // 4))


def _z(shape, dtype, device):
    return wp.zeros(shape, dtype=dtype, device=device)


class SimState:
    def __init__(self, cfg, device=None):
        self.device = device or wp.get_device()
        d = self.dims = Dims(cfg)
        dev = self.device
        N, G, B, MB, T, K, KS, L = d.N, d.G, d.B, d.MB, d.T, d.K, d.KS, d.L

        c = self.cells = Cells()
        for n, t in (("alive", wp.int32), ("angle", float), ("ang_vel", float), ("mass", float), ("energy", float),
                     ("age", wp.int32), ("genome_id", wp.int32), ("ip", wp.int32), ("wait", wp.int32),
                     ("state", wp.int32), ("uid", wp.int64), ("parent_uid", wp.int64), ("division_depth", wp.int32),
                     ("lineage_id", wp.int32), ("organism_id", wp.int32), ("birth_tick", wp.int64),
                     ("action_flags", wp.int32), ("fail_flags", wp.int32), ("origin", wp.int32),
                     ("light_prev", float), ("damage", float), ("traced", wp.int32)):
            setattr(c, n, _z(N, t, dev))
        c.pos = _z(N, wp.vec2, dev)
        c.vel = _z(N, wp.vec2, dev)
        c.regs = _z((N, isa.NUM_REGS), float, dev)
        c.traits = _z((N, T), float, dev)
        c.trait_targets = _z((N, T), float, dev)
        c.signal_out = _z((N, KS), float, dev)
        c.signal_prev = _z((N, KS), float, dev)
        c.bonds = wp.full((N, MB), -1, dtype=wp.int32, device=dev)
        c.traced.fill_(-1)

        g = self.genomes = Genomes()
        g.words = _z((G, L), wp.int32, dev)
        g.length = _z(G, wp.int32, dev)
        g.lab_off = _z((G, isa.NUM_LABELS + 1), wp.int32, dev)
        g.lab_pos = _z((G, L), wp.int16, dev)
        g.refcount = _z(G, wp.int32, dev)
        g.in_use = _z(G, wp.int32, dev)
        g.parent = wp.full(G, -1, dtype=wp.int32, device=dev)
        g.hash = _z(G, wp.int64, dev)
        g.first_tick = _z(G, wp.int64, dev)
        g.free_stack = wp.array(np.arange(G - 1, -1, -1, dtype=np.int32), device=dev)

        b = self.bonds = Bonds()
        for n in ("a", "b", "slot_a", "slot_b", "alive", "kill"):
            setattr(b, n, _z(B, wp.int32, dev))
        for n in ("rest_len", "rest_ang_a", "rest_ang_b", "strength", "strain"):
            setattr(b, n, _z(B, float, dev))
        b.free_stack = wp.array(np.arange(B - 1, -1, -1, dtype=np.int32), device=dev)

        f = self.fields = Fields()
        f.nutrient = _z((d.fnx, d.fny), float, dev)
        f.nutrient_tmp = _z((d.fnx, d.fny), float, dev)
        f.cap_map = wp.ones((d.fnx, d.fny), dtype=float, device=dev)
        f.chem = _z((K, d.fnx, d.fny), float, dev)
        f.chem_tmp = _z((K, d.fnx, d.fny), float, dev)
        f.obstacle = _z((d.fnx, d.fny), wp.int32, dev)
        f.d_nutrient = _z((d.fnx, d.fny), wp.int64, dev)
        f.d_chem = _z((K, d.fnx, d.fny), wp.int64, dev)
        f.demand = _z((d.fnx, d.fny), wp.int64, dev)
        f.grant = _z((d.fnx, d.fny), float, dev)
        f.opacity = _z((d.lnx, d.lny), float, dev)
        f.light_map = _z((d.lnx, d.lny), float, dev)

        h = self.hash = Hash()
        h.keys = _z(2 * N, wp.int32, dev)
        h.vals = _z(2 * N, wp.int32, dev)
        h.start = _z(d.hmax, wp.int32, dev)
        h.end = _z(d.hmax, wp.int32, dev)

        s = self.scratch = Scratch()
        s.sensors = _z((N, d.NS), float, dev)
        s.recv_agg = _z((N, KS), float, dev)
        s.recv_slot = _z((N, MB * KS), float, dev)
        s.contacts = _z((N, MAX_CONTACTS), wp.int32, dev)
        s.n_contacts = _z(N, wp.int32, dev)
        s.sector_best = _z((N, 4), wp.int32, dev)
        s.eb = _z((N, NUM_EB), float, dev)
        s.bite_amt = _z((N, MAX_CONTACTS), float, dev)
        for n in ("light_total", "light_recv", "a_thrust", "a_turn", "a_split_angle", "a_split_frac", "a_emit_amt",
                  "a_share", "a_eat", "thrust_force", "turn_torque", "torque", "bite_frac", "share_out"):
            setattr(s, n, _z(N, float, dev))
        for n in ("a_split_label", "a_bond_sector", "a_unbond", "a_emit_ch", "a_signal_count", "n_exec",
                  "bond_target", "bond_lock", "flag", "ticket", "flag2", "ticket2", "daughter", "dying",
                  "org_tmp", "org_size"):
            setattr(s, n, _z(N, wp.int32, dev))
        s.force = _z(N, wp.vec2, dev)
        s.uptake_q = _z(N, wp.int64, dev)
        s.bite_demand = _z(N, wp.int64, dev)
        s.bflag = _z(B, wp.int32, dev)
        s.bticket = _z(B, wp.int32, dev)
        s.gflag = _z(G, wp.int32, dev)
        s.gticket = _z(G, wp.int32, dev)

        j = self.jobs = MutJobs()
        j.src = _z(N, wp.int32, dev)
        j.dst = _z(N, wp.int32, dev)
        j.mult = _z(N, float, dev)
        j.uid = _z(N, wp.int64, dev)
        j.events = _z(N, wp.int32, dev)
        j.same = _z(N, wp.int32, dev)

        t = self.trace = Trace()
        t.tick = _z((TRACE_SLOTS, TRACE_LEN), wp.int64, dev)
        t.ip = _z((TRACE_SLOTS, TRACE_LEN), wp.int32, dev)
        t.word = _z((TRACE_SLOTS, TRACE_LEN), wp.int32, dev)
        t.regs = _z((TRACE_SLOTS, TRACE_LEN, isa.NUM_REGS), float, dev)
        t.head = _z(TRACE_SLOTS, wp.int32, dev)
        t.bp_op = wp.full(TRACE_SLOTS, -1, dtype=wp.int32, device=dev)
        t.bp_addr = wp.full(TRACE_SLOTS, -1, dtype=wp.int32, device=dev)
        t.hit = _z(TRACE_SLOTS, wp.int32, dev)

        self.cell_free_stack = wp.array(np.arange(N - 1, -1, -1, dtype=np.int32), device=dev)
        self.counters = _z(NUM_COUNTERS, wp.int64, dev)
        self.ledger = _z(NUM_LEDGER, wp.float64, dev)
        self.point_lights = _z(16, wp.vec3, dev)   # (x, y, I)
        cnt = np.zeros(NUM_COUNTERS, np.int64)
        cnt[C_CELL_FREE] = N
        cnt[C_GENOME_FREE] = G
        cnt[C_BOND_FREE] = B
        cnt[C_UID] = 1
        cnt[C_LINEAGE] = 1
        self.counters.assign(cnt)

    # ------------------------------------------------------------------ helpers
    def counters_np(self):
        return self.counters.numpy()

    def vram_bytes(self):
        total = 0
        for obj in (self.cells, self.genomes, self.bonds, self.fields, self.hash, self.scratch, self.jobs,
                    self.trace):
            for name in dir(obj):
                a = getattr(obj, name, None)
                if isinstance(a, wp.array):
                    total += a.capacity
        return total


def estimate_vram(cfg) -> int:
    """Rough upper bound of state memory in bytes (without warp/CUDA overhead)."""
    st = Dims(cfg)
    N, G, B, MB, KS, NS, L = st.N, st.G, st.B, st.MB, st.KS, st.NS, st.L
    cells = N * (4 * 20 + 8 * 3 + 8 * 2 + 4 * (8 + 2 * 9 + 2 * KS + MB))
    genomes = G * (L * 4 + L * 2 + 33 * 4 + 4 * 5 + 8 * 2)
    bonds = B * (4 * 11)
    scratch = N * (4 * (NS + KS + MB * KS + MAX_CONTACTS * 2 + 4 + NUM_EB + 40) + 8 * 4) + B * 8 + G * 8
    fields = st.fnx * st.fny * (4 * 5 + 8 * 2 + st.K * (4 * 2 + 8)) + st.lnx * st.lny * 8
    return cells + genomes + bonds + scratch + fields + 2 * N * 8 + st.hmax * 8
