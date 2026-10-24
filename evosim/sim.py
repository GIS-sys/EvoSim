"""Simulation orchestrator: tick pipeline, determinism keys, ledger, stats, seeding, edits.

Tick pipeline (spec section 8), each phase individually timed when profiling:
  1 hash  2 opacity  3 light  4 sense  5 vm  6 metabolism  7 interactions  8 births  9 deaths
  10 physics  11 fields  12 organisms (interval)  13 stats (interval)
"""
from __future__ import annotations

import math
import os
import time
from collections import deque

import numpy as np
import warp as wp

from . import rng, spawn
from .config import Config, avg_cos_pow, fill_struct, SimParams
from .genome import assembler, isa
from .kernels import fields, interactions, lifecycle, light, metabolism, organisms, physics, sense, spatial_hash
from .kernels.vm import vm_kernel
from .state import (C_LINEAGE, C_BIRTHS, C_BONDS_BROKEN, C_CELL_FREE, C_DEATHS, C_ENGULFS, C_GENOME_FREE, C_SPLIT_FAIL_CAP,
                    C_VCLAMP, C_WCLAMP, EB_INFLOWS, EB_NAMES, EB_UPTAKE, L_DEATH_E, L_DEATH_M, L_DECAY, L_DETRITUS,
                    L_GROWTH_M, L_INTERV_E, L_INTERV_F, L_INTERV_M, L_PRED_M, L_REGEN, L_UPTAKE_F, L_VICTIM_M,
                    LEDGER_NAMES, NUM_LEDGER, TRACE_SLOTS, SimState)

PHASES = ("hash", "opacity", "light", "sense", "vm", "metabolism", "interactions", "births", "deaths", "physics",
          "fields", "organisms", "stats")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LedgerError(RuntimeError):
    def __init__(self, report):
        super().__init__(report["message"])
        self.report = report


class Sim:
    def __init__(self, cfg: Config | None = None, device=None, seed_population=True):
        self.cfg = cfg or Config()
        self.device = wp.get_device(device) if device is not None else wp.get_device()
        self.tick = 0
        self.state = SimState(self.cfg, self.device)
        self._p = SimParams()
        self._p_version = -1
        self._lights_np = np.zeros((16, 3), np.float32)
        self.timings = {k: 0.0 for k in PHASES}
        self.timing_samples = 0
        self.last_ledger = None
        self.ledger_failures = 0
        self.stats_history = deque(maxlen=10000)
        self.last_stats = {}
        self.pending_interv = {"mass": 0.0, "energy": 0.0, "field": 0.0}
        self.interv_counts = {}
        self.events_hook = None               # callable(event_dict) set by the run layer
        self.env = {"hostile": None, "sun_mult": 1.0, "sun_angle_add": 0.0, "current": None}
        self.traced = {}                       # cell slot -> trace slot
        self._wall0 = time.perf_counter()
        self._tps_t = time.perf_counter()
        self._tps_tick = 0
        self.tps = 0.0
        self._init_world()
        if seed_population:
            self.seed_initial()

    # ================================================================== setup
    def _init_world(self):
        rs = np.random.default_rng(self.cfg.seed + 7919 * self.cfg.rng_salt)
        d = self.state.dims
        f = self.state.fields
        cap = fields.make_cap_map(self.cfg, d, rs)
        f.cap_map.assign(cap)
        f.nutrient.assign((cap * self.cfg.nutrient_cap).astype(np.float32))
        if self.cfg.obstacle_generator != "none":
            f.obstacle.assign(fields.make_obstacles(self.cfg, d, rs))

    def resolve_path(self, path):
        if os.path.isabs(path) or os.path.exists(path):
            return path
        return os.path.join(REPO_ROOT, path)

    def load_genome_file(self, path):
        d = self.state.dims
        return assembler.load_genome(self.resolve_path(path), k_chem=d.K, k_sig=d.KS, max_bonds=d.MB)

    def random_genome(self, rs, length=None, mode="uniform", weights=None):
        L = int(length or self.cfg.random_genome_len)
        if mode == "weighted":
            w = np.ones(isa.NUM_OPCODES) if weights is None else np.asarray(weights, float)
            ops = rs.choice(isa.NUM_OPCODES, size=L, p=w / w.sum())
            rest = rs.integers(0, 2**24, size=L)
            return [int((o << 24) | r) for o, r in zip(ops, rest)]
        return [int(x) for x in rs.integers(0, 2**32, size=L, dtype=np.uint64)]

    def seed_initial(self):
        cfg = self.cfg
        n = int(cfg.initial_count)
        if n <= 0:
            return
        rs = np.random.default_rng(cfg.seed + 1)
        files = [s.strip() for s in cfg.initial_seed_files.split(",") if s.strip()]
        genomes = []
        if cfg.initial_source in ("seed", "mix"):
            genomes += [("seed", self.load_genome_file(f)) for f in files]
        nseed = n if cfg.initial_source == "seed" else (n // 2 if cfg.initial_source == "mix" else 0)
        self.inject(nseed, [g for _, g in genomes], rs, origin=isa.ORIGIN_INJECTED) if nseed else None
        nrand = n - nseed
        if nrand > 0:
            self.inject(nrand, [self.random_genome(rs) for _ in range(nrand)], rs, origin=isa.ORIGIN_INJECTED,
                        one_genome_per_cell=True)

    # ================================================================== edits (host)
    def book_intervention(self, mass=0.0, energy=0.0, field=0.0):
        self.pending_interv["mass"] += mass
        self.pending_interv["energy"] += energy
        self.pending_interv["field"] += field

    def inject(self, k, genome_words_list, rs, origin=isa.ORIGIN_INJECTED, placement="uniform",
               one_genome_per_cell=False, mass=None, energy=None, traits=None, center=None, radius=None,
               region=None, weights=None, lineage=None):
        """Inject k cells. genome_words_list: genomes to draw from (cycled), or one per cell."""
        if k <= 0 or not genome_words_list:
            return np.zeros(0, np.int32)
        pos, skipped = spawn.place_positions(self, k, rs, mode=placement, center=center, radius=radius,
                                             region=region, weights_field=weights)
        k = len(pos)
        if k == 0:
            return np.zeros(0, np.int32)
        cnt = self.state.counters.numpy()
        k = min(k, int(cnt[C_CELL_FREE]))
        pos = pos[:k]
        if one_genome_per_cell:
            uniq = genome_words_list[:k]
            gidx = np.arange(k)
        else:
            uniq = genome_words_list
            gidx = np.arange(k) % len(uniq)
        gids = spawn.upload_genomes(self, uniq)
        ent = dict(pos=pos, angle=rs.random(k) * 2 * np.pi, gid=gids[gidx],
                   mass=np.full(k, mass if mass is not None else self.cfg.initial_mass),
                   energy=np.full(k, energy if energy is not None else self.cfg.initial_energy),
                   origin=np.full(k, origin, np.int32))
        if traits is not None:
            ent["traits"] = np.tile(np.asarray(traits, np.float32), (k, 1))
            ent["targets"] = ent["traits"]
        if lineage is not None:
            ent["lineage"] = np.full(k, lineage, np.int32)
        else:
            # injected founders start NEW lineages: one per distinct genome in this injection (so 2000 copies of a
            # seed are one lineage, while random genomes are one lineage each)
            cnt = self.state.counters.numpy()
            ent["lineage"] = (int(cnt[C_LINEAGE]) + gidx).astype(np.int32)
            cnt[C_LINEAGE] += len(uniq)
            self.state.counters.assign(cnt)
        slots = spawn.spawn_cells(self, ent)
        self.interv_counts["inject"] = self.interv_counts.get("inject", 0) + len(slots)
        # genomes uploaded but unused (capacity) are collected by GC (refcount 0, in_use 1)
        return slots

    def host_cells(self, fields_=("alive", "pos", "angle", "mass", "energy", "lineage_id", "organism_id",
                                  "genome_id", "uid", "division_depth", "age", "origin")):
        c = self.state.cells
        return {f: getattr(c, f).numpy() for f in fields_}

    def clone_cells(self, src_slots, rs, mutate_mult=0.0, placement="near", offset=20.0, unit="cell"):
        """Clone cells (unit='cell') or whole bonded organisms (unit='organism')."""
        if len(src_slots) == 0:
            return np.zeros(0, np.int32)
        if unit == "organism":
            out = []
            for s in src_slots:
                out.extend(self.clone_organism(int(s), rs, mutate_mult=mutate_mult))
            return np.array(out, np.int32)
        c = self.state.cells
        src = np.asarray(src_slots, np.int32)
        H = {f: getattr(c, f).numpy()[src] for f in ("pos", "vel", "angle", "mass", "energy", "genome_id", "regs",
                                                     "state", "ip", "traits", "trait_targets", "lineage_id", "uid",
                                                     "division_depth")}
        pos = []
        keep = []
        for k, p0 in enumerate(H["pos"]):
            pp, _ = spawn.place_positions(self, 1, rs, mode="near" if placement == "near" else "uniform",
                                          center=p0, radius=offset)
            if len(pp):
                pos.append(pp[0])
                keep.append(k)
        if not keep:
            return np.zeros(0, np.int32)
        keep = np.array(keep)
        gids = H["genome_id"][keep]
        if mutate_mult > 0:
            gids = spawn.mutate_genomes(self, gids, mutate_mult)
        ent = dict(pos=np.array(pos, np.float32), vel=H["vel"][keep], angle=H["angle"][keep], mass=H["mass"][keep],
                   energy=H["energy"][keep], gid=gids, regs=H["regs"][keep], state=H["state"][keep],
                   ip=H["ip"][keep], traits=H["traits"][keep], targets=H["trait_targets"][keep],
                   lineage=H["lineage_id"][keep], parent_uid=H["uid"][keep], depth=H["division_depth"][keep],
                   origin=np.full(len(keep), isa.ORIGIN_CLONED, np.int32))
        slots = spawn.spawn_cells(self, ent)
        self.interv_counts["clone"] = self.interv_counts.get("clone", 0) + len(slots)
        return slots

    def organism_members(self, slot):
        c = self.state.cells
        org = c.organism_id.numpy()
        alive = c.alive.numpy() != 0
        # organism ids may be stale between labelling intervals: relabel now (cheap, host-triggered)
        organisms.label(self.state)
        org = c.organism_id.numpy()
        return np.nonzero(alive & (org == org[slot]))[0]

    def clone_organism(self, slot, rs, mutate_mult=0.0, max_tries=8):
        """Copy a whole bonded component preserving relative positions, orientations and bond geometry."""
        st = self.state
        c, bd = st.cells, st.bonds
        mem = self.organism_members(slot)
        if len(mem) == 0:
            return []
        W, H = st.dims.world
        P = c.pos.numpy()[mem].astype(np.float64)
        ref = P[0].copy()
        rel = P - ref
        rel -= np.array([W, H]) * np.round(rel / np.array([W, H]))
        ext = float(np.max(np.linalg.norm(rel, axis=1))) + 5.0
        alive = c.alive.numpy() != 0
        Pall = c.pos.numpy()[alive]
        target = None
        for _ in range(max_tries):
            t = np.array([rs.random() * W, rs.random() * H])
            d = Pall - t
            d -= np.array([W, H]) * np.round(d / np.array([W, H]))
            if len(d) == 0 or np.min(np.linalg.norm(d, axis=1)) > ext * 1.5:
                target = t
                break
        if target is None:
            self.interv_counts["clone_skipped"] = self.interv_counts.get("clone_skipped", 0) + 1
            return []
        H_ = {f: getattr(c, f).numpy()[mem] for f in ("vel", "angle", "mass", "energy", "genome_id", "regs", "state",
                                                     "ip", "traits", "trait_targets", "lineage_id", "uid",
                                                     "division_depth", "bonds")}
        gids = H_["genome_id"]
        if mutate_mult > 0:
            gids = spawn.mutate_genomes(self, gids, mutate_mult)
        ent = dict(pos=np.mod(target + rel, [W, H]).astype(np.float32), vel=H_["vel"], angle=H_["angle"],
                   mass=H_["mass"], energy=H_["energy"], gid=gids, regs=H_["regs"], state=H_["state"], ip=H_["ip"],
                   traits=H_["traits"], targets=H_["trait_targets"], lineage=H_["lineage_id"], parent_uid=H_["uid"],
                   depth=H_["division_depth"], origin=np.full(len(mem), isa.ORIGIN_CLONED, np.int32))
        slots = spawn.spawn_cells(self, ent)
        if len(slots) < len(mem):
            return list(slots)
        remap = {int(m): int(s) for m, s in zip(mem, slots)}
        ba, bb = bd.a.numpy(), bd.b.numpy()
        sa, sb = bd.slot_a.numpy(), bd.slot_b.numpy()
        rl, raa, rbb, strg = bd.rest_len.numpy(), bd.rest_ang_a.numpy(), bd.rest_ang_b.numpy(), bd.strength.numpy()
        seen = set()
        newb = []
        for row in H_["bonds"]:
            for b in row:
                if b >= 0 and b not in seen:
                    seen.add(int(b))
                    if int(ba[b]) in remap and int(bb[b]) in remap:
                        newb.append((remap[int(ba[b])], remap[int(bb[b])], int(sa[b]), int(sb[b]), float(rl[b]),
                                     float(raa[b]), float(rbb[b]), float(strg[b])))
        spawn.insert_bonds(self, newb)
        self.interv_counts["clone_org"] = self.interv_counts.get("clone_org", 0) + 1
        return list(slots)

    def kill(self, slots):
        spawn.cull(self, slots)
        self.interv_counts["cull"] = self.interv_counts.get("cull", 0) + len(slots)

    def restart(self):
        """Fresh seeding with the same config (tick keeps counting; the genome archive is kept by the runner)."""
        self.state = SimState(self.cfg, self.device)
        self.traced = {}
        self.pending_interv = {"mass": 0.0, "energy": 0.0, "field": 0.0}
        self.last_ledger = None
        self._p_version = -1
        self._init_world()
        self.seed_initial()

    # ================================================================== tracing
    def set_traced(self, slots):
        """Trace up to TRACE_SLOTS cells."""
        c = self.state.cells
        tr = np.full(self.state.dims.N, -1, np.int32)
        self.traced = {}
        for k, s in enumerate(list(slots)[:TRACE_SLOTS]):
            tr[int(s)] = k
            self.traced[int(s)] = k
        c.traced.assign(tr)
        self.state.trace.head.zero_()
        self.state.trace.hit.zero_()

    def read_trace(self, slot):
        k = self.traced.get(int(slot))
        if k is None:
            return []
        t = self.state.trace
        head = int(t.head.numpy()[k])
        n = min(head, t.tick.shape[1])
        L = t.tick.shape[1]
        idx = [(head - n + q) % L for q in range(n)]
        ticks, ips, words, regs = t.tick.numpy()[k], t.ip.numpy()[k], t.word.numpy()[k], t.regs.numpy()[k]
        return [(int(ticks[q]), int(ips[q]), int(words[q]) & 0xFFFFFFFF, regs[q].copy()) for q in idx]

    # ================================================================== params
    def params(self, ledger_on=False):
        cfg = self.cfg
        p = self._p
        d = self.state.dims
        if self._p_version != cfg.version:
            fill_struct(cfg, p)
            hc, hnx, hny = spatial_hash.choose_bucket(cfg, d.world)
            p.d_hash_cell = hc
            p.d_hash_nx = hnx
            p.d_hash_ny = hny
            p.d_world = wp.vec2(*d.world)
            p.d_field_nx, p.d_field_ny = d.fnx, d.fny
            p.d_light_nx, p.d_light_ny = d.lnx, d.lny
            p.d_num_sensors = d.NS
            p.d_amb_weight_avg = avg_cos_pow(cfg.receptor_p)
            self._p_version = cfg.version
        t = self.tick
        p.d_tick = wp.int64(t)
        p.d_key = wp.uint32(rng.tick_key(cfg.seed, cfg.rng_salt, t))
        ang = cfg.sun_angle + cfg.sun_drift * t + self.env["sun_angle_add"]
        p.d_sun_dir = wp.vec2(math.cos(ang), math.sin(ang))
        day = 1.0
        if cfg.day_period > 0:
            sv = math.sin(2 * math.pi * t / cfg.day_period)
            day = max(0.0, sv) if cfg.day_curve == "sin" else (1.0 if sv >= 0 else 0.0)
        p.d_sun_I = cfg.sun_I * day * self.env["sun_mult"]
        p.d_light_map_offset = (cfg.light_map_drift * t) % d.world[0]
        p.d_nutrient_offset = (cfg.nutrient_drift * t) % d.world[0]
        p.d_ledger_on = int(ledger_on)
        hz = self.env.get("hostile")
        if hz:
            p.d_hostile_cx, p.d_hostile_cy, p.d_hostile_r, p.d_hostile_mult = hz["x"], hz["y"], hz["r"], hz["mult"]
        else:
            p.d_hostile_r = 0.0
            p.d_hostile_mult = 1.0
        cur = self.env.get("current")
        if cur:
            p.current_x, p.current_y = cfg.current_x + cur[0], cfg.current_y + cur[1]
        # point lights (positions may move)
        nl = min(len(cfg.point_lights), cfg.max_point_lights, 16)
        p.d_num_point_lights = nl
        if nl:
            L = self._lights_np
            for k, pl in enumerate(cfg.point_lights[:nl]):
                L[k] = ((pl.get("x", 0) + pl.get("vx", 0) * t) % d.world[0],
                        (pl.get("y", 0) + pl.get("vy", 0) * t) % d.world[1], pl.get("I", 1.0))
            self.state.point_lights.assign(L)
        p.d_n_traced = len(self.traced)
        return p

    # ================================================================== tick
    def _sums(self):
        st = self.state
        alive = st.cells.alive.numpy() != 0
        E = float(st.cells.energy.numpy()[alive].astype(np.float64).sum())
        M = float(st.cells.mass.numpy()[alive].astype(np.float64).sum())
        area = st.dims.fh * st.dims.fh
        N = float(st.fields.nutrient.numpy().astype(np.float64).sum() * area)
        return E, M, N

    def step(self, n=1):
        for _ in range(n):
            self._step_one()

    def _step_one(self):
        cfg = self.cfg
        st = self.state
        dev = self.device
        ledger_on = cfg.ledger_interval > 0 and (self.tick % cfg.ledger_interval == 0)
        prof = cfg.profile_interval > 0 and (self.tick % cfg.profile_interval == 0)
        p = self.params(ledger_on)
        if ledger_on:
            before = self._sums()
            st.ledger.zero_()
            interv = dict(self.pending_interv)
        self.pending_interv = {"mass": 0.0, "energy": 0.0, "field": 0.0}
        t_last = [time.perf_counter()]

        def mark(name):
            if prof:
                wp.synchronize_device(dev)
                now = time.perf_counter()
                self.timings[name] = 0.9 * self.timings[name] + 0.1 * (now - t_last[0]) * 1000 \
                    if self.timing_samples else (now - t_last[0]) * 1000
                t_last[0] = now

        if prof:
            wp.synchronize_device(dev)
            t_last[0] = time.perf_counter()
        N = st.dims.N
        spatial_hash.build(st, p)
        mark("hash")
        wp.launch(light.rasterize_kernel, st.dims.lnx * st.dims.lny, [st.cells, st.hash, st.fields, p], device=dev)
        mark("opacity")
        wp.launch(light.cell_light_kernel, N, [st.cells, st.fields, st.scratch, st.point_lights, p], device=dev)
        mark("light")
        wp.copy(st.cells.signal_prev, st.cells.signal_out)
        sense.compute(st, p)
        mark("sense")
        wp.launch(vm_kernel, N, [st.cells, st.genomes, st.scratch, st.trace, p], device=dev)
        mark("vm")
        metabolism.compute(st, p)
        mark("metabolism")
        interactions.compute(st, p)
        mark("interactions")
        lifecycle.births(st, p)
        mark("births")
        lifecycle.deaths(st, p)
        mark("deaths")
        physics.step(st, p, cfg.substeps, cfg.dt)
        mark("physics")
        fields.update(st, p, cfg)
        mark("fields")
        if cfg.organism_label_interval > 0 and self.tick % cfg.organism_label_interval == 0:
            organisms.label(st)
        mark("organisms")
        self.tick += 1
        if cfg.stats_interval > 0 and self.tick % cfg.stats_interval == 0:
            self.sample_stats()
        mark("stats")
        if prof:
            self.timing_samples += 1
        if ledger_on:
            self._check_ledger(before, interv)

    # ================================================================== ledger
    def _check_ledger(self, before, interv):
        after = self._sums()
        st = self.state
        eb = st.scratch.eb.numpy().astype(np.float64).sum(axis=0)
        L = st.ledger.numpy()
        ins = eb[:EB_INFLOWS].sum()
        outs = eb[EB_INFLOWS:].sum()
        dE = after[0] - before[0]
        expE = ins - outs - L[L_DEATH_E] + L[L_INTERV_E]
        dM = after[1] - before[1]
        expM = L[L_GROWTH_M] + L[L_PRED_M] - L[L_VICTIM_M] - L[L_DEATH_M] + L[L_INTERV_M]
        dN = after[2] - before[2]
        expN = L[L_REGEN] - L[L_DECAY] - L[L_UPTAKE_F] + L[L_DETRITUS] + L[L_INTERV_F]
        scaleE = abs(before[0]) + ins + outs + 1e-6
        scaleM = abs(before[1]) + abs(L[L_GROWTH_M]) + abs(L[L_DEATH_M]) + 1e-6
        scaleN = abs(before[2]) + abs(L[L_REGEN]) + abs(L[L_UPTAKE_F]) + 1e-6
        errs = {"energy": (dE - expE) / scaleE, "mass": (dM - expM) / scaleM, "nutrient": (dN - expN) / scaleN}
        uptake_check = (eb[EB_UPTAKE] - self.cfg.absorb_eff * L[L_UPTAKE_F]) / (abs(eb[EB_UPTAKE]) + 1e-6)
        errs["uptake"] = uptake_check if abs(eb[EB_UPTAKE]) > 1e-9 else 0.0
        rep = {"tick": self.tick - 1, "dE": dE, "expected_dE": expE, "dM": dM, "expected_dM": expM, "dN": dN,
               "expected_dN": expN, "errors": errs,
               "cell_terms": {n: float(v) for n, v in zip(EB_NAMES, eb)},
               "global_terms": {n: float(v) for n, v in zip(LEDGER_NAMES, L)},
               "interventions_between_ticks": interv}
        self.last_ledger = rep
        bad = {k: v for k, v in errs.items() if abs(v) > self.cfg.ledger_tol}
        rep["ok"] = not bad
        if bad and self.cfg.ledger_check:
            self.ledger_failures += 1
            rep["message"] = f"ledger imbalance at tick {self.tick - 1}: " + ", ".join(
                f"{k} rel.err {v:.3g}" for k, v in bad.items())
            if self.events_hook:
                self.events_hook({"type": "ledger_failure", "severity": "error", **rep})
            raise LedgerError(rep)
        return rep

    # ================================================================== stats
    def counters(self):
        return self.state.counters.numpy()

    def sample_stats(self):
        st = self.state
        c = st.cells
        alive = c.alive.numpy() != 0
        pop = int(alive.sum())
        cnt = st.counters.numpy()
        now = time.perf_counter()
        if now > self._tps_t:
            self.tps = (self.tick - self._tps_tick) / (now - self._tps_t)
        self._tps_t, self._tps_tick = now, self.tick
        s = {"tick": self.tick, "wall": now - self._wall0, "tps": self.tps, "population": pop,
             "births": int(cnt[C_BIRTHS]), "deaths": int(cnt[C_DEATHS]),
             "split_fail_capacity": int(cnt[C_SPLIT_FAIL_CAP]), "v_clamps": int(cnt[C_VCLAMP]),
             "w_clamps": int(cnt[C_WCLAMP]), "bonds_broken": int(cnt[C_BONDS_BROKEN])}
        # reset accumulating counters
        cnt2 = cnt.copy()
        cnt2[C_SPLIT_FAIL_CAP] = cnt2[C_VCLAMP] = cnt2[C_WCLAMP] = cnt2[C_BONDS_BROKEN] = 0
        st.counters.assign(cnt2)
        if pop:
            mass = c.mass.numpy()[alive]
            energy = c.energy.numpy()[alive]
            traits = c.traits.numpy()[alive]
            lin = c.lineage_id.numpy()[alive]
            org = c.organism_id.numpy()[alive]
            gid = c.genome_id.numpy()[alive]
            origin = c.origin.numpy()[alive]
            depth = c.division_depth.numpy()[alive]
            glen = st.genomes.length.numpy()[gid]
            _, org_sizes = np.unique(org, return_counts=True)
            ul, lc = np.unique(lin, return_counts=True)
            pl = lc / pop
            s.update({
                "organisms": int((org_sizes >= 2).sum()), "components": int(len(org_sizes)),
                "org_size_mean": float(org_sizes.mean()), "org_size_max": int(org_sizes.max()),
                "biomass": float(mass.sum()), "energy_total": float(energy.sum()),
                "mean_mass": float(mass.mean()), "genome_len_mean": float(glen.mean()),
                "genome_len_max": int(glen.max()), "distinct_genomes": int(len(np.unique(gid))),
                "distinct_lineages": int(len(ul)), "lineage_shannon": float(-(pl * np.log(pl)).sum()),
                "dominant_lineage": int(ul[np.argmax(lc)]), "dominant_lineage_frac": float(lc.max() / pop),
                "max_depth": int(depth.max()),
            })
            for t, name in enumerate(isa.TRAIT_NAMES):
                s[f"trait_{name}"] = float(traits[:, t].mean())
            for k, name in enumerate(isa.ORIGIN_NAMES):
                s[f"origin_{name}"] = int((origin == k).sum())
        else:
            s.update({"organisms": 0, "components": 0, "org_size_mean": 0.0, "org_size_max": 0, "biomass": 0.0,
                      "energy_total": 0.0, "distinct_genomes": 0, "distinct_lineages": 0, "lineage_shannon": 0.0,
                      "dominant_lineage_frac": 0.0, "genome_len_mean": 0.0})
        area = st.dims.fh * st.dims.fh
        s["nutrient_total"] = float(st.fields.nutrient.numpy().astype(np.float64).sum() * area)
        s["genome_table_used"] = st.dims.G - int(cnt[C_GENOME_FREE])
        if self.last_ledger:
            for k, v in self.last_ledger["errors"].items():
                s[f"ledger_err_{k}"] = float(v)
        for k, v in self.timings.items():
            s[f"ms_{k}"] = float(v)
        for k, v in self.interv_counts.items():
            s[f"interv_{k}"] = v
        self.interv_counts = {}
        self.last_stats = s
        self.stats_history.append(s)
        return s

    def has_nan(self):
        c = self.state.cells
        alive = c.alive.numpy() != 0
        for name in ("pos", "vel", "angle", "ang_vel", "mass", "energy"):
            a = getattr(c, name).numpy()[alive]
            if not np.all(np.isfinite(a)):
                return name
        if not np.all(np.isfinite(c.regs.numpy()[alive])):
            return "regs"
        if not np.all(np.isfinite(self.state.fields.nutrient.numpy())):
            return "nutrient"
        return None

    def population(self):
        return int((self.state.cells.alive.numpy() != 0).sum())

    def state_hash(self):
        """Hash of the full dynamic state (alive cells + bonds + genomes + fields + counters)."""
        import hashlib
        h = hashlib.sha256()
        st = self.state
        c = st.cells
        alive = c.alive.numpy() != 0
        h.update(alive.tobytes())
        for name in ("pos", "vel", "angle", "ang_vel", "mass", "energy", "age", "genome_id", "ip", "wait", "regs",
                     "state", "traits", "trait_targets", "signal_out", "bonds", "uid", "lineage_id"):
            h.update(np.ascontiguousarray(getattr(c, name).numpy()[alive]).tobytes())
        h.update(st.bonds.alive.numpy().tobytes())
        h.update(st.fields.nutrient.numpy().tobytes())
        h.update(st.fields.chem.numpy().tobytes())
        h.update(st.genomes.hash.numpy().tobytes())
        h.update(st.counters.numpy()[:5].tobytes())
        return h.hexdigest()
