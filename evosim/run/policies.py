"""Population control policies (spec section 17).

Policies are evaluated on the host every `policy_eval_interval` ticks; their actions are applied
between ticks (before the next tick runs): culls are GPU kernels, injection/cloning go through the
host-assisted spawn path (placement search needs it). All randomness is seeded from
(seed, salt, tick, policy index, firing number), so runs with tick-based triggers replay exactly.
Wall-clock triggers are allowed but every firing is logged with its tick.
"""
from __future__ import annotations

import copy
import json
import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .. import rng
from ..genome import isa

TRIGGER_TYPES = ("population_below", "population_above", "extinction", "organisms_below", "diversity_below",
                 "dominance_above", "biomass_below", "nutrient_total_above", "stagnation", "interval", "at_tick",
                 "anomaly", "composite")
ACTION_TYPES = ("inject_random", "inject_seed", "inject_archive", "clone_random", "duplicate_population", "cull",
                "rollback", "restart", "set_param", "snapshot", "screenshot", "log_event", "pause", "stop_run")


def _num(v, capacity):
    """Numbers may be given as '95%cap' (fraction of MAX_CELLS)."""
    if isinstance(v, str) and v.endswith("%cap"):
        return float(v[:-4]) / 100.0 * capacity
    return float(v)


# ============================================================================= statistics (lazy)
class StatsView:
    """Lazily computed population statistics for one evaluation."""

    def __init__(self, sim, runner=None):
        self.sim = sim
        self.runner = runner
        self._c = {}

    def _arrays(self):
        if "arr" not in self._c:
            c = self.sim.state.cells
            al = np.nonzero(c.alive.numpy() != 0)[0]
            self._c["arr"] = dict(al=al, lin=c.lineage_id.numpy()[al], org=c.organism_id.numpy()[al],
                                  gid=c.genome_id.numpy()[al], mass=c.mass.numpy()[al],
                                  depth=c.division_depth.numpy()[al])
        return self._c["arr"]

    @property
    def capacity(self):
        return self.sim.state.dims.N

    @property
    def population(self):
        if "pop" not in self._c:
            from ..state import C_CELL_FREE
            self._c["pop"] = self.capacity - int(self.sim.state.counters.numpy()[C_CELL_FREE])
        return self._c["pop"]

    @property
    def nutrient_total(self):
        a = self.sim.state.fields.nutrient.numpy().astype(np.float64)
        return float(a.sum() * self.sim.state.dims.fh ** 2)

    @property
    def biomass(self):
        return float(self._arrays()["mass"].astype(np.float64).sum())

    def distinct_genomes(self):
        return int(len(np.unique(self._arrays()["gid"])))

    def genome_hash_counts(self, min_cells):
        a = self._arrays()
        u, c = np.unique(a["gid"], return_counts=True)
        h = self.sim.state.genomes.hash.numpy()[u[c >= min_cells]]
        return set(int(x) for x in h)

    def shannon(self):
        a = self._arrays()
        if len(a["lin"]) == 0:
            return 0.0
        _, c = np.unique(a["lin"], return_counts=True)
        p = c / c.sum()
        return float(-(p * np.log(p)).sum())

    def dominance(self, scope="lineage"):
        a = self._arrays()
        if len(a["lin"]) == 0:
            return 0.0, -1
        key = a["lin"] if scope == "lineage" else a["gid"]
        u, c = np.unique(key, return_counts=True)
        return float(c.max() / c.sum()), int(u[np.argmax(c)])

    def organisms(self, min_cells=2):
        a = self._arrays()
        if len(a["org"]) == 0:
            return 0
        _, c = np.unique(a["org"], return_counts=True)
        return int((c >= min_cells).sum())

    def metric(self, name):
        a = self._arrays()
        if name == "population":
            return float(self.population)
        if name == "org_size_max":
            return float(np.unique(a["org"], return_counts=True)[1].max()) if len(a["org"]) else 0.0
        if name == "org_size_mean":
            return float(np.unique(a["org"], return_counts=True)[1].mean()) if len(a["org"]) else 0.0
        if name == "genome_len_mean":
            g = self.sim.state.genomes.length.numpy()[a["gid"]]
            return float(g.mean()) if len(g) else 0.0
        if name == "distinct_genomes":
            return float(self.distinct_genomes())
        if name == "biomass":
            return self.biomass
        if name == "max_depth":
            return float(a["depth"].max()) if len(a["depth"]) else 0.0
        raise KeyError(f"unknown metric {name}")

    def anomaly(self, nan=True, ledger=True, clamp_rate=None):
        vals = {}
        hit = False
        if nan:
            bad = self.sim.has_nan()
            vals["nan"] = bad
            hit |= bad is not None
        if ledger:
            bad = self.sim.last_ledger is not None and not self.sim.last_ledger.get("ok", True)
            vals["ledger_failed"] = bool(bad)
            hit |= bool(bad)
        if clamp_rate is not None:
            from ..state import C_VCLAMP
            r = int(self.sim.state.counters.numpy()[C_VCLAMP]) / max(1, self.population)
            vals["v_clamp_rate"] = r
            hit |= r > clamp_rate
        return hit, vals


# ============================================================================= policy object
@dataclass
class Policy:
    name: str
    trigger: dict
    actions: list                         # list of action dicts (a single "action" is accepted too)
    enabled: bool = True
    cooldown: int = 0
    max_firings: int | None = None
    max_per_hour: int | None = None
    priority: int = 0
    exclusive: bool = False
    # runtime
    firings: int = 0
    last_tick: int = -10**18
    wall_times: list = field(default_factory=list)
    last_values: dict = field(default_factory=dict)
    last_fired_wall: float = 0.0

    def to_dict(self, runtime=False):
        d = {k: copy.deepcopy(v) for k, v in self.__dict__.items()
             if k not in ("firings", "last_tick", "wall_times", "last_values", "last_fired_wall")}
        if runtime:
            d.update(firings=self.firings, last_tick=self.last_tick, wall_times=list(self.wall_times))
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        if "action" in d:
            d["actions"] = [d.pop("action")]
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ============================================================================= engine
class PolicyEngine:
    def __init__(self, policies=None, eval_interval=10):
        self.policies: list[Policy] = list(policies or [])
        self.eval_interval = eval_interval
        self.tstate: dict[str, dict] = {}           # trigger runtime state (hysteresis, stagnation, ...)
        self.ramps: list[dict] = []                  # set_param ramps / timed overrides
        self.fire_log = deque(maxlen=500)

    # ---------------------------------------------------------------- config
    def add(self, policy: Policy):
        self.policies.append(policy)

    def set_policies(self, policies):
        self.policies = list(policies)
        self.tstate.clear()

    def to_dict(self, runtime=True):
        return {"eval_interval": self.eval_interval, "policies": [p.to_dict(runtime) for p in self.policies],
                "tstate": self.tstate if runtime else {}, "ramps": self.ramps if runtime else []}

    @classmethod
    def from_dict(cls, d):
        e = cls([Policy.from_dict(p) for p in d.get("policies", [])], d.get("eval_interval", 10))
        for p, pd in zip(e.policies, d.get("policies", [])):
            p.firings = pd.get("firings", 0)
            p.last_tick = pd.get("last_tick", -10**18)
            p.wall_times = list(pd.get("wall_times", []))
        e.tstate = d.get("tstate", {})
        e.ramps = d.get("ramps", [])
        return e

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(runtime=False), f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # ---------------------------------------------------------------- triggers
    def _eval(self, node, key, sim, view, now_wall):
        """Return (fired, values) for a trigger node. `key` addresses its runtime state."""
        t = node["type"]
        st = self.tstate.setdefault(key, {})
        cap = view.capacity
        tick = sim.tick

        def hysteresis(value, below, thr, rearm):
            cond = value < thr if below else value > thr
            if rearm is None:
                return cond
            armed = st.get("armed", True)
            if not armed and (value > rearm if below else value < rearm):
                armed = True
            fire = armed and cond
            if fire:
                armed = False
            st["armed"] = armed
            return fire

        if t == "population_below":
            v = view.population
            n = _num(node["n"], cap)
            r = node.get("rearm")
            return hysteresis(v, True, n, None if r is None else _num(r, cap)), {"population": v, "n": n}
        if t == "population_above":
            v = view.population
            n = _num(node["n"], cap)
            r = node.get("rearm")
            return hysteresis(v, False, n, None if r is None else _num(r, cap)), {"population": v, "n": n}
        if t == "extinction":
            return view.population == 0, {"population": view.population}
        if t == "organisms_below":
            v = view.organisms(node.get("min_cells", 2))
            return hysteresis(v, True, node["n"], node.get("rearm")), {"organisms": v, "n": node["n"]}
        if t == "diversity_below":
            if "shannon" in node:
                v = view.shannon()
                return hysteresis(v, True, node["shannon"], node.get("rearm")), {"shannon": v}
            v = view.distinct_genomes()
            return hysteresis(v, True, node["n"], node.get("rearm")), {"distinct_genomes": v}
        if t == "dominance_above":
            v, who = view.dominance(node.get("scope", "lineage"))
            return hysteresis(v, False, node["x"], node.get("rearm")), {"dominance": v, "who": who}
        if t == "biomass_below":
            v = view.biomass
            return hysteresis(v, True, node["x"], node.get("rearm")), {"biomass": v}
        if t == "nutrient_total_above":
            v = view.nutrient_total
            return hysteresis(v, False, node["x"], node.get("rearm")), {"nutrient_total": v}
        if t == "stagnation":
            T = node["ticks"]
            if node.get("metric", "new_genome") == "new_genome":
                seen = set(st.get("seen", []))
                now = view.genome_hash_counts(node.get("min_cells", 100))
                if "last" not in st or now - seen:
                    st["last"] = tick
                st["seen"] = sorted(seen | now)
                stalled = tick - st["last"]
            else:
                v = view.metric(node["metric"])
                if "best" not in st or v > st["best"] + node.get("min_delta", 0.0):
                    st["best"], st["last"] = v, tick
                stalled = tick - st["last"]
            fire = stalled >= T
            if fire:
                st["last"] = tick                          # re-arm after firing
            return fire, {"stalled_ticks": stalled}
        if t == "interval":
            if "every_ticks" in node:
                last = st.get("last", 0)
                if tick - last >= node["every_ticks"]:
                    st["last"] = tick
                    return True, {"every_ticks": node["every_ticks"]}
                return False, {}
            last = st.get("last_wall", now_wall)
            st.setdefault("last_wall", last)
            if now_wall - last >= node["every_seconds"]:
                st["last_wall"] = now_wall
                return True, {"every_seconds": node["every_seconds"], "wall_clock": True}
            return False, {}
        if t == "at_tick":
            if not st.get("done") and tick >= node["tick"]:
                st["done"] = True
                return True, {"tick": tick}
            return False, {}
        if t == "anomaly":
            return view.anomaly(node.get("nan", True), node.get("ledger", True), node.get("clamp_rate"))
        if t == "composite":
            res = [self._eval(ch, f"{key}/{i}", sim, view, now_wall) for i, ch in enumerate(node["children"])]
            vals = {f"c{i}": r[1] for i, r in enumerate(res)}
            fired = all(r[0] for r in res) if node.get("op", "and") == "and" else any(r[0] for r in res)
            return fired, vals
        raise ValueError(f"unknown trigger {t}")

    # ---------------------------------------------------------------- evaluation
    def due(self, tick):
        return self.eval_interval > 0 and tick % self.eval_interval == 0

    def evaluate(self, runner):
        sim = runner.sim
        tick = sim.tick
        now = time.time()
        view = StatsView(sim, runner)
        fired_names = []
        for idx, pol in sorted(enumerate(self.policies), key=lambda ip: -ip[1].priority):
            if not pol.enabled:
                continue
            fired, vals = self._eval(pol.trigger, f"{pol.name}#{idx}", sim, view, now)
            pol.last_values = vals
            if not fired:
                continue
            if tick - pol.last_tick < pol.cooldown:
                continue
            if pol.max_firings is not None and pol.firings >= pol.max_firings:
                continue
            pol.wall_times = [w for w in pol.wall_times if now - w < 3600.0]
            if pol.max_per_hour is not None and len(pol.wall_times) >= pol.max_per_hour:
                continue
            self.fire(pol, idx, runner, vals)
            fired_names.append(pol.name)
            if pol.exclusive:
                break
        return fired_names

    def fire(self, pol: Policy, idx: int, runner, values=None, manual=False):
        sim = runner.sim
        pol.firings += 1
        pol.last_tick = sim.tick
        pol.wall_times.append(time.time())
        pol.last_fired_wall = time.time()
        key = rng.run_key(sim.cfg.seed, sim.cfg.rng_salt)
        seed = (int(key) * 1000003 + sim.tick * 7919 + idx * 131 + pol.firings) % (2**63)
        rs = np.random.default_rng(seed)
        for act in pol.actions:
            run_action(act, runner, rs, self)
        rec = {"type": "policy_fired", "severity": "info", "policy": pol.name, "tick": sim.tick,
               "firing": pol.firings, "trigger_values": _plain(values or {}), "manual": manual,
               "actions": [a["type"] for a in pol.actions]}
        self.fire_log.append(rec)
        runner.emit(rec)

    # ---------------------------------------------------------------- set_param ramps
    def apply_ramps(self, sim, history=None):
        keep = []
        for r in self.ramps:
            t = sim.tick
            if t < r["start"]:
                keep.append(r)
                continue
            n = r["ramp"]
            s = 1.0 if n <= 0 else min(1.0, (t - r["start"]) / n)
            v = r["from"] + (r["to"] - r["from"]) * s
            if sim.cfg.get(r["name"]) != v:
                sim.cfg.set(r["name"], v)
            if r.get("duration") and t >= r["start"] + r["duration"] and not r.get("reverting"):
                r.update(reverting=True, start=t, **{"from": r["to"], "to": r["orig"], "ramp": 0, "duration": 0})
                keep.append(r)
            elif s < 1.0 or r.get("duration"):
                keep.append(r)
            if s >= 1.0 and history and not r.get("logged"):
                history(r["name"], v, "policy")
                r["logged"] = True
        self.ramps = keep


def _plain(o):
    return json.loads(json.dumps(o, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x)))


# ============================================================================= actions
def _alive(sim):
    return np.nonzero(sim.state.cells.alive.numpy() != 0)[0]


def _free_cap(sim):
    from ..state import C_CELL_FREE
    return int(sim.state.counters.numpy()[C_CELL_FREE])


def _choose(sim, selection, k, rs, candidates=None):
    al = _alive(sim) if candidates is None else candidates
    if len(al) == 0 or k <= 0:
        return np.zeros(0, np.int32)
    c = sim.state.cells
    k = min(k, len(al))
    if selection == "uniform" or selection == "random":
        return al[rs.choice(len(al), k, replace=False)]
    if selection == "energy_weighted":
        w = np.maximum(c.energy.numpy()[al].astype(np.float64), 1e-9)
        return al[rs.choice(len(al), k, replace=False, p=w / w.sum())]
    if selection == "top_energy":
        e = c.energy.numpy()[al]
        return al[np.argsort(-e)[:k]]
    if selection == "rarity_weighted":
        lin = c.lineage_id.numpy()[al]
        _, inv, cnt = np.unique(lin, return_inverse=True, return_counts=True)
        w = 1.0 / cnt[inv]
        return al[rs.choice(len(al), k, replace=False, p=w / w.sum())]
    if selection == "oldest":
        return al[np.argsort(c.birth_tick.numpy()[al])[:k]]
    if selection == "lowest_energy":
        return al[np.argsort(c.energy.numpy()[al])[:k]]
    if selection == "dominant_lineage":
        lin = c.lineage_id.numpy()[al]
        u, cn = np.unique(lin, return_counts=True)
        mem = al[lin == u[np.argmax(cn)]]
        return mem[rs.choice(len(mem), min(k, len(mem)), replace=False)]
    raise ValueError(f"unknown selection {selection}")


def _mut(a):
    m = a.get("mutate", False)
    return 0.0 if m is False else (1.0 if m is True else float(m))


def run_action(a: dict, runner, rs, engine: PolicyEngine):
    sim = runner.sim
    t = a["type"]
    cap = sim.state.dims.N
    place = a.get("placement", "uniform")
    region = runner.regions.get(a["region"]) if a.get("region") else None
    kw = dict(placement="region" if region else place, region=region)
    if t == "inject_random":
        k = int(_num(a.get("k", 100), cap))
        lo, hi = a.get("len_range", [16, 64])
        weights = a.get("opcode_weights")
        mode = "weighted" if a.get("mode") == "weighted" else "uniform"
        gens = [sim.random_genome(rs, int(rs.integers(lo, hi + 1)), mode, weights) for _ in range(k)]
        sim.inject(k, gens, rs, origin=isa.ORIGIN_INJECTED, one_genome_per_cell=True, mass=a.get("mass"),
                   energy=a.get("energy"), traits=a.get("traits"), **kw)
    elif t == "inject_seed":
        files = a["files"]
        weights = np.asarray(a.get("weights") or [1] * len(files), float)
        k = int(_num(a.get("k", 100), cap))
        gens = [sim.load_genome_file(f) for f in files]
        pick = rs.choice(len(files), size=k, p=weights / weights.sum())
        for gi in np.unique(pick):
            sim.inject(int((pick == gi).sum()), [gens[gi]], rs, origin=isa.ORIGIN_INJECTED, mass=a.get("mass"),
                       energy=a.get("energy"), traits=a.get("traits"), **kw)
    elif t == "inject_archive":
        if runner.archive is None:
            return
        k = int(_num(a.get("k", 100), cap))
        gens = runner.archive.select(a.get("selection", "random"), k, rs)
        if gens:
            sim.inject(k, gens, rs, origin=isa.ORIGIN_INJECTED, mass=a.get("mass"), energy=a.get("energy"), **kw)
    elif t == "clone_random":
        k = int(_num(a.get("k", 1), cap))
        src = _choose(sim, a.get("selection", "uniform"), k, rs)
        sim.clone_cells(src, rs, mutate_mult=_mut(a), placement=a.get("placement", "near"),
                        offset=a.get("offset", 20.0), unit=a.get("unit", "cell"))
    elif t == "duplicate_population":
        al = _alive(sim)
        k = min(int(round(a.get("fraction", 1.0) * len(al))), _free_cap(sim))
        src = _choose(sim, "uniform", k, rs)
        sim.clone_cells(src, rs, mutate_mult=_mut(a), placement=a.get("placement", "near"),
                        offset=a.get("offset", 20.0), unit=a.get("unit", "cell"))
    elif t == "cull":
        al = _alive(sim)
        if "region" in a and a["region"]:
            reg = runner.regions[a["region"]]
            pos = sim.state.cells.pos.numpy()[al]
            al = al[reg.contains(pos, sim.state.dims.world)]
        n_now = len(al)
        if "reduce_to" in a:
            k = max(0, len(_alive(sim)) - int(_num(a["reduce_to"], cap)))
        elif "fraction" in a:
            k = int(round(a["fraction"] * n_now))
        else:
            k = int(_num(a.get("k", 0), cap))
        sel = a.get("selection", "random")
        sim.kill(_choose(sim, sel, k, rs, candidates=al) if sel != "region" else al[:k])
    elif t == "rollback":
        runner.rollback(a.get("n", 1), tweaks=a.get("tweaks"), reason="policy")
    elif t == "restart":
        runner.restart(reason="policy")
    elif t == "set_param":
        name = a["name"]
        cur = sim.cfg.get(name)
        to = a["value"] if "value" in a else cur * a["mult"]
        engine.ramps = [r for r in engine.ramps if r["name"] != name or r.get("reverting")]
        engine.ramps.append({"name": name, "start": sim.tick, "ramp": int(a.get("ramp_ticks", 0)), "from": cur,
                             "to": to, "orig": cur, "duration": int(a.get("duration", 0))})
        engine.apply_ramps(sim, runner.log_param)
    elif t == "snapshot":
        runner.write_checkpoint(reason="policy")
    elif t == "screenshot":
        runner.request_screenshot()
    elif t == "log_event":
        runner.emit({"type": "policy_log", "severity": a.get("severity", "info"), "message": a.get("message", ""),
                     "tick": sim.tick})
    elif t == "pause":
        runner.request_pause("policy")
    elif t == "stop_run":
        runner.request_stop("policy")
    else:
        raise ValueError(f"unknown action {t}")


# ============================================================================= built-in presets (17.4)
def builtin(name: str, target: int = 20000, **kw) -> list[Policy]:
    if name == "none":
        return []
    if name == "keep_alive_basic":
        return [Policy("keep_alive", {"type": "population_below", "n": 1000},
                       [{"type": "inject_seed", "k": 500, "files": kw.get("files", ["seeds/seed_replicator.asm"])}],
                       cooldown=2000)]
    if name == "fill_to_target":
        return [Policy("fill_to_target", {"type": "population_below", "n": target},
                       [{"type": "clone_random", "k": 1, "selection": "rarity_weighted", "unit": "organism"}],
                       cooldown=1)]
    if name == "diversity_guard":
        return [Policy("diversity_guard", {"type": "dominance_above", "x": 0.8, "scope": "lineage"},
                       [{"type": "cull", "fraction": 0.3, "selection": "dominant_lineage"},
                        {"type": "inject_archive", "k": 200, "selection": "random"}], cooldown=20000)]
    if name == "extinction_rollback":
        return [Policy("extinction_rollback", {"type": "extinction"},
                       [{"type": "rollback", "n": 1},
                        {"type": "set_param", "name": "nutrient_regen", "mult": 1.5, "duration": 10000}],
                       max_per_hour=3, priority=10, exclusive=True),
                Policy("extinction_restart", {"type": "extinction"}, [{"type": "restart"}], priority=5)]
    if name == "overpop_cap":
        return [Policy("overpop_cap", {"type": "population_above", "n": "95%cap"},
                       [{"type": "cull", "reduce_to": "85%cap", "selection": "random"}])]
    raise KeyError(name)


BUILTIN_NAMES = ("none", "keep_alive_basic", "fill_to_target", "diversity_guard", "extinction_rollback",
                 "overpop_cap")


def combine(*names, **kw) -> list[Policy]:
    out = []
    for n in names:
        out.extend(builtin(n, **kw))
    return out
