import json
import os
import time

import numpy as np
import pytest

from evosim.config import Config
from evosim.io import snapshot
from evosim.run import env as envmod
from evosim.run.policies import Policy, PolicyEngine, builtin, combine
from evosim.run.rundir import CheckpointManager, Retention, RunDir
from evosim.run.runner import Runner, Watchdog, WatchdogConfig
from evosim.sim import Sim
from helpers import cellarr


def small_cfg(**kw):
    base = dict(max_cells=2048, world_w=128.0, world_h=128.0, initial_count=40, seed=5, ledger_interval=25,
                initial_seed_files="seeds/seed_replicator.asm", stats_interval=50)
    base.update(kw)
    return Config(**base)


def mk_runner(tmp_path, device, cfg=None, **kw):
    kw.setdefault("hardware_stats", False)
    kw.setdefault("stats_format", "csv")
    return Runner.create(cfg or small_cfg(), root=str(tmp_path), device=device, **kw)


# ----------------------------------------------------------------------------- schedule
def test_schedule_keyframes_and_modulators_deterministic():
    sc = envmod.Schedule()
    sc.add_keyframe("sun_I", 100, 0.2, ramp=0)
    sc.add_keyframe("sun_I", 200, 1.0, ramp=100, kind="linear")
    sc.add_keyframe("nutrient_cap", 50, 0.01, ramp=100, kind="smooth")
    sc.add_modulator("ambient_I", wave="sine", period=100, amplitude=0.5)
    sc.add_modulator("ambient_I", wave="square", period=40, amplitude=0.1)
    cfg = Config(sun_I=1.0, nutrient_cap=0.05, ambient_I=0.2)

    def vals(ts):
        return [tuple(round(v, 9) for v in sc.values_at(t, cfg).values()) for t in ts]
    ts = [0, 49, 50, 99, 100, 150, 199, 200, 250, 300, 400, 12345]
    a = vals(ts)
    b = vals(ts)
    assert a == b
    v = lambda t: sc.values_at(t, cfg)
    assert v(0)["sun_I"] == 1.0 and v(100)["sun_I"] == 0.2 and v(150)["sun_I"] == 0.2
    assert v(250)["sun_I"] == pytest.approx(0.6) and v(300)["sun_I"] == 1.0 and v(5000)["sun_I"] == 1.0
    assert v(50)["nutrient_cap"] == pytest.approx(0.05) and v(150)["nutrient_cap"] == pytest.approx(0.01)
    assert 0.01 < v(100)["nutrient_cap"] < 0.05                       # smooth ramp in between
    assert v(25)["ambient_I"] == pytest.approx(0.2 + 0.5 - 0.1, abs=1e-9)      # sine at its peak, square in its negative half
    # serialisation round trip keeps values
    sc2 = envmod.Schedule.from_dict(json.loads(json.dumps(sc.to_dict())))
    assert [tuple(sc2.values_at(t, cfg).values()) for t in ts] == [tuple(sc.values_at(t, cfg).values()) for t in ts]


def test_schedule_warmup_example_and_apply(device, tmp_path):
    """Spec example: warm-up with predation off and rich nutrients, then predation on and nutrients ramp down."""
    sc = envmod.Schedule()
    sc.add_keyframe("predation_enabled", 0, 0)
    sc.add_keyframe("predation_enabled", 100, 1)
    sc.add_keyframe("nutrient_regen", 100, 0.0005, ramp=200)
    sim = Sim(small_cfg(initial_count=5, nutrient_regen=0.005, ledger_interval=0), device=device)
    for _ in range(50):
        sc.apply(sim)
        sim.step(1)
    assert sim.cfg.predation_enabled is False
    sim.tick = 100
    sc.apply(sim)
    assert sim.cfg.predation_enabled is True
    sim.tick = 200
    sc.apply(sim)
    assert 0.0005 < sim.cfg.nutrient_regen < 0.005


# ----------------------------------------------------------------------------- events
def test_random_events_are_deterministic_and_logged(device):
    def run():
        sim = Sim(small_cfg(initial_count=30, ledger_interval=0), device=device)
        es = envmod.EventSystem()
        for t in ("catastrophe", "nutrient_bloom", "eclipse", "light_shift", "current", "hostile_zone"):
            es.configs[t].enabled = True
            es.configs[t].mean_interval = 40
            es.configs[t].duration = 20
        log = []
        for _ in range(300):
            es.step(sim, log.append)
            sim.step(1)
        return [(e["event"], e["tick"], round(e["x"], 3), round(e["magnitude"], 4)) for e in log
                if e["type"] == "environment_event"], sim.state_hash()
    a, b = run(), run()
    assert a == b and len(a[0]) >= 4
    assert {e[0] for e in a[0]} >= {"catastrophe"} or len(a[0]) > 3


def test_event_effects(device):
    sim = Sim(small_cfg(initial_count=200, max_cells=2048, ledger_interval=1), device=device)
    es = envmod.EventSystem()
    log = []
    n0 = sim.population()
    es.trigger(sim, "catastrophe", log.append, at=(64.0, 64.0), mag=1.0)
    es.configs["catastrophe"].size_lo = es.configs["catastrophe"].size_hi = 1000.0
    es.trigger(sim, "catastrophe", log.append, at=(64.0, 64.0), mag=1.0)
    sim.step(2)
    assert sim.population() < n0 * 0.1 and sim.last_ledger["ok"]            # culls are booked as interventions
    es.trigger(sim, "eclipse", log.append, mag=0.0)
    assert sim.env["sun_mult"] == 0.0
    es.configs["eclipse"].duration = 5
    es.active[0]["end"] = sim.tick + 5
    for _ in range(6):
        es.step(sim, log.append)
        sim.step(1)
    assert sim.env["sun_mult"] == 1.0
    before = float(sim.state.fields.nutrient.numpy().sum())
    es.trigger(sim, "nutrient_bloom", log.append, at=(30.0, 30.0), mag=0.2)
    sim.step(1)
    assert sim.last_ledger["ok"]


# ----------------------------------------------------------------------------- policies
class FakeSim:
    """Minimal stand-in so triggers/cooldowns can be tested without a GPU."""
    def __init__(self):
        self.tick = 0
        self.fired = 0


def _engine_with(pol, device, pop=40, **cfgkw):
    sim = Sim(small_cfg(initial_count=pop, ledger_interval=0, **cfgkw), device=device)
    eng = PolicyEngine([pol], eval_interval=1)

    class R:                                                        # runner stand-in
        regions = {}
        archive = None
        events = []
        def __init__(s): s.sim = sim; s.checkpoints = 0
        def emit(s, ev): s.events.append(ev)
        def log_param(s, *a): pass
        def write_checkpoint(s, **k): s.checkpoints += 1
        def request_pause(s, w=""): s.paused = w
        def request_stop(s, w=""): s.stopped = w
        def rollback(s, *a, **k): s.rolled = True
        def restart(s, **k): s.restarted = True
        def request_screenshot(s): s.shot = True
    return sim, eng, R()


def test_policy_population_below_with_cooldown(device):
    pol = Policy("keep", {"type": "population_below", "n": 100},
                 [{"type": "inject_seed", "k": 20, "files": ["seeds/seed_replicator.asm"]}], cooldown=5)
    sim, eng, r = _engine_with(pol, device, pop=40)
    fires = []
    for t in range(12):
        sim.tick = t
        fires.append(bool(eng.evaluate(r)))
    assert fires.count(True) == 3 and fires[0] and not fires[1] and fires[5] and fires[10]    # every 5 ticks
    assert sim.population() == 40 + 3 * 20
    assert r.events[0]["type"] == "policy_fired" and r.events[0]["trigger_values"]["population"] == 40


def test_policy_hysteresis_fires_once_until_rearmed(device):
    pol = Policy("h", {"type": "population_below", "n": 100, "rearm": 150},
                 [{"type": "log_event", "message": "x"}])
    sim, eng, r = _engine_with(pol, device, pop=40)
    seq = []
    for t in range(4):
        sim.tick = t
        seq.append(bool(eng.evaluate(r)))
    assert seq == [True, False, False, False]                      # no re-fire while still below
    rs = np.random.default_rng(0)
    sim.inject(120, [sim.load_genome_file("seeds/seed_replicator.asm")], rs)           # population 160 > rearm
    sim.tick = 10
    assert not eng.evaluate(r)
    sim.kill(np.nonzero(cellarr(sim, "alive"))[0][:100])
    sim.step(1)
    sim.tick = 11
    assert eng.evaluate(r)                                          # armed again, below again -> fires


def test_policy_triggers_and_composite(device):
    sim, eng, r = _engine_with(Policy("x", {"type": "extinction"}, []), device, pop=10)
    view = __import__("evosim.run.policies", fromlist=["StatsView"]).StatsView(sim)
    ev = lambda node: eng._eval(node, "k" + str(id(node)), sim, view, time.time())[0]
    assert not ev({"type": "extinction"})
    assert ev({"type": "population_above", "n": 5}) and not ev({"type": "population_above", "n": 50})
    assert ev({"type": "organisms_below", "n": 1})                       # no bonded organisms
    assert ev({"type": "diversity_below", "n": 100})
    assert ev({"type": "dominance_above", "x": 0.5})                    # a single seed lineage... per-cell lineages
    assert ev({"type": "biomass_below", "x": 1e9}) and not ev({"type": "biomass_below", "x": 1.0})
    assert ev({"type": "nutrient_total_above", "x": 1.0})
    assert ev({"type": "composite", "op": "and", "children": [{"type": "population_above", "n": 5},
                                                              {"type": "biomass_below", "x": 1e9}]})
    assert not ev({"type": "composite", "op": "and", "children": [{"type": "population_above", "n": 5},
                                                                  {"type": "extinction"}]})
    assert ev({"type": "composite", "op": "or", "children": [{"type": "extinction"},
                                                             {"type": "population_above", "n": 5}]})
    sim.tick = 50
    assert ev({"type": "at_tick", "tick": 50}) is True
    assert ev({"type": "at_tick", "tick": 50}) is False                      # one-shot (same state key reused below)
    node = {"type": "interval", "every_ticks": 10}
    key = "iv"
    hits = []
    for t in range(0, 35):
        sim.tick = t
        hits.append(eng._eval(node, key, sim, view, 0.0)[0])
    assert [i for i, h in enumerate(hits) if h] == [10, 20, 30]
    stag = {"type": "stagnation", "metric": "population", "ticks": 20}
    out = []
    for t in range(0, 70, 5):
        sim.tick = t
        out.append(eng._eval(stag, "stag", sim, view, 0.0)[0])
    assert any(out) and out[0] is False
    assert ev({"type": "anomaly", "nan": True, "ledger": False}) is False


def test_policy_actions_cull_clone_set_param(device):
    pol = Policy("cap", {"type": "population_above", "n": 30},
                 [{"type": "cull", "reduce_to": 20, "selection": "oldest"},
                  {"type": "set_param", "name": "nutrient_regen", "value": 0.02, "duration": 10}])
    sim, eng, r = _engine_with(pol, device, pop=60, nutrient_regen=0.001)
    sim.tick = 1
    eng.evaluate(r)
    sim.step(1)
    assert sim.population() == 20
    assert sim.cfg.nutrient_regen == pytest.approx(0.02)
    for _ in range(12):
        eng.apply_ramps(sim)
        sim.tick += 1
    assert sim.cfg.nutrient_regen == pytest.approx(0.001)                  # timed override reverted
    pol2 = Policy("dup", {"type": "interval", "every_ticks": 1}, [{"type": "duplicate_population", "fraction": 1.0}])
    sim2, eng2, r2 = _engine_with(pol2, device, pop=20)
    sim2.tick = 1
    eng2.evaluate(r2)
    assert sim2.population() == 40
    pol3 = Policy("clone", {"type": "interval", "every_ticks": 1},
                  [{"type": "clone_random", "k": 5, "selection": "rarity_weighted", "mutate": True}])
    sim3, eng3, r3 = _engine_with(pol3, device, pop=20)
    sim3.tick = 1
    eng3.evaluate(r3)
    assert sim3.population() == 25
    al = cellarr(sim3, "origin")[cellarr(sim3, "alive") != 0]
    assert (al == 2).sum() == 5                                            # origin=cloned
    sim3.step(1)


def test_policy_priority_exclusive_max_firings(device):
    a = Policy("a", {"type": "interval", "every_ticks": 1}, [{"type": "log_event", "message": "a"}], priority=10,
               exclusive=True, max_firings=2)
    b = Policy("b", {"type": "interval", "every_ticks": 1}, [{"type": "log_event", "message": "b"}], priority=1)
    sim, eng, r = _engine_with(a, device, pop=5)
    eng.add(b)
    names = []
    for t in range(1, 6):
        sim.tick = t
        names.append(tuple(eng.evaluate(r)))
    assert names[0] == ("a",) and names[1] == ("a",) and names[2] == ("b",)     # a exhausted -> b takes over


def test_policy_serialization_and_builtins():
    for n in ("none", "keep_alive_basic", "fill_to_target", "diversity_guard", "extinction_rollback", "overpop_cap"):
        pols = builtin(n)
        eng = PolicyEngine(pols)
        eng2 = PolicyEngine.from_dict(json.loads(json.dumps(eng.to_dict())))
        assert [p.to_dict() for p in eng2.policies] == [p.to_dict() for p in eng.policies]
    assert len(combine("keep_alive_basic", "overpop_cap")) == 2
    er = builtin("extinction_rollback")
    assert er[0].max_per_hour == 3 and er[0].exclusive and er[1].actions[0]["type"] == "restart"


# ----------------------------------------------------------------------------- checkpoints
def _fake_ckpts(rd, specs):
    """specs: (tick, age_seconds, good, keep). Writes tiny valid containers."""
    cm = CheckpointManager(rd)
    now = time.time()
    paths = []
    for tick, age, good, keep in specs:
        p = os.path.join(rd.ckpt_dir, f"ckpt_{tick:012d}_{tick % 10**9:09d}.evck")
        snapshot.write_container(p, b"S" + b"x" * 100, {"tick": tick, "wall_time": now - age, "good": good})
        if keep:
            cm.mark_keep(p)
        paths.append(p)
    return cm, now, paths


def test_retention_tiers(tmp_path):
    rd = RunDir(str(tmp_path / "r"))
    specs = [(t, 1800 - t, True, False) for t in range(1000)][:0]
    specs = []
    for i in range(30):                                   # 30 recent checkpoints, 1 min apart
        specs.append((10000 + i, (30 - i) * 60, True, False))
    for h in range(2, 30):                                # one per hour for 28 hours (older than the recent ones)
        specs.append((9000 - h, h * 3600 + 5, True, False))
        specs.append((9000 - h + 0.5 if False else 8000 - h, h * 3600 + 1800, True, False))
    for d in (2, 3, 4):                                   # several per day
        for k in range(3):
            specs.append((5000 - d * 10 - k, d * 86400 + k * 3600, True, False))
    specs.append((100, 40 * 86400, True, True))           # keep-forever
    cm, now, paths = _fake_ckpts(rd, specs)
    kept = cm.retained_set(cm.list(), now)
    items = {m["path"]: m for m in cm.list()}
    newest = sorted(items.values(), key=lambda m: -m["tick"])[:10]
    assert all(m["path"] in kept for m in newest)                         # last 10
    assert paths[-1] in kept                                              # keep-forever
    hourly = [m for m in items.values() if 3600 <= now - m["wall_time"] <= 24 * 3600 and m["path"] in kept]
    buckets = {int((now - m["wall_time"]) // 3600) for m in hourly}
    assert len(hourly) <= 24 and len(buckets) == len(hourly)              # at most one per hour bucket
    daily = [m for m in items.values() if now - m["wall_time"] > 24 * 3600 and m["path"] in kept and not m["keep"]]
    assert len({int((now - m["wall_time"]) // 86400) for m in daily}) == len(daily)
    assert len(kept) < len(items)
    removed = cm.prune(now)
    assert set(removed) == set(items) - kept
    assert all(os.path.exists(p) for p in kept)


def test_disk_budget_pruning_and_frames_disabled(tmp_path):
    rd = RunDir(str(tmp_path / "r"))
    now = time.time()
    cm = CheckpointManager(rd, Retention(last_n=100), max_bytes=4000)
    for i in range(12):
        p = os.path.join(rd.ckpt_dir, f"ckpt_{i:012d}_{i:09d}.evck")
        snapshot.write_container(p, b"S" + os.urandom(600), {"tick": i, "wall_time": now - (12 - i), "good": True})
    cm.mark_keep(os.path.join(rd.ckpt_dir, "ckpt_000000000000_000000000.evck"))
    open(os.path.join(rd.frames_dir, "f.png"), "wb").write(b"x" * 20000)
    acts = cm.enforce_budget()
    assert rd.size_bytes() > 0
    names = [os.path.basename(m["path"]) for m in cm.list()]
    assert "ckpt_000000000000_000000000.evck" in names                    # keep-forever survives
    assert any(n.startswith("ckpt_000000000011") for n in names)           # newest good survives
    assert "oldest_checkpoint" in acts and "frames_disabled" in acts
    assert not cm.frames_enabled and not os.listdir(rd.frames_dir)


def test_corrupt_latest_falls_back_to_previous(device, tmp_path):
    r = mk_runner(tmp_path, device, ckpt_every_ticks=0)
    r.sim.step(10)
    r.write_checkpoint(block=True)
    r.sim.step(10)
    r.write_checkpoint(block=True)
    latest = r.ckpt.list()[-1]["path"]
    data = bytearray(open(latest, "rb").read())
    data[-5] ^= 0xFF
    open(latest, "wb").write(bytes(data))
    sim, extra, meta = r.ckpt.load_latest_good(device=device)
    assert sim.tick == 10
    r.close("test")


# ----------------------------------------------------------------------------- runner: rollback / NaN
def test_nan_triggers_rollback_with_salt_change(device, tmp_path):
    wd = Watchdog(WatchdogConfig(interval=10, nan="rollback", stability_tweak="extra_substep"))
    r = mk_runner(tmp_path, device, ckpt_every_ticks=0, watchdog=wd)
    r.sim.step(20)
    r.write_checkpoint(block=True)
    assert r.ckpt.good() and r.ckpt.good()[-1]["tick"] == 20
    substeps0, salt0 = r.sim.cfg.substeps, r.sim.cfg.rng_salt
    r.advance(15)
    pos = cellarr(r.sim, "pos").copy()
    al = np.nonzero(cellarr(r.sim, "alive"))[0]
    pos[al[0]] = [np.nan, np.nan]                                   # inject a NaN into the state
    r.sim.state.cells.pos.assign(pos)
    r.advance(15)                                                   # watchdog (every 10 ticks) must notice
    ev = [e for e in r.elog.recent if e["type"] == "rollback"]
    assert ev and ev[0]["to_tick"] == 20
    assert r.sim.cfg.rng_salt > salt0 and r.sim.cfg.substeps == substeps0 + 1
    assert r.sim.has_nan() is None
    hist = [json.loads(l) for l in open(r.rd.history_path)]
    assert any(h["param"] == "rng_salt" and h["source"] == "rollback" for h in hist)    # salt change recorded
    assert any(h["param"] == "substeps" for h in hist)
    bad = [m for m in r.ckpt.list() if m.get("reason") == "before_rollback"]
    assert bad and not bad[0]["good"]                                # the failing state is kept, but never rolled to
    r.advance(30)
    assert r.sim.has_nan() is None
    r.close("test")


def test_rollback_limit_stops_run(device, tmp_path):
    wd = Watchdog(WatchdogConfig(interval=1, nan="rollback", max_rollbacks_per_hour=2))
    r = mk_runner(tmp_path, device, ckpt_every_ticks=0, watchdog=wd)
    r.sim.step(5)
    r.write_checkpoint(block=True)
    for _ in range(4):
        pos = cellarr(r.sim, "pos").copy()
        pos[np.nonzero(cellarr(r.sim, "alive"))[0][0]] = [np.nan, np.nan]
        r.sim.state.cells.pos.assign(pos)
        r.advance(2)
        if r.stop_requested:
            break
    assert r.stop_requested and "rollback limit" in r.stop_requested
    assert any(e["type"] == "rollback_limit" for e in r.elog.recent)
    r.close("test")


def test_runner_checkpoints_stats_status_and_resume(device, tmp_path):
    r = mk_runner(tmp_path, device, ckpt_every_ticks=40, stop={"max_ticks": 130}, status_every=0.0)
    r.engine = PolicyEngine(builtin("keep_alive_basic"), eval_interval=10)
    r.run(sleep_prevention=False)
    cks = r.ckpt.list()
    assert len(cks) >= 3 and cks[-1]["reason"] == "exit" and cks[-1]["tick"] >= 130
    st = json.load(open(r.rd.status_path))
    assert st["tick"] >= 130 and st["population"] is not None
    rows = __import__("evosim.run.rundir", fromlist=["StatsWriter"]).StatsWriter.read_all(r.rd.stats_dir)
    assert len(rows) >= 2 and "population" in rows[0]
    events = [json.loads(l) for l in open(r.rd.events_path)]
    assert {"run_start", "checkpoint_written", "run_stop"} <= {e["type"] for e in events}
    for f in ("config.json", "env.json", "policies.json", "schedule.json", "notes.md", "log.txt",
              "archive.sqlite"):
        assert os.path.exists(os.path.join(r.rd.path, f)), f
    pop, tick, h = r.sim.population(), r.sim.tick, r.sim.state_hash()
    r2 = Runner.resume(r.rd.path, device=device, overrides={"sun_I": 0.5}, hardware_stats=False, stats_format="csv")
    assert r2.sim.tick == tick and r2.sim.state_hash() == h              # resumed exactly from the exit checkpoint
    assert r2.sim.cfg.sun_I == 0.5
    hist = [json.loads(l) for l in open(r2.rd.history_path)]
    assert any(x["param"] == "sun_I" and x["source"] == "CLI override" for x in hist)
    assert [p.name for p in r2.engine.policies] == ["keep_alive"]
    r2.close("test")


def test_stop_conditions_and_extinction(device, tmp_path):
    r = mk_runner(tmp_path, device, cfg=small_cfg(initial_count=3, max_cells=1024, initial_mass=1.0,
                                                  initial_energy=0.001), ckpt_every_ticks=0,
                  stop={"stop_on_extinction": True, "extinction_grace_evals": 2})
    r.run(sleep_prevention=False)
    assert "extinction" in (r.stop_requested or "")
    r = mk_runner(tmp_path, device, ckpt_every_ticks=0,
                  stop={"metric": {"metric": "population", "op": ">=", "value": 60}, "metric_every": 10})
    r.run(sleep_prevention=False)
    assert "population" in (r.stop_requested or "") and r.sim.population() >= 60
