"""Environment control: schedule (keyframes + periodic modulators), random events, named regions.

Everything here is a deterministic function of (seed, salt, tick): schedule values depend only on
the tick, event times are drawn from the counter-based RNG (Poisson process), event geometry too.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from .. import rng
from ..config import PARAM_SPECS


# ============================================================================= regions
@dataclass
class Region:
    name: str
    shape: str = "rect"                    # rect | circle
    x: float = 0.0                         # rect: left/bottom corner; circle: centre
    y: float = 0.0
    w: float = 100.0                       # rect: size; circle: radius = w
    h: float = 100.0

    def contains(self, pos, world):
        pos = np.asarray(pos, np.float64)
        d = pos - self._anchor()
        d -= np.array(world) * np.round(d / np.array(world))
        if self.shape == "circle":
            return np.hypot(d[..., 0], d[..., 1]) <= self.w
        c = d - np.array([self.w / 2, self.h / 2])
        c -= np.array(world) * np.round(c / np.array(world))
        return (np.abs(c[..., 0]) <= self.w / 2) & (np.abs(c[..., 1]) <= self.h / 2)

    def _anchor(self):
        return np.array([self.x, self.y]) if self.shape == "rect" else np.array([self.x, self.y])

    def sample(self, rs):
        if self.shape == "circle":
            a, r = rs.random() * 2 * math.pi, self.w * math.sqrt(rs.random())
            return self.x + r * math.cos(a), self.y + r * math.sin(a)
        return self.x + rs.random() * self.w, self.y + rs.random() * self.h

    def to_dict(self):
        return dict(self.__dict__)


# ============================================================================= schedule
def _smooth(s):
    return s * s * (3 - 2 * s)


@dataclass
class Keyframe:
    tick: int
    value: float
    ramp: int = 0                          # ticks to reach value (0 = jump)
    kind: str = "linear"                   # linear | smooth


@dataclass
class Modulator:
    wave: str = "sine"                     # sine | square | triangle
    period: float = 1000.0
    amplitude: float = 0.1
    phase: float = 0.0                     # fraction of a period

    def value(self, tick):
        x = (tick / self.period + self.phase) % 1.0
        if self.wave == "sine":
            w = math.sin(2 * math.pi * x)
        elif self.wave == "square":
            w = 1.0 if x < 0.5 else -1.0
        else:
            w = (1.0 - 4.0 * abs(x - 0.25)) if x < 0.5 else (-1.0 + 4.0 * abs(x - 0.75))     # 0 -> 1 -> 0 -> -1 -> 0
        return self.amplitude * w


@dataclass
class ParamSchedule:
    param: str
    keyframes: list = field(default_factory=list)
    modulators: list = field(default_factory=list)
    initial: float | None = None           # value before the first keyframe (default: the config value at start)

    def base_value(self, tick, initial):
        cur = float(initial)
        for kf in sorted(self.keyframes, key=lambda k: k.tick):
            if tick < kf.tick:
                break
            if kf.ramp <= 0 or tick >= kf.tick + kf.ramp:
                cur = kf.value
            else:
                s = (tick - kf.tick) / kf.ramp
                if kf.kind == "smooth":
                    s = _smooth(s)
                return cur + (kf.value - cur) * s
        return cur

    def value(self, tick, initial):
        v = self.base_value(tick, self.initial if self.initial is not None else initial)
        for m in self.modulators:
            v += m.value(tick)
        return v


class Schedule:
    """Applies keyframes/modulators to live parameters each tick. The schedule is part of saved state."""

    def __init__(self, items=None):
        self.items: dict[str, ParamSchedule] = {}
        self.initial: dict[str, float] = {}
        for it in items or []:
            self.items[it.param] = it

    def add_keyframe(self, param, tick, value, ramp=0, kind="linear"):
        s = self.items.setdefault(param, ParamSchedule(param))
        s.keyframes.append(Keyframe(int(tick), float(value), int(ramp), kind))

    def add_modulator(self, param, wave="sine", period=1000.0, amplitude=0.1, phase=0.0):
        s = self.items.setdefault(param, ParamSchedule(param))
        s.modulators.append(Modulator(wave, float(period), float(amplitude), float(phase)))

    def values_at(self, tick, cfg):
        out = {}
        for p, s in self.items.items():
            if p not in self.initial:
                self.initial[p] = float(cfg.get(p))
            out[p] = s.value(tick, self.initial[p])
        return out

    def apply(self, sim, history=None):
        """Set every scheduled live parameter to its scheduled value for sim.tick. Returns changed names."""
        changed = []
        for p, v in self.values_at(sim.tick, sim.cfg).items():
            spec = PARAM_SPECS[p]
            if spec.kind == "int":
                v = int(round(v))
            if spec.lo is not None:
                v = min(max(v, spec.lo), spec.hi) if spec.hi is not None else max(v, spec.lo)
            cur = sim.cfg.get(p)
            if cur != v:
                sim.cfg.set(p, v)
                changed.append(p)
        if changed and history is not None:
            for kf_p in changed:
                s = self.items[kf_p]
                # log only at keyframe boundaries to keep config_history small
                if any(sim.tick in (k.tick, k.tick + k.ramp) for k in s.keyframes):
                    history(kf_p, sim.cfg.get(kf_p), "schedule")
        return changed

    def to_dict(self):
        return {"initial": self.initial,
                "items": [{"param": s.param, "initial": s.initial,
                           "keyframes": [k.__dict__ for k in s.keyframes],
                           "modulators": [m.__dict__ for m in s.modulators]} for s in self.items.values()]}

    @classmethod
    def from_dict(cls, d):
        sc = cls()
        sc.initial = dict(d.get("initial", {}))
        for it in d.get("items", []):
            ps = ParamSchedule(it["param"], [Keyframe(**k) for k in it.get("keyframes", [])],
                               [Modulator(**m) for m in it.get("modulators", [])], it.get("initial"))
            sc.items[ps.param] = ps
        return sc

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ============================================================================= random events
EVENT_TYPES = ("catastrophe", "nutrient_bloom", "hostile_zone", "eclipse", "light_shift", "current",
               "obstacle_change")


@dataclass
class EventConfig:
    enabled: bool = False
    mean_interval: float = 20000.0         # ticks (Poisson)
    mag_lo: float = 0.3                    # meaning depends on the type (see apply_event)
    mag_hi: float = 0.8
    size_lo: float = 60.0                  # region radius range
    size_hi: float = 200.0
    duration: int = 2000                   # for timed events


DEFAULT_EVENTS = {
    "catastrophe": EventConfig(mag_lo=0.3, mag_hi=0.9),                       # fraction of cells killed in region
    "nutrient_bloom": EventConfig(mag_lo=0.05, mag_hi=0.3),                   # concentration added in region
    "hostile_zone": EventConfig(mag_lo=3.0, mag_hi=10.0, duration=3000),      # basal cost multiplier
    "eclipse": EventConfig(mag_lo=0.0, mag_hi=0.3, duration=3000),            # sun intensity multiplier
    "light_shift": EventConfig(mag_lo=0.5, mag_hi=3.0),                       # sun angle jump (rad)
    "current": EventConfig(mag_lo=0.02, mag_hi=0.1, duration=2000),           # drift force per unit mass
    "obstacle_change": EventConfig(mag_lo=1.0, mag_hi=3.0, size_lo=15.0, size_hi=50.0),
}


class EventSystem:
    """Poisson event generator + appliers. Deterministic: draws use rng(run key, type id, P_EVENT, counter)."""

    def __init__(self, configs=None):
        self.configs = {k: EventConfig(**v.__dict__) for k, v in DEFAULT_EVENTS.items()}
        for k, v in (configs or {}).items():
            self.configs[k] = v if isinstance(v, EventConfig) else EventConfig(**v)
        self.counters = {k: 0 for k in EVENT_TYPES}
        self.next_tick = {}                # type -> tick of next event
        self.active = []                   # timed effects: dict(type, end, ...)
        self.log = []                      # last events, for the UI

    # -- determinism helpers
    @staticmethod
    def _u(sim, type_idx, counter, slot):
        key = rng.run_key(sim.cfg.seed, sim.cfg.rng_salt)
        return float(rng.np_rng_uniform(key, type_idx, rng.P_EVENT, counter * 16 + slot))

    def _schedule_next(self, sim, t, base_tick):
        cfg = self.configs[t]
        i = EVENT_TYPES.index(t)
        u = self._u(sim, i, self.counters[t], 0)
        self.next_tick[t] = base_tick + max(1, int(-cfg.mean_interval * math.log(max(u, 1e-9))))

    def step(self, sim, emit):
        """Called once per tick before sim.step. emit(event_dict) logs."""
        tick = sim.tick
        for t in EVENT_TYPES:
            cfg = self.configs[t]
            if not cfg.enabled:
                self.next_tick.pop(t, None)
                continue
            if t not in self.next_tick:
                self._schedule_next(sim, t, tick)
            if tick >= self.next_tick[t]:
                self.trigger(sim, t, emit, source="random")
                self.counters[t] += 1
                self._schedule_next(sim, t, tick)
        self._expire(sim, tick, emit)

    def trigger(self, sim, t, emit, source="manual", at=None, mag=None):
        """Apply event t now (also used by the UI at the cursor)."""
        cfg = self.configs[t]
        i = EVENT_TYPES.index(t)
        c = self.counters[t] if source == "random" else 10**6 + self.counters[t]
        u = lambda slot: self._u(sim, i, c, slot)
        W, H = sim.state.dims.world
        cx, cy = (at if at is not None else (u(1) * W, u(2) * H))
        r = cfg.size_lo + u(3) * (cfg.size_hi - cfg.size_lo)
        m = mag if mag is not None else cfg.mag_lo + u(4) * (cfg.mag_hi - cfg.mag_lo)
        ev = {"type": "environment_event", "event": t, "tick": sim.tick, "x": cx, "y": cy, "radius": r,
              "magnitude": m, "source": source, "severity": "info"}
        if t == "catastrophe":
            reg = Region("event", "circle", cx, cy, r)
            al = sim.state.cells.alive.numpy() != 0
            pos = sim.state.cells.pos.numpy()
            inside = np.nonzero(al & reg.contains(pos, (W, H)))[0]
            rs = np.random.default_rng(int(u(5) * 2**31))
            kill = inside[rs.random(len(inside)) < m]
            sim.kill(kill)
            ev["killed"] = int(len(kill))
        elif t == "nutrient_bloom":
            f = sim.state.fields
            fh = sim.state.dims.fh
            X = (np.arange(sim.state.dims.fnx) + 0.5) * fh
            Y = (np.arange(sim.state.dims.fny) + 0.5) * fh
            dx = X - cx; dx -= W * np.round(dx / W)
            dy = Y - cy; dy -= H * np.round(dy / H)
            mask = (dx[:, None] ** 2 + dy[None, :] ** 2) <= r * r
            arr = f.nutrient.numpy()
            arr[mask] += m
            f.nutrient.assign(arr)
            sim.book_intervention(field=float(m * mask.sum() * fh * fh))
        elif t == "hostile_zone":
            sim.env["hostile"] = {"x": cx, "y": cy, "r": r, "mult": m}
            self.active.append({"type": t, "end": sim.tick + cfg.duration})
        elif t == "eclipse":
            sim.env["sun_mult"] = m
            self.active.append({"type": t, "end": sim.tick + cfg.duration})
        elif t == "light_shift":
            sim.env["sun_angle_add"] = sim.env.get("sun_angle_add", 0.0) + (m if u(6) < 0.5 else -m)
        elif t == "current":
            a = u(5) * 2 * math.pi
            sim.env["current"] = [m * math.cos(a), m * math.sin(a)]
            self.active.append({"type": t, "end": sim.tick + cfg.duration})
        elif t == "obstacle_change":
            if sim.cfg.obstacles_enabled:
                f = sim.state.fields
                fh = sim.state.dims.fh
                X = (np.arange(sim.state.dims.fnx) + 0.5) * fh
                Y = (np.arange(sim.state.dims.fny) + 0.5) * fh
                dx = X - cx; dx -= W * np.round(dx / W)
                dy = Y - cy; dy -= H * np.round(dy / H)
                mask = (dx[:, None] ** 2 + dy[None, :] ** 2) <= r * r
                arr = f.obstacle.numpy()
                add = u(5) < 0.5
                arr[mask] = 1 if add else 0
                f.obstacle.assign(arr)
                if add:                                       # cells inside the new rock are removed
                    al = sim.state.cells.alive.numpy() != 0
                    pos = sim.state.cells.pos.numpy()
                    ix = (pos[:, 0] / fh).astype(int) % sim.state.dims.fnx
                    iy = (pos[:, 1] / fh).astype(int) % sim.state.dims.fny
                    sim.kill(np.nonzero(al & (arr[ix, iy] != 0))[0])
                ev["added"] = bool(add)
            else:
                ev["skipped"] = "obstacles disabled"
        self.log.append(ev)
        self.log = self.log[-200:]
        emit(ev)
        return ev

    def _expire(self, sim, tick, emit):
        keep = []
        for a in self.active:
            if tick >= a["end"]:
                t = a["type"]
                if t == "hostile_zone":
                    sim.env["hostile"] = None
                elif t == "eclipse":
                    sim.env["sun_mult"] = 1.0
                elif t == "current":
                    sim.env["current"] = None
                emit({"type": "environment_event_end", "event": t, "tick": tick, "severity": "info"})
            else:
                keep.append(a)
        self.active = keep

    def to_dict(self):
        return {"configs": {k: v.__dict__ for k, v in self.configs.items()}, "counters": self.counters,
                "next_tick": self.next_tick, "active": self.active}

    @classmethod
    def from_dict(cls, d):
        e = cls(d.get("configs"))
        e.counters = dict(d.get("counters", e.counters))
        e.next_tick = dict(d.get("next_tick", {}))
        e.active = list(d.get("active", []))
        return e
