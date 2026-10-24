"""Typed parameter set with metadata. The UI, JSON presets and the GPU parameter struct are
all generated from PARAM_SPECS, so adding a ParamSpec adds a slider and a kernel field.
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from typing import Any

import warp as wp

from .genome import isa

MAX_CHEM = 8          # upper bound for k_chem (per-channel params are generated for this many)


@dataclass(frozen=True)
class ParamSpec:
    name: str
    default: Any
    lo: float | None = None
    hi: float | None = None
    unit: str = ""
    group: str = "misc"
    desc: str = ""
    live: bool = True                 # False -> requires restart
    kind: str = "float"               # float | int | bool | enum | str
    choices: tuple = ()
    log: bool = False                 # logarithmic slider


def _p(*a, **k):
    return ParamSpec(*a, **k)


_SPECS: list[ParamSpec] = [
    # ------------------------------------------------------------------ world
    _p("world_w", 2048.0, 128, 16384, "u", "World", "world width (torus)", live=False),
    _p("world_h", 2048.0, 128, 16384, "u", "World", "world height (torus)", live=False),
    _p("field_cell", 4.0, 1, 32, "u", "World", "nutrient/chemical grid cell size", live=False),
    _p("light_cell", 2.0, 0.5, 16, "u", "World", "opacity/light grid cell size", live=False),
    _p("max_cells", 262144, 1024, 2097152, "", "World", "cell capacity MAX_CELLS", live=False, kind="int"),
    _p("seed", 1, 0, 2**31 - 1, "", "World", "global RNG seed", live=False, kind="int"),
    _p("rng_salt", 0, 0, 2**31 - 1, "", "World", "changed on rollback so failures don't replay", live=False, kind="int"),
    _p("obstacles_enabled", False, kind="bool", group="World", desc="static obstacle mask"),
    _p("obstacle_generator", "none", group="World", kind="enum",
       choices=("none", "rocks", "maze", "channels", "ring"), live=False, desc="obstacle generator"),
    # ------------------------------------------------------------------ time
    _p("dt", 1.0, 0.05, 4, "tick", "Time", "time step per tick"),
    _p("substeps", 4, 1, 16, "", "Time", "physics substeps per tick", kind="int"),
    _p("instr_per_tick", 16, 1, 256, "", "Time", "max instructions per cell per tick", kind="int"),
    _p("max_wait", 64, 0, 10000, "ticks", "Time", "clamp for WAIT", kind="int"),
    # ------------------------------------------------------------------ physics
    _p("density", 1.0, 0.1, 10, "m/u^2", "Physics", "mass density rho"),
    _p("drag_lin", 0.5, 0, 10, "", "Physics", "linear drag  F = -drag_lin*r*v"),
    _p("drag_quad", 0.0, 0, 10, "", "Physics", "quadratic drag F = -drag_quad*r*|v|v"),
    _p("ang_drag", 0.5, 0, 10, "", "Physics", "angular drag torque = -ang_drag*r^3*w"),
    _p("v_max", 4.0, 0.1, 50, "u/tick", "Physics", "speed clamp"),
    _p("w_max", 1.0, 0.01, 10, "rad/tick", "Physics", "angular speed clamp"),
    _p("k_contact", 10.0, 0, 200, "", "Physics", "contact stiffness"),
    _p("contact_exp", 1.5, 1, 3, "", "Physics", "contact force exponent p"),
    _p("contact_damp", 0.5, 0, 10, "", "Physics", "contact normal damping"),
    _p("armor_density", 0.5, 0, 4, "", "Physics", "ARMOR raises density by (1+x*ARMOR)"),
    _p("armor_drag", 1.0, 0, 10, "", "Physics", "ARMOR raises drag by (1+x*ARMOR)"),
    _p("obstacle_k", 20.0, 0, 200, "", "Physics", "obstacle contact stiffness"),
    _p("current_x", 0.0, -1, 1, "force/mass", "Physics", "uniform drift force (current event)"),
    _p("current_y", 0.0, -1, 1, "force/mass", "Physics", "uniform drift force (current event)"),
    # ------------------------------------------------------------------ bonds
    _p("max_bonds", 6, 1, 12, "", "Bonds", "bond slots per cell MAX_BONDS", live=False, kind="int"),
    _p("k_bond", 5.0, 0, 100, "", "Bonds", "bond spring stiffness"),
    _p("c_bond", 0.5, 0, 20, "", "Bonds", "bond spring damping"),
    _p("k_bond_angle", 1.0, 0, 20, "", "Bonds", "angular spring stiffness at both ends (x rest_len^2)"),
    _p("break_strain", 0.6, 0.01, 10, "", "Bonds", "strain at which a bond breaks (x(1+ADHESION))"),
    _p("bond_create_cost", 0.5, 0, 10, "E", "Bonds", "energy to create a bond"),
    _p("bond_upkeep", 0.002, 0, 0.1, "E/tick", "Bonds", "energy per bond end per tick"),
    _p("bond_consent_mode", "mutual", group="Bonds", kind="enum",
       choices=("unilateral", "mutual", "adhesion_contest"), desc="BOND consent rule"),
    _p("rest_length_tracks_radius", True, kind="bool", group="Bonds", desc="rest length follows current radii"),
    # ------------------------------------------------------------------ light
    _p("sun_I", 1.0, 0, 10, "", "Light", "sun intensity"),
    _p("sun_angle", 0.785, -math.pi, math.pi, "rad", "Light", "direction TO the sun"),
    _p("sun_drift", 0.0, -0.01, 0.01, "rad/tick", "Light", "sun angle drift"),
    _p("day_period", 0, 0, 1000000, "ticks", "Light", "day length (0 = no cycle)", kind="int"),
    _p("day_curve", "sin", group="Light", kind="enum", choices=("sin", "square"), desc="day intensity curve"),
    _p("ambient_I", 0.05, 0, 5, "", "Light", "uniform non-occludable light"),
    _p("k_water", 0.002, 0, 0.1, "1/u", "Light", "water absorption for point lights"),
    _p("shadow_ray_len", 48.0, 0, 512, "u", "Light", "sun shadow ray length"),
    _p("ray_steps", 16, 1, 128, "", "Light", "ray march steps", kind="int"),
    _p("base_opacity", 0.02, 0, 5, "1/u", "Light", "opacity of an unpigmented body"),
    _p("pigment_opacity", 1.0, 0, 20, "1/u", "Light", "extra opacity per unit PIGMENT"),
    _p("obstacle_opacity", 2.0, 0, 20, "1/u", "Light", "opacity of obstacles"),
    _p("light_map_mode", "none", group="Light", kind="enum", choices=("none", "bands", "patches"),
       desc="spatial modulation of sun"),
    _p("light_map_contrast", 0.5, 0, 1, "", "Light", "light map contrast"),
    _p("light_map_scale", 512.0, 16, 4096, "u", "Light", "light map feature size"),
    _p("light_map_drift", 0.0, -1, 1, "u/tick", "Light", "light map drift along x"),
    _p("receptor_p", 4.0, 0.5, 64, "", "Light", "photoreceptor directionality exponent p"),
    _p("photo_eff", 0.02, 0, 1, "E/(I*u^2)", "Light", "photosynthesis efficiency"),
    _p("point_light_r0", 64.0, 1, 1024, "u", "Light", "point light falloff radius r0"),
    _p("max_point_lights", 4, 0, 16, "", "Light", "point lights evaluated", live=False, kind="int"),
    # ------------------------------------------------------------------ nutrients
    _p("nutrient_pattern", "uniform", group="Nutrients", kind="enum", choices=("uniform", "patches", "vents"),
       live=False, desc="nutrient capacity pattern"),
    _p("nutrient_cap", 0.05, 0, 5, "m/u^2", "Nutrients", "regeneration target concentration"),
    _p("nutrient_regen", 0.001, 0, 0.5, "1/tick", "Nutrients", "relaxation rate toward cap"),
    _p("D_nutrient", 0.5, 0, 20, "u^2/tick", "Nutrients", "nutrient diffusion"),
    _p("nutrient_decay", 0.0, 0, 0.1, "1/tick", "Nutrients", "nutrient decay"),
    _p("absorb_rate", 0.2, 0, 5, "1/tick", "Nutrients", "uptake rate"),
    _p("absorb_eff", 1.0, 0, 5, "E/m", "Nutrients", "energy per unit nutrient absorbed"),
    _p("detritus_fraction", 0.5, 0, 1, "", "Nutrients", "fraction of a corpse's mass returned to the field"),
    _p("nutrient_drift", 0.0, -1, 1, "u/tick", "Nutrients", "pattern drift along x"),
    _p("pattern_scale", 256.0, 16, 4096, "u", "Nutrients", "patch / vent size"),
    _p("vent_count", 8, 1, 256, "", "Nutrients", "number of vents", live=False, kind="int"),
    # ------------------------------------------------------------------ chemicals
    _p("k_chem", 4, 1, MAX_CHEM, "", "Chemicals", "number of chemical channels", live=False, kind="int"),
    _p("emit_cost", 0.1, 0, 10, "E/unit", "Chemicals", "energy per unit emitted"),
    _p("passive_leak", 0.0, 0, 10, "", "Chemicals", "channel-0 leak per unit basal metabolism"),
] + [
    _p(f"D_chem_{k}", 1.0, 0, 20, "u^2/tick", "Chemicals", f"diffusion of channel {k}") for k in range(MAX_CHEM)
] + [
    _p(f"chem_decay_{k}", 0.01, 0, 1, "1/tick", "Chemicals", f"decay of channel {k}") for k in range(MAX_CHEM)
] + [
    # ------------------------------------------------------------------ metabolism
    _p("basal", 0.0005, 0, 0.05, "E/(m*tick)", "Metabolism", "basal cost per mass"),
    _p("growth_eff", 0.5, 0.01, 1, "m/E", "Metabolism", "mass gained per energy spent on growth"),
    _p("growth_rate", 0.01, 0, 0.5, "1/tick", "Metabolism", "max growth per tick, fraction of mass"),
    _p("growth_threshold", 0.5, 0, 1, "", "Metabolism", "grow only when energy fraction above this"),
    _p("max_mass", 40.0, 1, 1000, "m", "Metabolism", "maximum mass"),
    _p("min_mass", 1.0, 0.01, 100, "m", "Metabolism", "die below this mass"),
    _p("max_energy_per_mass", 2.0, 0.1, 100, "E/m", "Metabolism", "energy capacity per mass"),
    _p("max_age", 100000, 1, 10**9, "ticks", "Metabolism", "age limit", kind="int"),
    _p("max_age_enabled", False, kind="bool", group="Metabolism", desc="enable age limit"),
    _p("instr_energy_cost", 0.0005, 0, 0.05, "E/instr", "Metabolism", "energy per executed instruction"),
    _p("trait_change_rate", 0.02, 0, 1, "1/tick", "Metabolism", "max trait change per tick"),
    _p("trait_change_cost", 0.05, 0, 10, "E/(m*unit)", "Metabolism", "cost per unit trait change per mass"),
    _p("trait_budget", 3.0, 0, 9, "", "Metabolism", "sum-of-traits budget"),
    _p("trait_budget_enabled", False, kind="bool", group="Metabolism", desc="enforce trait budget"),
] + [
    _p(f"upkeep_{n}", d, 0, 0.05, "E/(m*tick)", "Metabolism", f"upkeep of trait {n}")
    for n, d in zip(isa.TRAIT_NAMES, (0.0002, 0.0002, 0.0003, 0.0004, 0.0003, 0.0002, 0.0002, 0.0002, 0.0))
] + [
    # ------------------------------------------------------------------ motion
    _p("thrust_cost", 0.01, 0, 1, "", "Motion", "thrust energy coefficient"),
    _p("thrust_exp", 2.0, 1, 3, "", "Motion", "thrust cost exponent"),
    _p("max_thrust", 2.0, 0, 50, "force", "Motion", "max thrust at MOTOR=1"),
    _p("motor_base", 0.1, 0, 1, "", "Motion", "thrust capacity at MOTOR=0 (fraction of max)"),
    _p("turn_cost", 0.005, 0, 1, "", "Motion", "energy per unit |torque|"),
    _p("max_torque", 2.0, 0, 50, "", "Motion", "max |TURN| torque"),
    # ------------------------------------------------------------------ reproduction
    _p("split_cost", 0.5, 0, 50, "E", "Reproduction", "energy cost of a split"),
    _p("min_split_mass", 6.0, 0.1, 500, "m", "Reproduction", "minimum mass to split"),
    _p("min_frac", 0.2, 0.01, 0.5, "", "Reproduction", "min daughter mass fraction"),
    _p("reg_inherit", "copy", group="Reproduction", kind="enum", choices=("copy", "zero"),
       desc="daughter register inheritance"),
    # ------------------------------------------------------------------ predation
    _p("predation_enabled", True, kind="bool", group="Predation", desc="global predation switch"),
    _p("bite_rate", 2.0, 0, 50, "m/tick", "Predation", "bite mass rate"),
    _p("bite_eff", 0.7, 0, 2, "E/m", "Predation", "energy per mass bitten"),
    _p("bite_mass_gain", 0.0, 0, 1, "", "Predation", "fraction of bitten mass the predator gains as mass"),
    _p("armor_factor", 0.8, 0, 1, "", "Predation", "bite reduction per unit target ARMOR"),
    _p("engulf_ratio", 4.0, 1, 100, "", "Predation", "engulf when m_pred > ratio * m_prey"),
    _p("can_eat_bonded", False, kind="bool", group="Predation", desc="bonded neighbours can be bitten"),
    _p("same_org_eating", False, kind="bool", group="Predation", desc="allow eating own organism"),
    # ------------------------------------------------------------------ signals
    _p("k_sig", 4, 1, 8, "", "Signals", "signal channels K_sig", live=False, kind="int"),
    _p("signal_cost", 0.001, 0, 1, "E", "Signals", "cost per SIGNAL"),
    _p("share_eff", 0.9, 0, 1, "", "Signals", "SHARE transfer efficiency"),
    _p("recv_mode", "mean", group="Signals", kind="enum", choices=("sum", "mean", "max"), desc="RECV aggregation"),
    _p("apoptosis_share", True, kind="bool", group="Signals", desc="DIE passes energy to bonded neighbours"),
    # ------------------------------------------------------------------ mutation
    _p("mu_point", 0.002, 0, 0.2, "1/word", "Mutation", "point mutation rate per word", log=True),
    _p("p_word_replace", 0.3, 0, 1, "", "Mutation", "fraction of point mutations replacing the word"),
    _p("imm_nudge_sigma", 0.25, 0, 10, "", "Mutation", "Gaussian nudge for SET immediates (0 = random byte)"),
    _p("mu_insert", 0.02, 0, 1, "1/copy", "Mutation", "insertion rate per copy", log=True),
    _p("mu_delete", 0.02, 0, 1, "1/copy", "Mutation", "deletion rate per copy", log=True),
    _p("mu_duplicate", 0.005, 0, 1, "1/copy", "Mutation", "segment duplication rate", log=True),
    _p("mu_invert", 0.002, 0, 1, "1/copy", "Mutation", "segment inversion rate", log=True),
    _p("mu_transpose", 0.002, 0, 1, "1/copy", "Mutation", "segment transposition rate", log=True),
    _p("dup_len_mean", 4.0, 1, 64, "words", "Mutation", "mean segment length (geometric)"),
    _p("somatic_rate_multiplier", 0.1, 0, 1, "", "Mutation", "rate multiplier for bonded splits"),
    _p("min_genome_len", 4, 1, 64, "words", "Mutation", "minimum genome length", kind="int"),
    _p("evolvable_mutation_rate", False, kind="bool", group="Mutation",
       desc="word 0 (if NOP) scales all rates by 2^((a-128)/32)"),
    # ------------------------------------------------------------------ sensing
    _p("sensor_noise", 0.01, 0, 1, "", "Sensing", "additive Gaussian sensor noise sigma"),
    _p("photoreceptor_gain", 4.0, 0, 100, "", "Sensing", "light noise /= (1 + gain*PHOTORECEPTOR)"),
    # ------------------------------------------------------------------ population
    _p("initial_count", 2000, 0, 2097152, "", "Population", "cells at start", live=False, kind="int"),
    _p("initial_source", "seed", group="Population", kind="enum", choices=("seed", "random", "mix"),
       live=False, desc="initial genome source"),
    _p("initial_seed_files", "seeds/seed_replicator.asm", group="Population", kind="str", live=False,
       desc="comma-separated seed .asm files"),
    _p("initial_mass", 8.0, 0.1, 500, "m", "Population", "mass of injected cells"),
    _p("initial_energy", 8.0, 0, 1000, "E", "Population", "energy of injected cells"),
    _p("random_genome_len", 32, 4, isa.MAX_GENOME_LEN, "words", "Population", "length of random genomes", kind="int"),
    # ------------------------------------------------------------------ diagnostics
    _p("ledger_interval", 100, 1, 100000, "ticks", "Diagnostics", "ledger is measured every N ticks", kind="int"),
    _p("ledger_check", True, kind="bool", group="Diagnostics", desc="assert ledger balance when measured"),
    _p("ledger_tol", 1e-3, 1e-7, 1, "", "Diagnostics", "relative ledger tolerance", log=True),
    _p("organism_label_interval", 10, 1, 10000, "ticks", "Diagnostics", "connected-component interval", kind="int"),
    _p("stats_interval", 100, 1, 100000, "ticks", "Diagnostics", "stats sampling interval", kind="int"),
    _p("profile_interval", 0, 0, 100000, "ticks", "Diagnostics", "per-phase timing every N ticks (0 = off)", kind="int"),
]

PARAM_SPECS: dict[str, ParamSpec] = {s.name: s for s in _SPECS}
GROUPS = list(dict.fromkeys(s.group for s in _SPECS))


def _validate_specs():
    for s in _SPECS:
        if s.kind == "enum":
            assert s.default in s.choices, s.name


_validate_specs()


class Config:
    """Attribute-style access to parameter values plus point lights."""

    def __init__(self, **overrides):
        object.__setattr__(self, "_v", {k: copy.deepcopy(s.default) for k, s in PARAM_SPECS.items()})
        object.__setattr__(self, "point_lights", [])   # list of dict(x, y, I, vx, vy)
        object.__setattr__(self, "version", 0)
        for k, v in overrides.items():
            self.set(k, v)

    # -- access
    def __getattr__(self, k):
        try:
            return self._v[k]
        except KeyError:
            raise AttributeError(k) from None

    def __setattr__(self, k, v):
        if k in ("point_lights", "version"):
            object.__setattr__(self, k, v)
        else:
            self.set(k, v)

    def set(self, k, v):
        if k not in PARAM_SPECS:
            raise KeyError(f"unknown parameter {k!r}")
        s = PARAM_SPECS[k]
        if s.kind == "int":
            v = int(v)
        elif s.kind == "float":
            v = float(v)
        elif s.kind == "bool":
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                v = bool(v)
            elif not isinstance(v, bool):
                v = str(v).lower() in ("1", "true", "yes", "on", "1.0")
        elif s.kind == "enum":
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                v = s.choices[int(v)]
            if v not in s.choices:
                raise ValueError(f"{k}: {v!r} not in {s.choices}")
        elif s.kind == "str":
            v = str(v)
        self._v[k] = v
        object.__setattr__(self, "version", self.version + 1)

    def get(self, k):
        return self._v[k]

    def enum_index(self, k):
        return PARAM_SPECS[k].choices.index(self._v[k])

    def modified(self):
        return {k: v for k, v in self._v.items() if v != PARAM_SPECS[k].default}

    # -- io
    def to_dict(self):
        return {"params": dict(self._v), "point_lights": copy.deepcopy(self.point_lights)}

    @classmethod
    def from_dict(cls, d):
        c = cls()
        params = d.get("params", d)
        for k, v in params.items():
            if k in PARAM_SPECS:
                c.set(k, v)
        c.point_lights = copy.deepcopy(d.get("point_lights", []))
        return c

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def copy(self):
        return Config.from_dict(self.to_dict())

    def exploit_warnings(self):
        """Static checks for parameter combinations that create free energy (evolution WILL find them)."""
        out = []
        loop = self.detritus_fraction * self.absorb_eff * self.growth_eff
        if loop >= 1.0:
            out.append(f"detritus_fraction*absorb_eff*growth_eff = {loop:.2f} >= 1: dying cells return more energy-mass "
                       "than they cost, which is an exploitable perpetual-motion loop")
        if self.bite_eff * self.bite_mass_gain * self.growth_eff >= 1.0 and self.bite_mass_gain > 0:
            out.append("bite_eff*bite_mass_gain*growth_eff >= 1: predators gain more than they eat")
        if self.share_eff > 1.0:
            out.append("share_eff > 1 creates energy")
        return out

    def diff(self, other: "Config"):
        return {k: (self._v[k], other._v[k]) for k in PARAM_SPECS if self._v[k] != other._v[k]}


# ------------------------------------------------------------------------- GPU parameter struct
def _wp_type(spec: ParamSpec):
    if spec.kind == "float":
        return float
    if spec.kind in ("int", "bool", "enum"):
        return int
    return None


def _make_struct():
    ann = {}
    for s in _SPECS:
        t = _wp_type(s)
        if t is not None:
            ann[s.name] = t
    # derived per-tick values (prefix d_)
    ann.update(
        d_tick=wp.int64,
        d_key=wp.uint32,
        d_world=wp.vec2,
        d_sun_dir=wp.vec2,
        d_sun_I=float,
        d_amb_weight_avg=float,      # mean of max(0,cos)^p over the circle
        d_light_map_offset=float,
        d_nutrient_offset=float,
        d_hash_cell=float,
        d_hash_nx=int,
        d_hash_ny=int,
        d_field_nx=int,
        d_field_ny=int,
        d_light_nx=int,
        d_light_ny=int,
        d_num_sensors=int,
        d_num_point_lights=int,
        d_ledger_on=int,
        d_hostile_mult=float,
        d_hostile_cx=float,
        d_hostile_cy=float,
        d_hostile_r=float,
        d_n_traced=int,
    )
    cls = type("SimParams", (), {"__annotations__": ann, "__module__": __name__})
    return wp.struct(cls)


SimParams = _make_struct()


def avg_cos_pow(p: float, n: int = 4096) -> float:
    import numpy as np
    th = (np.arange(n) + 0.5) / n * 2 * np.pi
    return float(np.mean(np.maximum(0.0, np.cos(th)) ** p))


def fill_struct(cfg: Config, st=None):
    st = st if st is not None else SimParams()
    for s in _SPECS:
        t = _wp_type(s)
        if t is None:
            continue
        v = cfg._v[s.name]
        if s.kind == "enum":
            v = s.choices.index(v)
        elif s.kind == "bool":
            v = int(bool(v))
        setattr(st, s.name, t(v))
    return st
