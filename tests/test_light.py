"""M3 acceptance: a hand-built cup of pigmented cells around a photoreceptor makes LIGHT depend on
the direction of the source, with numeric thresholds."""
import math

import numpy as np
import pytest

from helpers import make_sim, put


def _scene(device, sun_angle, cup, rw):
    sim = make_sim(device, light_cell=1.0, sun_I=1.0, ambient_I=0.0, shadow_ray_len=24.0, ray_steps=32,
                   pigment_opacity=3.0, base_opacity=0.0, sun_angle=sun_angle, receptor_p=4.0, max_mass=2.0)
    c = np.array([128.0, 128.0])
    rec = put(sim, [c], traits={"PHOTORECEPTOR": 1.0, "RECEPTOR_WIDTH": rw}, mass=1.0, energy=1.0, angle=[0.0])
    if cup:   # arc of pigmented cells behind and beside the receptor; the opening faces +x
        arc = [c + 4.0 * np.array([math.cos(a), math.sin(a)]) for a in np.deg2rad(np.arange(100, 261, 20))]
        put(sim, arc, traits={"PIGMENT": 1.0}, mass=1.5, energy=1.0)
    sim.step(1)
    s = sim.state.scratch
    return float(s.light_recv.numpy()[rec[0]]), float(s.light_total.numpy()[rec[0]])


DIRS = {"front": 0.0, "side_l": math.pi / 2, "back": math.pi, "side_r": -math.pi / 2}


def test_cup_makes_light_directional(device):
    cup = {k: _scene(device, a, True, rw=1.0)[0] for k, a in DIRS.items()}     # omnidirectional receptor
    nocup = {k: _scene(device, a, False, rw=1.0)[0] for k, a in DIRS.items()}
    for k in DIRS:                                   # control: without the cup every direction is equal
        assert nocup[k] == pytest.approx(1.0, abs=1e-4)
    assert cup["front"] > 0.95                      # light through the cup opening arrives
    assert cup["side_l"] < 0.30 and cup["side_r"] < 0.30
    assert cup["back"] < 0.15
    assert cup["front"] > 3 * cup["side_l"] and cup["front"] > 6 * cup["back"]
    assert cup["side_l"] == pytest.approx(cup["side_r"], rel=0.05)    # mirror-symmetric cup


def test_receptor_axis_alone_gives_directionality(device):
    narrow = {k: _scene(device, a, False, rw=0.0)[0] for k, a in DIRS.items()}   # Euglena-like eyespot
    assert narrow["front"] == pytest.approx(1.0, abs=1e-4)
    assert narrow["back"] == 0.0 and narrow["side_l"] < 1e-4
    wide = _scene(device, math.pi, False, rw=1.0)[0]
    assert wide == pytest.approx(1.0, abs=1e-4)


def test_own_footprint_is_excluded(device):
    sim = make_sim(device, light_cell=1.0, sun_I=1.0, ambient_I=0.0, shadow_ray_len=24.0, ray_steps=32,
                   base_opacity=2.0, sun_angle=0.0)
    s = put(sim, [[128.0, 128.0]], mass=2.0, energy=1.0)
    sim.step(1)
    assert float(sim.state.scratch.light_total.numpy()[s[0]]) == pytest.approx(1.0, abs=1e-3)


def test_point_light_falloff_and_water_absorption(device):
    sim = make_sim(device, sun_I=0.0, ambient_I=0.0, k_water=0.01, point_light_r0=50.0, max_point_lights=2)
    sim.cfg.point_lights = [dict(x=128.0, y=128.0, I=2.0)]
    near, far = put(sim, [[128.0, 148.0], [128.0, 228.0]], mass=1.0, energy=1.0)
    sim.step(1)
    lt = sim.state.scratch.light_total.numpy()
    exp = lambda r: 2.0 / (1 + r * r / 2500.0) * math.exp(-0.01 * r)
    assert lt[near] == pytest.approx(exp(20.0), rel=2e-2)
    assert lt[far] == pytest.approx(exp(100.0), rel=2e-2)


def test_day_night_cycle(device):
    sim = make_sim(device, sun_I=1.0, ambient_I=0.0, day_period=100, day_curve="sin")
    s = put(sim, [[100.0, 100.0]], mass=1.0, energy=1.0)
    vals = {}
    for t in (0, 25, 75):
        sim.tick = t
        sim.step(1)
        vals[t] = float(sim.state.scratch.light_total.numpy()[s[0]])
    assert vals[25] == pytest.approx(1.0, abs=1e-3) and vals[75] == 0.0 and vals[0] == pytest.approx(0.0, abs=1e-3)


def test_photosynthesis_scales_with_light_and_trait(device):
    sim = make_sim(device, sun_I=1.0, ambient_I=0.0, photo_eff=0.05, growth_rate=0.0)
    a, b = put(sim, [[50.0, 50.0], [150.0, 150.0]], traits=np.array([[1] + [0] * 8, [0.5] + [0] * 8], np.float32),
               mass=4.0, energy=1.0)
    sim.step(1)
    e = sim.state.cells.energy.numpy()
    r = math.sqrt(4.0 / math.pi)
    assert e[a] - 1.0 == pytest.approx(0.05 * 1.0 * 1.0 * math.pi * r * r, rel=2e-2)
    assert (e[b] - 1.0) == pytest.approx(0.5 * (e[a] - 1.0), rel=1e-3)
