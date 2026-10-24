import numpy as np
import warp as wp

from evosim import dims


@wp.kernel
def _k(p: wp.array(dtype=wp.vec2), q: wp.array(dtype=wp.vec2), world: wp.vec2, out_d: wp.array(dtype=wp.vec2),
       out_w: wp.array(dtype=wp.vec2), out_n: wp.array(dtype=wp.vec2)):
    i = wp.tid()
    out_d[i] = dims.min_image(q[i] - p[i], world)
    out_w[i] = dims.wrap_pos(p[i], world)
    out_n[i] = dims.min_image(p[i] - q[i], world)


def test_torus_min_image_and_wrap(device):
    rs = np.random.default_rng(0)
    W = np.array([200.0, 120.0], np.float32)
    P = rs.uniform(-500, 500, (4000, 2)).astype(np.float32)
    Q = rs.uniform(-500, 500, (4000, 2)).astype(np.float32)
    d, w, n = (wp.zeros(len(P), dtype=wp.vec2, device=device) for _ in range(3))
    wp.launch(_k, len(P), [wp.array(P, dtype=wp.vec2, device=device), wp.array(Q, dtype=wp.vec2, device=device),
                           wp.vec2(*W), d, w, n], device=device)
    d, w, n = d.numpy(), w.numpy(), n.numpy()
    assert np.all(w >= 0) and np.all(w < W)
    assert np.all(np.abs(d) <= W / 2 + 1e-3)
    assert np.array_equal(d, -n)                             # exact antisymmetry (Newton's 3rd law relies on it)
    ref = dims.np_min_image(Q.astype(np.float64) - P, W)
    assert np.allclose(d, ref, atol=2e-3)
    # wrapping across the edge keeps distances: a cell just left of 0 and one just right are close
    far = dims.np_min_image(np.array([199.0, 0.0]) - np.array([1.0, 0.0]), W)
    assert abs(far[0]) == 2.0


def test_numpy_helpers():
    W = np.array([100.0, 100.0])
    assert np.allclose(dims.np_min_image([60.0, -60.0], W), [-40.0, 40.0])
    assert np.allclose(dims.np_wrap_pos([-1.0, 101.0], W), [99.0, 1.0])
