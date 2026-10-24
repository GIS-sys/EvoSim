"""Runs the whole UI (every panel open, run mode, overlays, rewind, screenshot) under a virtual display with
software OpenGL. Skipped when xvfb-run / the UI dependencies are not available."""
import os
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("glfw")
pytest.importorskip("moderngl")
pytest.importorskip("imgui_bundle")
pytest.importorskip("OpenGL")
ROOT = os.path.dirname(os.path.dirname(__file__))


@pytest.mark.skipif(sys.platform == "win32" or not shutil.which("xvfb-run"), reason="needs xvfb-run (Linux CI)")
def test_ui_smoke_all_panels(tmp_path):
    env = dict(os.environ, LIBGL_ALWAYS_SOFTWARE="1", HOME=str(tmp_path))
    r = subprocess.run(["xvfb-run", "-a", "-s", "-screen 0 1600x950x24", sys.executable, "-m", "evosim", "ui",
                        "--preset", "fast_debug", "--set", "max_cells=4096", "--set", "initial_count=300",
                        "--set", "world_w=192", "--set", "world_h=192", "--device", "cpu", "--smoke", "40"],
                       cwd=str(tmp_path), env={**env, "PYTHONPATH": ROOT}, capture_output=True, text=True, timeout=280)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    # the UI prints "[ui] panel 'x' failed" / "[ui] tab ... failed" / "[ui] GL error ..." for its own problems
    # (the X11 "failed to convert selection to string" GLFW warning on an empty clipboard is harmless)
    assert "[ui] " not in r.stdout + r.stderr, (r.stdout + r.stderr)[-3000:]
    assert os.path.exists(tmp_path / "screenshots" / "ui_smoke.png")
