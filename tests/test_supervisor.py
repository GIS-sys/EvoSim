"""M8 acceptance: killing the process mid-run recovers automatically from the last good checkpoint."""
import json
import os
import subprocess
import sys
import time

import pytest

psutil = pytest.importorskip("psutil")
ROOT = os.path.dirname(os.path.dirname(__file__))


def _children_running_sim(run_dir):
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        cl = p.info["cmdline"] or []
        if "evosim" in " ".join(cl) and "run" in cl and "--run-dir" in cl and run_dir in " ".join(cl) \
                and "supervise" not in cl:
            out.append(p)
    return out


def test_supervisor_restarts_killed_child_from_checkpoint(tmp_path):
    root = str(tmp_path / "runs")
    args = [sys.executable, "-m", "evosim", "supervise", "--root", root, "--name", "sv", "--delay", "1", "--",
            "--preset", "fast_debug", "--set", "max_cells=2048", "--set", "initial_count=60",
            "--set", "world_w=128", "--set", "world_h=128", "--ticks", "1500", "--checkpoint-ticks", "100",
            "--stats-format", "csv", "--device", "cpu", "--no-sleep-prevention"]
    sup = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        run_dir = None
        t0 = time.time()
        killed_tick = None
        while time.time() - t0 < 240:
            if run_dir is None and os.path.isdir(root) and os.listdir(root):
                run_dir = os.path.join(root, os.listdir(root)[0])
            if run_dir:
                ck = os.path.join(run_dir, "checkpoints")
                n = len([f for f in os.listdir(ck) if f.endswith(".evck")]) if os.path.isdir(ck) else 0
                kids = _children_running_sim(run_dir)
                if n >= 3 and kids and killed_tick is None:
                    for k in kids:
                        k.kill()                                  # SIGKILL: no handlers run, no final checkpoint
                    killed_tick = n
                    break
            time.sleep(0.3)
        assert killed_tick is not None, "child never produced 3 checkpoints"
        sup.wait(timeout=240)
        assert sup.returncode == 0
        ev = [json.loads(l) for l in open(os.path.join(run_dir, "events.jsonl"))]
        types = [e["type"] for e in ev]
        assert "supervisor_child_died" in types and "supervisor_restart" in types
        assert "run_resume" in types                                # the second launch resumed from a checkpoint
        stops = [e for e in ev if e["type"] == "run_stop"]
        assert stops and stops[-1]["tick"] >= 1500
        assert types.count("supervisor_launch") == 2
    finally:
        if sup.poll() is None:
            sup.kill()
