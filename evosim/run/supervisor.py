"""Supervisor: runs the headless sim as a child process and restarts it from the latest good checkpoint
when it exits abnormally (driver reset, native crash, OOM, kill). Covers failures in-process handlers can't.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from .rundir import CheckpointManager, EventLog, RunDir


def supervise(run_dir: str, child_args: list[str], delay=5.0, max_restarts_per_hour=6, python=None,
              poll=0.5, max_total_restarts=None):
    """child_args: arguments after `evosim run`/`resume` that apply to every launch (stop conditions etc.).
    The first launch is `run --run-dir D` (which resumes if D already has a good checkpoint)."""
    rd = RunDir(run_dir)
    elog = EventLog(rd.events_path)
    py = python or sys.executable
    restarts: list[float] = []
    total = 0
    cmd = [py, "-m", "evosim", "run", "--run-dir", run_dir, *child_args]
    while True:
        elog.emit({"type": "supervisor_launch", "severity": "info", "cmd": cmd, "restart_number": total})
        proc = subprocess.Popen(cmd)
        while proc.poll() is None:
            time.sleep(poll)
        code = proc.returncode
        if code == 0:
            elog.emit({"type": "supervisor_done", "severity": "info"})
            return 0
        now = time.time()
        restarts = [t for t in restarts if now - t < 3600]
        elog.emit({"type": "supervisor_child_died", "severity": "error", "exit_code": code})
        if len(restarts) >= max_restarts_per_hour or (max_total_restarts is not None and total >= max_total_restarts):
            elog.emit({"type": "supervisor_giving_up", "severity": "critical", "restarts_last_hour": len(restarts)})
            return 2
        restarts.append(now)
        total += 1
        elog.emit({"type": "supervisor_restart", "severity": "warning", "delay": delay, "restart_number": total,
                   "from_checkpoint": (CheckpointManager(rd).good() or [{}])[-1].get("tick")})
        time.sleep(delay)
