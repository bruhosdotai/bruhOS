"""`bruhos lerobot` must not turn a tampered trace into training data.

Reproduced on main: after editing one step of a sealed episode, `bruhos verify`
exits 2 with ``TAMPERED``, while ``bruhos lerobot`` on the same directory still
exits 0 and prints frames. The README calls the trace the record of execution
("execution is data"), and these frames are what a learner consumes, so an
unverified export silently launders a forged episode into a dataset.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody
from bruhos.trace import TraceWriter


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "bruhos.cli", *args],
                          capture_output=True, text=True)


@pytest.fixture()
def sealed_trace(tmp_path: Path) -> Path:
    ep = Runtime(SimBody(), RulesPlanner()).run("Pick up the cube.")
    d = TraceWriter(tmp_path).write(ep)
    assert _cli("verify", str(d)).returncode == 0
    return d


def _tamper(d: Path) -> None:
    f = d / "steps.jsonl"
    lines = f.read_text().splitlines()
    step = json.loads(lines[0])
    step["call"]["args"] = {"tampered": True}
    lines[0] = json.dumps(step, sort_keys=True)
    f.write_text("\n".join(lines) + "\n")


def test_lerobot_refuses_a_tampered_trace(sealed_trace: Path):
    assert _cli("lerobot", str(sealed_trace)).returncode == 0
    _tamper(sealed_trace)
    assert _cli("verify", str(sealed_trace)).returncode == 2
    out = _cli("lerobot", str(sealed_trace))
    assert out.returncode == 2, "tampered trace was exported"
    assert "TAMPERED" in (out.stdout + out.stderr)
    assert '"frames"' not in out.stdout


def test_lerobot_force_exports_with_a_warning(sealed_trace: Path):
    _tamper(sealed_trace)
    out = _cli("lerobot", str(sealed_trace), "--force")
    assert out.returncode == 0
    assert '"frames"' in out.stdout
    assert "TAMPERED" in out.stderr
