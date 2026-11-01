"""`bruhos lerobot` must not export a trace that fails verification.

A trace whose step hashes no longer match its trace_root is exactly what
`bruhos verify` reports as TAMPERED — but `bruhos lerobot` used to export
it anyway, turning the tampered outcome into training frames with a clean
exit code. Export must refuse such a trace instead.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.cli import main
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody
from bruhos.trace import LocalSink, SinkDispatcher

import json


def _saved_wave_episode(tmp_path):
    d = SinkDispatcher([LocalSink(str(tmp_path))])
    ep = Runtime(SimBody(), RulesPlanner(), d).run("Wave hello.")
    assert d.flush(5)
    return tmp_path / ep.episode_id


def test_lerobot_refuses_a_step_tampered_trace(tmp_path, capsys):
    """On base, this exported the tampered outcome with exit 0."""
    directory = _saved_wave_episode(tmp_path)
    steps_path = directory / "steps.jsonl"
    lines = steps_path.read_text().strip().split("\n")
    step = json.loads(lines[0])
    step["outcome"]["status"] = "failed"
    step["outcome"]["detail"] = "tampered by test"
    lines[0] = json.dumps(step)
    steps_path.write_text("\n".join(lines) + "\n")

    assert main(["lerobot", str(directory)]) == 2
    out = capsys.readouterr().out
    assert "TAMPERED" in out
    assert '"frames"' not in out


def test_lerobot_still_exports_a_verified_trace(tmp_path, capsys):
    directory = _saved_wave_episode(tmp_path)
    assert main(["lerobot", str(directory)]) == 0
    out = capsys.readouterr().out
    assert '"frames"' in out
    assert "TAMPERED" not in out
