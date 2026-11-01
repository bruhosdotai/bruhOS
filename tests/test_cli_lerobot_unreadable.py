"""`bruhos lerobot` must report TAMPERED, not crash, on an unreadable trace.

Failing-test-first for a reproduced defect: `cmd_lerobot` called `load_episode`
unguarded, so a nonexistent path, a missing `steps.jsonl`, or a corrupt step
(a missing required field) raised out of the CLI with a traceback. The sibling
command `bruhos verify` already reports exactly these files as TAMPERED with
exit 2 instead of crashing, and the export gate for loadable-but-tampered
traces lives in open PR #108 — but an unreadable trace never reaches that
gate, because the load itself crashes first. Export must give the same
verdict as verify for the same input.
"""

import json
from pathlib import Path

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.cli import main
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody
from bruhos.trace import LocalSink


def _trace_dir(tmp_path):
    ep = Runtime(SimBody(), RulesPlanner()).run("Wave hello.")
    return Path(LocalSink(tmp_path).emit(ep)["path"])


def test_lerobot_valid_trace_still_exports(tmp_path, capsys):
    assert main(["lerobot", str(_trace_dir(tmp_path))]) == 0
    assert '"frames"' in capsys.readouterr().out


def test_lerobot_missing_path_is_tampered_not_a_crash(tmp_path, capsys):
    assert main(["lerobot", str(tmp_path / "no-such-episode")]) == 2
    assert "TAMPERED" in capsys.readouterr().out


def test_lerobot_missing_steps_file_is_tampered_not_a_crash(tmp_path, capsys):
    d = _trace_dir(tmp_path)
    (d / "steps.jsonl").unlink()
    assert main(["lerobot", str(d)]) == 2
    assert "TAMPERED" in capsys.readouterr().out


def test_lerobot_step_missing_field_is_tampered_not_a_crash(tmp_path, capsys):
    d = _trace_dir(tmp_path)
    lines = (d / "steps.jsonl").read_text().splitlines()
    step = json.loads(lines[0])
    del step["outcome"]
    lines[0] = json.dumps(step)
    (d / "steps.jsonl").write_text("\n".join(lines) + "\n")
    assert main(["lerobot", str(d)]) == 2
    assert "TAMPERED" in capsys.readouterr().out
