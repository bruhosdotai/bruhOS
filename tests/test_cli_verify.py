"""``bruhos verify`` must report TAMPERED, not crash, on an unreadable trace.

Failing-test-first for a reproduced defect: ``cmd_verify`` called ``load_episode``
unguarded, so a trace directory with a corrupt ``steps.jsonl`` (a step missing a
required field), a missing ``steps.jsonl``, or a nonexistent path raised out of
the CLI with a traceback instead of the command's own verdict. The dashboard
already treats exactly these files as corrupt traces it skips; the verify
command — the tool whose whole job is to say OK or TAMPERED — must do the
same and exit 2.
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


def test_verify_valid_trace_still_passes(tmp_path):
    assert main(["verify", str(_trace_dir(tmp_path))]) == 0


def test_verify_step_missing_field_is_tampered_not_a_crash(tmp_path, capsys):
    d = _trace_dir(tmp_path)
    lines = (d / "steps.jsonl").read_text().splitlines()
    step = json.loads(lines[0])
    del step["outcome"]
    (d / "steps.jsonl").write_text(json.dumps(step) + "\n")
    assert main(["verify", str(d)]) == 2
    assert "TAMPERED" in capsys.readouterr().out


def test_verify_truncated_step_line_is_tampered_not_a_crash(tmp_path, capsys):
    d = _trace_dir(tmp_path)
    (d / "steps.jsonl").write_text('{"index": 0, "t": 0.0, "call": ')
    assert main(["verify", str(d)]) == 2
    assert "TAMPERED" in capsys.readouterr().out


def test_verify_missing_steps_file_is_tampered_not_a_crash(tmp_path, capsys):
    d = _trace_dir(tmp_path)
    (d / "steps.jsonl").unlink()
    assert main(["verify", str(d)]) == 2
    assert "TAMPERED" in capsys.readouterr().out


def test_verify_nonexistent_path_is_tampered_not_a_crash(tmp_path, capsys):
    assert main(["verify", str(tmp_path / "no-such-episode")]) == 2
    assert "TAMPERED" in capsys.readouterr().out
