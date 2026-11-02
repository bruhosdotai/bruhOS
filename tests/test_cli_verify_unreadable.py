"""`bruhos verify` must report TAMPERED, not crash, on an unreadable trace.

The sibling command `bruhos lerobot` already reports unreadable traces as
TAMPERED with exit 2 instead of a traceback (open PR #111). `bruhos verify`
is the trace's own integrity verdict, yet it calls `load_episode` unguarded:
a corrupt `episode.json`, a truncated or malformed `steps.jsonl` line
(`json.loads` raises `JSONDecodeError`, a `ValueError`), or a nonexistent
path (`FileNotFoundError`, an `OSError`) raise out of the CLI with a
traceback. A user auditing a suspicious trace directory is exactly the case
where the command must return a verdict, not a stack.
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


def test_verify_valid_trace_still_ok(tmp_path):
    assert main(["verify", str(_trace_dir(tmp_path))]) == 0


def test_verify_corrupt_step_line_reports_tampered(tmp_path, capsys):
    d = _trace_dir(tmp_path)
    p = d / "steps.jsonl"
    lines = p.read_text().splitlines()
    lines[0] = "{ not valid json"
    p.write_text("\n".join(lines) + "\n")
    assert main(["verify", str(d)]) == 2
    out = capsys.readouterr().out
    assert "TAMPERED" in out


def test_verify_truncated_step_line_reports_tampered(tmp_path, capsys):
    d = _trace_dir(tmp_path)
    p = d / "steps.jsonl"
    p.write_text(p.read_text()[: p.stat().st_size - 3])
    assert main(["verify", str(d)]) == 2
    assert "TAMPERED" in capsys.readouterr().out


def test_verify_corrupt_header_reports_tampered(tmp_path, capsys):
    d = _trace_dir(tmp_path)
    (d / "episode.json").write_text("{oops")
    assert main(["verify", str(d)]) == 2
    assert "TAMPERED" in capsys.readouterr().out


def test_verify_missing_field_reports_tampered(tmp_path, capsys):
    d = _trace_dir(tmp_path)
    lines = [json.loads(l) for l in (d / "steps.jsonl").read_text().splitlines()]
    del lines[0]["hash"]
    (d / "steps.jsonl").write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    assert main(["verify", str(d)]) == 2
    assert "TAMPERED" in capsys.readouterr().out


def test_verify_missing_directory_reports_tampered(tmp_path, capsys):
    assert main(["verify", str(tmp_path / "no-such-episode")]) == 2
    assert "TAMPERED" in capsys.readouterr().out


def test_verify_loadable_but_tampered_still_exit_2(tmp_path):
    d = _trace_dir(tmp_path)
    p = d / "steps.jsonl"
    lines = [json.loads(l) for l in p.read_text().splitlines()]
    lines[0]["outcome"]["status"] = "failed"
    p.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    assert main(["verify", str(d)]) == 2
