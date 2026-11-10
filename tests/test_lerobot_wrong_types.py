"""Tests ensuring to_lerobot_frames tolerates wrong-typed step fields.

A loadable trace whose step fields hold the wrong JSON type (a string where
a number, dict, or list is expected) must export with safe default values
rather than crashing the `bruhos lerobot` command with an uncaught
traceback. This is the wrong-type sibling of the null/missing tolerance
in open #88, which guards None and missing keys only: every shape below
still raises under #88's patched code (a truthy string passes its
`or {}` guards, and `float("abc")` still raises).
"""

import json

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.cli import main
from bruhos.runtime import Runtime
from bruhos.schemas import Episode, Step
from bruhos.sim.kinematic import SimBody
from bruhos.trace import LocalSink, load_episode, to_lerobot_frames, verify_episode
from bruhos.trace.hashing import seal as _seal
from bruhos.trace import TraceWriter


def _step(**overrides):
    fields = {
        "index": 0,
        "t": 10.0,
        "call": {"skill": "wave", "args": {}},
        "outcome": {"status": "ok"},
        "obs_before": {
            "imu": {"roll_deg": 1.5, "pitch_deg": 2.5, "fallen": False},
            "line": {"visible": True, "offset": 3.5},
            "objects": [{"id": "red_block"}],
            "holding": "red_block",
        },
        "kind": "skill",
        "hash": "abc1",
    }
    fields.update(overrides)
    return Step(**fields)


def _episode_with(step):
    return Episode(
        episode_id="ep-wrong-types",
        instruction="Wave hello.",
        brain="test",
        body="test",
        started_at=10.0,
        steps=[step],
        success=True,
    )


def test_lerobot_frames_tolerates_wrong_typed_numeric_fields():
    step = _step(obs_before={
        "imu": {"roll_deg": "abc", "pitch_deg": "x", "fallen": False},
        "line": {"visible": True, "offset": "far"},
        "objects": [],
        "holding": None,
    })
    frames = to_lerobot_frames(_episode_with(step))
    assert len(frames) == 1
    # line.visible is a genuine bool, so it still exports 1.0; every
    # wrong-typed numeric field defaults to 0.0.
    assert frames[0]["observation.state"] == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert frames[0]["timestamp"] == 0.0


def test_lerobot_frames_tolerates_wrong_typed_call_outcome_and_obs():
    step = _step(call="wave", outcome="ok", obs_before="not-a-dict")
    frames = to_lerobot_frames(_episode_with(step))
    assert len(frames) == 1
    assert frames[0]["observation.state"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert frames[0]["action"] == [-1.0, 0.0]


def test_lerobot_frames_tolerates_wrong_typed_imu_and_objects():
    step = _step(obs_before={"imu": "xx", "line": {}, "objects": "xx", "holding": None})
    frames = to_lerobot_frames(_episode_with(step))
    assert len(frames) == 1
    # "xx" must not be counted as two objects; a non-list objects is empty.
    assert frames[0]["observation.state"][4] == 0.0


def test_lerobot_frames_tolerates_wrong_typed_timestamp():
    step = _step(t="x")
    frames = to_lerobot_frames(_episode_with(step))
    assert len(frames) == 1
    assert frames[0]["timestamp"] == 0.0


def _saved_wave_episode(tmp_path):
    ep = Runtime(SimBody(), RulesPlanner()).run("Wave hello.")
    return tmp_path / LocalSink(tmp_path).emit(ep)["path"]


def _edit_first_step(trace_dir, mutate):
    p = trace_dir / "steps.jsonl"
    lines = p.read_text().splitlines()
    step = json.loads(lines[0])
    mutate(step)
    lines[0] = json.dumps(step)
    p.write_text("\n".join(lines) + "\n")


def test_cli_lerobot_reports_frames_not_traceback_on_wrong_typed_trace(tmp_path, capsys):
    d = _saved_wave_episode(tmp_path)
    _edit_first_step(d, lambda s: s["obs_before"]["imu"].update(roll_deg="abc"))
    rc = main(["lerobot", str(d)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["frames"]) == len(load_episode(d).steps)
    assert out["frames"][0]["observation.state"][0] == 0.0


def test_cli_lerobot_exports_re_sealed_wrong_typed_trace(tmp_path, capsys):
    d = _saved_wave_episode(tmp_path)
    _edit_first_step(d, lambda s: s["obs_before"]["imu"].update(roll_deg="abc"))
    ep = load_episode(d)
    _seal(ep)
    TraceWriter(d.parent).write(ep)
    assert verify_episode(load_episode(d))
    rc = main(["lerobot", str(d)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["frames"][0]["observation.state"][0] == 0.0
