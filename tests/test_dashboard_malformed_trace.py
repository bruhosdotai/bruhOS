"""One malformed trace must not take down the whole read-only console.

`status_data()` documents the opposite behaviour in its own comment —

    # Concurrent/incomplete records do not hide other readable episodes.

— but the guard is `except TRACE_ERRORS` with `TRACE_ERRORS = (OSError, ValueError,
KeyError, TypeError, AttributeError, OverflowError)`. A trace directory is written
by whatever produced it (another agent, a crashed run, a third-party exporter), so
a deeply nested JSON header is a realistic input, and `json.loads` answers it with
`RecursionError` — not a subclass of any of those. `status_data()` then raises, and
`GET /status` returns 500 for every trace in the directory, not just the broken one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.dashboard import status_data
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody
from bruhos.trace import TraceWriter


@pytest.fixture()
def traces(tmp_path: Path) -> Path:
    TraceWriter(tmp_path).write(Runtime(SimBody(), RulesPlanner()).run("Pick up the cube."))
    return tmp_path


def _broken(root: Path, name: str, episode_json: str, steps_jsonl: str = "") -> Path:
    d = root / name
    d.mkdir()
    (d / "episode.json").write_text(episode_json)
    (d / "steps.jsonl").write_text(steps_jsonl)
    return d


def test_deeply_nested_header_does_not_hide_readable_episodes(traces: Path):
    _broken(traces, "broken-recursion", "[" * 200_000)
    data = status_data(traces)
    assert len(data["episodes"]) == 1, "one unreadable header hid the good episode"
    assert data["skipped"] >= 1


def test_json_that_is_not_an_object_is_counted_not_raised(traces: Path):
    _broken(traces, "broken-shape", json.dumps([1, 2, 3]))
    data = status_data(traces)
    assert len(data["episodes"]) == 1
    assert data["skipped"] >= 1


def test_unreadable_steps_file_is_counted_not_raised(traces: Path):
    d = _broken(traces, "broken-steps", "{}")
    (d / "episode.json").write_text(json.dumps({
        "episode_id": "x", "instruction": "y", "brain": "rules", "body": "sim",
        "started_at": 1.0, "plan": [], "success": True, "recoveries": 0, "falls": 0,
        "ended_at": 2.0, "end_reason": "completed", "skillpack_hash": "sha256:x",
        "brain_version": "t", "trace_root": "0" * 64,
    }))
    (d / "steps.jsonl").write_text("{not json\n")
    data = status_data(traces)
    assert len(data["episodes"]) == 1
    assert data["skipped"] >= 1
