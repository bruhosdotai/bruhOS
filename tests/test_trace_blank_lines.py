"""Tests for loading traces whose steps.jsonl contains blank or whitespace-only lines.

Per the JSON Lines specification (jsonlines.org), empty lines may be ignored by parsers.
Standard tools and editors often introduce trailing newlines or blank lines, which must
not crash load_episode with a JSONDecodeError.
"""

from __future__ import annotations

from pathlib import Path

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody
from bruhos.trace import LocalSink, load_episode, to_lerobot_frames, verify_episode


def _saved_episode(tmp_path: Path):
    ep = Runtime(SimBody(), RulesPlanner()).run("Put the red one in the box.")
    d = Path(LocalSink(tmp_path).emit(ep)["path"])
    return ep, d


def test_load_episode_tolerates_trailing_blank_lines(tmp_path):
    ep, d = _saved_episode(tmp_path)
    steps_file = d / "steps.jsonl"

    # Append trailing newlines to steps.jsonl
    with steps_file.open("a", encoding="utf-8") as f:
        f.write("\n\n")

    loaded = load_episode(d)
    assert len(loaded.steps) == len(ep.steps)
    assert verify_episode(loaded)
    assert loaded.trace_root == ep.trace_root


def test_load_episode_tolerates_interleaved_blank_and_whitespace_lines(tmp_path):
    ep, d = _saved_episode(tmp_path)
    steps_file = d / "steps.jsonl"

    # Read original lines and interleave blank and whitespace lines
    lines = steps_file.read_text(encoding="utf-8").splitlines()
    corrupted_content = ""
    for line in lines:
        corrupted_content += line + "\n\n   \n\t\n"

    steps_file.write_text(corrupted_content, encoding="utf-8")

    loaded = load_episode(d)
    assert len(loaded.steps) == len(ep.steps)
    assert verify_episode(loaded)
    assert loaded.trace_root == ep.trace_root


def test_lerobot_frames_from_trace_with_blank_lines(tmp_path):
    ep, d = _saved_episode(tmp_path)
    steps_file = d / "steps.jsonl"
    with steps_file.open("a", encoding="utf-8") as f:
        f.write("\n")

    loaded = load_episode(d)
    frames = to_lerobot_frames(loaded)
    assert len(frames) == len(ep.steps)
