"""TraceWriter must never replace a sealed episode on disk with a different one.

On ``main`` ``write()`` did ``mkdir(exist_ok=True)`` and rewrote both files, so a second episode
with the same ``episode_id`` (a UUID-prefix collision, a clock reset, a copied traces directory)
silently destroyed the first record and its hash chain.
"""

from __future__ import annotations

import time

import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody
from bruhos.trace import LocalSink, SinkDispatcher, TraceWriter, load_episode, verify_episode
from bruhos.trace.hashing import seal


def _two_episodes_same_id():
    first = Runtime(SimBody(), RulesPlanner()).run("Wave hello.")
    second = Runtime(SimBody(), RulesPlanner()).run("Stop.")
    second.episode_id = first.episode_id
    seal(second)                                   # the id is part of the chain seed
    assert first.trace_root != second.trace_root
    return first, second


def test_different_episode_with_same_id_is_refused_and_first_trace_survives(tmp_path):
    first, second = _two_episodes_same_id()
    writer = TraceWriter(tmp_path)
    path = writer.write(first)
    with pytest.raises(FileExistsError, match=first.episode_id):
        writer.write(second)
    saved = load_episode(path)
    assert saved.instruction == first.instruction
    assert saved.trace_root == first.trace_root
    assert verify_episode(saved)


def test_zero_step_episodes_share_a_root_but_are_still_refused(tmp_path):
    """Raised in review: every zero-step episode seals to the empty Merkle root, so root equality
    must not be taken as identity. Two different 'no plan' episodes with one id -> refuse."""
    first = Runtime(SimBody(), RulesPlanner()).run("Do a backflip.")
    second = Runtime(SimBody(), RulesPlanner()).run("Fly to the moon.")
    assert first.end_reason == second.end_reason == "no plan for instruction"
    second.episode_id = first.episode_id
    seal(second)
    assert first.trace_root == second.trace_root and first.header() != second.header()
    writer = TraceWriter(tmp_path)
    path = writer.write(first)
    original = (path / "episode.json").read_bytes()
    with pytest.raises(FileExistsError, match=first.episode_id):
        writer.write(second)
    assert (path / "episode.json").read_bytes() == original
    assert load_episode(path).instruction == "Do a backflip."


def test_rewriting_the_same_sealed_episode_is_idempotent(tmp_path):
    ep = Runtime(SimBody(), RulesPlanner()).run("Wave hello.")
    writer = TraceWriter(tmp_path)
    assert writer.write(ep) == writer.write(ep)
    assert verify_episode(load_episode(tmp_path / ep.episode_id))
    empty = Runtime(SimBody(), RulesPlanner()).run("Do a backflip.")       # zero steps, empty root
    assert writer.write(empty) == writer.write(empty)


def test_local_sink_reports_the_refusal_as_an_error_receipt(tmp_path):
    first, second = _two_episodes_same_id()
    sinks = SinkDispatcher([LocalSink(tmp_path)])
    sinks.submit(first)
    sinks.submit(second)
    assert sinks.flush(5)
    receipts = [r for _, _, r in sinks.receipts]
    assert receipts[0] == {"path": str(tmp_path / first.episode_id)}
    assert isinstance(receipts[1], str) and receipts[1].startswith("error:") and first.episode_id in receipts[1]
    assert verify_episode(load_episode(tmp_path / first.episode_id))
    assert load_episode(tmp_path / first.episode_id).instruction == first.instruction
