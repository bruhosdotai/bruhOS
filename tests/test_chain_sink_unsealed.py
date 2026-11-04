"""ChainSink must not anchor an episode that was never sealed.

An unsealed episode has ``trace_root == ""``, so the anchor payload used to
carry ``"trace_root": "0x"`` — an on-chain attestation to nothing, identical
for every unsealed episode. ``Sink.emit`` is documented as persisting or
publishing a *sealed* episode, so anchoring must refuse instead of publishing
the empty root (the runtime seals before submitting, so normal flow is
unaffected; the dispatcher already turns a failing sink into an error receipt
rather than taking the runtime down).
"""

import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Episode
from bruhos.sim.kinematic import SimBody
from bruhos.trace import ChainSink


def test_emit_refuses_unsealed_episode_and_submits_nothing():
    ep = Episode(episode_id="e1", instruction="Wave hello.", brain="rules", body="sim", started_at=0.0)
    assert ep.trace_root == ""
    calls = []
    sink = ChainSink(submit=calls.append)
    with pytest.raises(ValueError, match="unsealed"):
        sink.emit(ep)
    assert calls == []


def test_anchor_payload_refuses_unsealed_episode():
    ep = Episode(episode_id="e1", instruction="Wave hello.", brain="rules", body="sim", started_at=0.0)
    with pytest.raises(ValueError, match="unsealed"):
        ChainSink.anchor_payload(ep)


def test_emit_still_anchors_sealed_episode():
    ep = Runtime(SimBody(), RulesPlanner()).run("Wave hello.")
    assert len(ep.trace_root) == 64
    calls = []
    sink = ChainSink(submit=lambda p: (calls.append(p), "0xtx")[1])
    receipt = sink.emit(ep)
    assert receipt["tx"] == "0xtx"
    assert receipt["payload"]["trace_root"] == "0x" + ep.trace_root
    assert len(calls) == 1
