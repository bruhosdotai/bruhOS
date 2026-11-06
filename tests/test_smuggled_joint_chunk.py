"""A joint-action vector is rejected even when a skill plan is in the same payload.

``guard_no_joint_actions`` used to inspect only one container (``plan``, else
``actions``, else the raw value). Two shapes therefore executed as ordinary
skill calls:

- ``{"plan": [wave], "actions": <16-wide chunk>}`` — the chunk is never read
- ``{"skill": "wave", "args": {}, "action": <chunk>}`` — ``SkillCall.from_dict``
  keeps only ``skill`` and ``args``, so the vector is dropped before the trace

The body then waves, and the sealed episode shows a normal ``wave`` with no
record of the chunk. The contract is that this output is an
``IntegrationError`` and never reaches the body.
"""

from __future__ import annotations

import pytest

from bruhos.brain.adapters.groot_s2 import GR00TS2Planner
from bruhos.brain.base import IntegrationError, parse_plan
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody

CHUNK = [[0.01 * j for j in range(16)] for _ in range(4)]


def test_chunk_beside_a_skill_plan_is_rejected():
    raw = {"plan": [{"skill": "wave", "args": {}}], "actions": CHUNK}
    with pytest.raises(IntegrationError):
        parse_plan(raw)


def test_chunk_on_the_skill_object_is_rejected():
    raw = {"plan": [{"skill": "wave", "args": {}, "action": [0.1] * 16}]}
    with pytest.raises(IntegrationError):
        parse_plan(raw)


def test_runtime_does_not_run_a_plan_that_carries_a_chunk():
    body = SimBody()
    brain = GR00TS2Planner(reasoner=lambda *a: {
        "plan": [{"skill": "wave", "args": {}}],
        "actions": CHUNK,
    })
    ep = Runtime(body, brain).run("Wave.")
    assert not ep.success
    assert ep.steps and ep.steps[0].kind == "rejected"
    assert body.world.events == []


def test_scalar_skill_args_are_not_action_chunks():
    calls = parse_plan({"plan": [{
        "skill": "walk_toward",
        "args": {"target": "box", "stop_distance_m": 0.12},
    }]})
    assert [(c.skill, c.args["stop_distance_m"]) for c in calls] == [("walk_toward", 0.12)]


def test_empty_actions_list_beside_a_plan_is_not_a_chunk():
    calls = parse_plan({"plan": [{"skill": "wave", "args": {}}], "actions": []})
    assert [c.skill for c in calls] == ["wave"]
