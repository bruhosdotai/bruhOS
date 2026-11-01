"""`parse_plan` must accept a bare single skill call, not just a plan container.

Failing-test-first for a reproduced defect: a brain that returns one skill as
a bare object — `{"skill": "wave", "args": {}}` — passes the safety guard
(`guard_no_joint_actions` falls back to the dict itself and finds no number
chunk), but `parse_plan` only ever read the `"plan"` key, so the call died
with `ValueError: plan must be a list, got NoneType` and the episode ended
"plan rejected" instead of executing one valid skill. This is the same
contradiction open PR #96 fixes for the `{"actions": [...]}` container, but
for a different shape: a single call object, not a container of calls.
"""

import pytest

from bruhos.brain.base import Planner, parse_plan
from bruhos.runtime import Runtime
from bruhos.schemas import SkillCall
from bruhos.sim.kinematic import SimBody


def test_bare_single_call_dict():
    assert parse_plan({"skill": "wave", "args": {}}) == [SkillCall("wave", {})]


def test_bare_single_call_json_string():
    assert parse_plan('{"skill": "walk_toward", "args": {"target": "box"}}') == [
        SkillCall("walk_toward", {"target": "box"})
    ]


def test_bare_single_call_runs_in_the_runtime():
    class SingleCallPlanner(Planner):
        def plan(self, instruction, obs, catalog):
            return parse_plan({"skill": "wave", "args": {}})

    body = SimBody()
    ep = Runtime(body, SingleCallPlanner()).run("Wave hello.")
    assert ep.success
    assert "wave" in body.world.events


def test_plan_container_unchanged():
    assert parse_plan({"plan": [{"skill": "wave", "args": {}}]}) == [SkillCall("wave", {})]
    assert parse_plan([{"skill": "wave", "args": {}}]) == [SkillCall("wave", {})]


def test_dict_without_a_skill_is_still_rejected():
    with pytest.raises(ValueError):
        parse_plan({"foo": "wave"})
    with pytest.raises(ValueError):
        parse_plan({"skill": 42, "args": {}})
