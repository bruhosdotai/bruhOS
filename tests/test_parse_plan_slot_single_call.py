"""`parse_plan` must accept one call in the ``plan`` slot, not just a list of calls.

Failing-test-first for a reproduced defect: a brain that returns a single skill
as the ``plan`` value — ``{"plan": {"skill": "wave", "args": {}}}`` or
``{"plan": "wave()"}`` — passes the safety guard (a lone call object is not a
joint chunk and a shorthand string is not one either), but ``parse_plan`` only
ever read a list from the ``"plan"`` key, so the call died with
``ValueError: plan must be a list, got dict`` (or ``got str``) and the episode
ended "plan rejected" instead of executing one valid skill. This is the same
contradiction open PR #112 fixes for a bare single call object and open PR #96
fixes for the ``{"actions": [...]}`` container, but for a different shape: a
single call sitting in the ``plan`` slot itself.
"""

import pytest

from bruhos.brain.base import IntegrationError, Planner, parse_plan
from bruhos.runtime import Runtime
from bruhos.schemas import SkillCall
from bruhos.sim.kinematic import SimBody


def test_plan_slot_holding_a_single_call_dict():
    assert parse_plan({"plan": {"skill": "wave", "args": {}}}) == [SkillCall("wave", {})]


def test_plan_slot_holding_a_single_call_dict_without_args():
    assert parse_plan({"plan": {"skill": "wave"}}) == [SkillCall("wave", {})]


def test_plan_slot_holding_a_shorthand_string():
    assert parse_plan({"plan": "wave()"}) == [SkillCall("wave", {})]
    assert parse_plan({"plan": "walk_toward(red_block)"}) == [
        SkillCall("walk_toward", {"target": "red_block"})
    ]


def test_plan_slot_holding_a_single_call_json_string():
    assert parse_plan('{"plan": {"skill": "wave", "args": {}}}') == [SkillCall("wave", {})]


def test_plan_slot_single_call_runs_in_the_runtime():
    class SlotCallPlanner(Planner):
        def plan(self, instruction, obs, catalog):
            return parse_plan({"plan": {"skill": "wave", "args": {}}})

    body = SimBody()
    ep = Runtime(body, SlotCallPlanner()).run("Wave hello.")
    assert ep.success, ep.end_reason
    assert [c["skill"] for c in ep.plan] == ["wave"]
    assert "wave" in body.world.events


def test_plan_slot_garbage_still_rejected():
    with pytest.raises(ValueError, match="plan must be a list"):
        parse_plan({"plan": {"note": "not a call"}})
    with pytest.raises(ValueError, match="plan must be a list"):
        parse_plan({"plan": {"skill": 42}})
    # A bare number in the plan slot is still a joint chunk, caught by the guard.
    with pytest.raises(IntegrationError, match="continuous action chunk"):
        parse_plan({"plan": 42})
