import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody


def plan(text, holding=None):
    return RulesPlanner().plan(text, Observation(t=0.0, frame=0, holding=holding), [])


def plan_skills(calls):
    return [(c.skill, dict(c.args)) for c in calls]


@pytest.mark.parametrize("text", [
    "Put the red block in the box.",
    "Put the red one in the box.",
    "Bring the red block to the box.",
    "Place the red cube into the box.",
    "Drop the red block in the bin.",
    "Carry the red block to the basket.",
])
def test_put_held_object_in_box_plans_walk_and_release(text):
    calls = plan(text, holding="red_block")
    assert plan_skills(calls) == [("walk_toward", {"target": "box"}), ("release", {})]


def test_put_held_object_generic_tonypi_holding():
    calls = plan("Put the red block in the box.", holding="object")
    assert plan_skills(calls) == [("walk_toward", {"target": "box"}), ("release", {})]


def test_end_to_end_held_block_delivery():
    body = SimBody()
    body.world.holding = body.world.get("red_block")
    ep = Runtime(body, RulesPlanner()).run("Put the red block in the box.")
    assert ep.success, ep.end_reason
    assert "in_box:red_block" in body.world.events
    assert [c["skill"] for c in ep.plan] == ["walk_toward", "release"]


def test_holding_different_object_does_not_falsely_deliver():
    calls = plan("Put the red block in the box.", holding="blue_block")
    assert ("walk_toward", {"target": "box"}) not in plan_skills(calls)[:1]
