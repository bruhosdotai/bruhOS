"""The rules planner must not plan to grab the box.

The box is the container every delivery puts objects into: the sim's grab
primitive refuses it by construction (its reachability predicate excludes the
box) and ``release`` treats it as the destination, so no body can grab it.
Planning ``walk_toward(box) > grab`` for "Pick up the box." therefore walks
the robot to the box and fails the grab — then burns the one replan failing
again. A request bruhOS cannot perform must produce no plan, the same answer
the planner gives for "Do a backflip." (the eval scores that as ``no_plan``).
The box stays a valid target for the skills that can act on it: walking to
it, facing it and detecting it are unaffected.
"""

import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation, SkillCall
from bruhos.sim.kinematic import SimBody


def plan(text, holding=None):
    return RulesPlanner().plan(text, Observation(t=0.0, frame=0, holding=holding), [])


@pytest.mark.parametrize("text", [
    "Pick up the box.",
    "Grab the bin.",
    "Take the basket.",
    "Hold the box.",
    "Put the box in the box.",
])
def test_no_plan_for_grabbing_the_box(text):
    assert plan(text) == []


def test_no_plan_for_grabbing_the_box_end_to_end():
    body = SimBody()
    ep = Runtime(body, RulesPlanner()).run("Pick up the box.")
    assert not ep.success
    assert ep.plan == []
    assert ep.end_reason == "no plan for instruction"
    assert ep.steps == []
    assert body.world.events == []


@pytest.mark.parametrize("text,expected", [
    ("Pick up the red block.",
     [SkillCall("walk_toward", {"target": "red_block"}), SkillCall("grab", {})]),
    ("Walk to the box.",
     [SkillCall("walk_toward", {"target": "box"})]),
    ("Face the box.",
     [SkillCall("look_at", {"target": "box"})]),
    ("Find the box.",
     [SkillCall("detect", {"target": "box"})]),
    ("Put the red block in the box.",
     [SkillCall("detect", {"target": "red_block"}), SkillCall("walk_toward", {"target": "red_block"}),
      SkillCall("grab", {}), SkillCall("walk_toward", {"target": "box"}), SkillCall("release", {})]),
    ("Put it in the box.",
     [SkillCall("walk_toward", {"target": "box"}), SkillCall("release", {})]),
])
def test_unaffected_instructions_keep_their_plans(text, expected):
    holding = "red_block" if text == "Put it in the box." else None
    assert plan(text, holding=holding) == expected
