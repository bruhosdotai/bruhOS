"""Existence questions must be answered by a detect.

Failing-test-first for a reproduced defect: the rules planner answers
"Can you see the red block?" and "What do you see?" with a detect, but
"Is there a red block?" -- the same question in existence form -- plans
nothing, so the robot reports "no plan for instruction" instead of
looking. A detect is the honest answer either way: it returns OK with
"found <target>" when the object is in view and FAILED with "<target>
not found after full scan" when a full revolution finds nothing.

One guard: a question about containment ("Is there a red block in the
box?") gets no plan. Objects in the box are hidden from observe(), so a
scan would report "not found" for a block that is in fact in the box --
a false negative, not an answer.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation, SkillCall
from bruhos.sim.kinematic import SimBody, World, WorldObject


def _plan(text):
    return RulesPlanner().plan(text, Observation(t=0.0, frame=0), [])


def _run(text, world=None):
    body = SimBody(world or World.tabletop())
    return body, Runtime(body, RulesPlanner()).run(text)


def test_is_there_plans_detect():
    assert _plan("Is there a red block?") == [SkillCall("detect", {"target": "red_block"})]
    assert _plan("Is there a box?") == [SkillCall("detect", {"target": "box"})]
    assert _plan("Is there a yellow ball?") == [SkillCall("detect", {"target": "yellow_ball"})]
    assert _plan("Is the red block here?") == [SkillCall("detect", {"target": "red_block"})]


def test_are_there_plans_detect():
    # The plural noun has no singular mapping in the planner, so a plural
    # question resolves to its color, which Detection.matches still
    # resolves against the same object.
    assert _plan("Are there any red blocks?") == [SkillCall("detect", {"target": "red"})]
    assert _plan("Are there any blocks?") == []


def test_is_there_in_the_box_has_no_plan():
    assert _plan("Is there a red block in the box?") == []
    assert _plan("Is there a red block in the bin?") == []


def test_is_there_end_to_end_found():
    _, ep = _run("Is there a red block?")
    assert ep.success, ep.end_reason
    assert [c["skill"] for c in ep.plan] == ["detect"]


def test_is_there_end_to_end_absent():
    world = World(objects=[WorldObject("block", "blue", 0.7, -0.3),
                           WorldObject("box", None, 0.9, 0.0),
                           WorldObject("ball", "yellow", 0.4, -0.6)])
    _, ep = _run("Is there a red block?", world)
    assert not ep.success
    assert "not found" in ep.end_reason


def test_unaffected_instructions_keep_their_plans():
    assert _plan("Can you see the red block?") == [SkillCall("detect", {"target": "red_block"})]
    assert _plan("Find the green block.") == [SkillCall("detect", {"target": "green_block"})]
    assert _plan("What do you see?") == [SkillCall("detect", {})]
    assert _plan("Do a backflip.") == []
    assert _plan("Pick up the red block.") == [
        SkillCall("walk_toward", {"target": "red_block"}), SkillCall("grab", {})]
