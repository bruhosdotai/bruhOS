"""The rules planner must act on give/hand instructions.

"Give me the red block." and "Hand me the red block." are pickup requests,
synonymous with "Bring me the red block." -- but the planner only routed
bring/fetch/carry (and take/get/grab/hold) into the pickup branch, so give
and hand returned no plan and the episode ended "no plan for instruction"
even though the skill set can do exactly that: walk to the object and grab
it. With a box destination ("Give the red block to the box.") the same
verbs must take the delivery branch, like bring/fetch/deliver, instead of
planning a pickup that never delivers.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation, SkillCall
from bruhos.sim.kinematic import SimBody, World


def _plan(text, holding=None):
    return RulesPlanner().plan(text, Observation(t=0.0, frame=0, holding=holding), [])


def _run(text, world=None):
    body = SimBody(world or World.tabletop())
    return body, Runtime(body, RulesPlanner()).run(text)


def test_give_hand_plan_a_pickup():
    for text in ("Give me the red block.", "Hand me the red block.", "Give me the blue block."):
        target = "blue_block" if "blue" in text else "red_block"
        assert _plan(text) == [
            SkillCall("walk_toward", {"target": target}),
            SkillCall("grab", {}),
        ], text


def test_give_hand_to_box_plan_a_delivery():
    for text in ("Give the red block to the box.", "Hand the red block to the box.",
                 "Give the blue block into the box."):
        target = "blue_block" if "blue" in text else "red_block"
        assert [c.skill for c in _plan(text)] == [
            "detect", "walk_toward", "grab", "walk_toward", "release",
        ], text
        assert _plan(text)[1].args == {"target": target}, text


def test_give_me_block_end_to_end_holds_it():
    for text in ("Give me the red block.", "Hand me the red block."):
        body, ep = _run(text)
        assert ep.success, text
        assert ep.end_reason == "completed", text
        assert body.world.holding is not None and body.world.holding.name == "red_block", text


def test_give_hand_to_box_end_to_end_delivers():
    for text in ("Give the red block to the box.", "Hand the red block to the box."):
        body, ep = _run(text)
        assert ep.success, text
        assert "grab:red_block" in body.world.events, text
        assert "in_box:red_block" in body.world.events, text
        assert body.world.holding is None, text


def test_pick_up_unaffected():
    # "Bring me ..." is covered by open PR #212, not asserted here.
    body, ep = _run("Pick up the red block.")
    assert ep.success
    assert body.world.holding is not None and body.world.holding.name == "red_block"
