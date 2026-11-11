"""Placing an object "next to the box" must not deliver it into the box.

Failing-test-first for a reproduced defect: the rules planner's into-box
matcher reads the "to the box" inside "next to the box" as a delivery into
the box, so "Put the red block next to the box." plans [detect, walk_toward,
grab, walk_toward(box), release] and the episode reports success with event
``in_box:red_block`` -- the block ends up in the box, the opposite of "next
to" it. The same silent drop happens for bring/drop/move with a "next to
the box" destination. "next to" is a spatial relation, not containment (the
same reasoning open PR #247 applies to non-box object destinations), and the
skill set cannot place an object beside the box: release within reach of the
box is containment in the sim. The honest outcome is no plan. A bare
"beside"/"near the box" already yields no plan, and delivery into the box,
navigation to the box, and picking up a block located next to the box are
unchanged.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody, World


def _run(text, world=None):
    body = SimBody(world or World.tabletop())
    return body, Runtime(body, RulesPlanner()).run(text)


def _plan(text):
    return RulesPlanner().plan(text, Observation(t=0.0, frame=0), [])


def test_put_next_to_box_has_no_plan():
    assert _plan("Put the red block next to the box.") == []


def test_bring_drop_move_next_to_box_have_no_plan():
    for text in ("Bring the red block next to the box.",
                 "Drop the red block next to the box.",
                 "Move the red block next to the box."):
        assert _plan(text) == [], text


def test_next_to_bin_and_basket_have_no_plan():
    assert _plan("Put the red block next to the bin.") == []
    assert _plan("Put the red block next to the basket.") == []


def test_next_to_box_is_no_plan_end_to_end():
    body, ep = _run("Put the red block next to the box.")
    assert not ep.success
    assert ep.end_reason == "no plan for instruction"
    assert body.world.events == []


def test_put_into_box_still_delivers():
    body, ep = _run("Put the red block into the box.")
    assert ep.success, ep.end_reason
    assert "in_box:red_block" in body.world.events


def test_walk_to_box_and_pickup_next_to_box_are_unchanged():
    assert [c.skill for c in _plan("Walk to the box.")] == ["walk_toward"]
    # "next to the box" locates the block here: it is a pickup, not a delivery.
    assert [c.skill for c in _plan("Take the red block next to the box.")] == ["walk_toward", "grab"]
    assert [c.skill for c in _plan("Look at the red block next to the box.")] == ["look_at"]
