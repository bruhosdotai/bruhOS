"""The rules planner must not plan to grab the ball.

The sim's grab primitive refuses the ball by construction (its reachability
predicate excludes the ball, like the box): the hand cannot close on it. Planning
``walk_toward(ball) > grab`` for "Pick up the yellow ball." therefore walks the
robot to the ball and fails the grab -- then burns the one replan failing again
with the identical plan. The same doomed plan appears for "Put the yellow ball
in the box." (detect > walk_toward > grab > ...). A request bruhOS cannot perform
must produce no plan, the same answer the planner gives for "Do a backflip."
(the eval scores that as ``no_plan``) and for picking up the box itself.
The ball stays a valid target for the skills that can act on it: walking to it,
facing it, detecting it and kicking it are unaffected.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody, World


def _run(text, world=None):
    body = SimBody(world or World.tabletop())
    return body, Runtime(body, RulesPlanner()).run(text)


def test_pick_up_ball_has_no_plan():
    body, ep = _run("Pick up the yellow ball.")
    assert not ep.success
    assert ep.end_reason == "no plan for instruction"
    assert ep.plan == []
    assert ep.steps == []
    assert (body.world.x, body.world.y) == (0.0, 0.0)
    assert body.world.events == []


def test_take_ball_and_put_ball_in_box_have_no_plan():
    for text in ("Take the ball.", "Put the yellow ball in the box."):
        body, ep = _run(text)
        assert not ep.success, text
        assert ep.end_reason == "no plan for instruction", text
        assert ep.plan == [], text
        assert ep.steps == [], text
        assert body.world.events == [], text


def test_ball_stays_valid_for_non_grab_skills():
    body, ep = _run("Walk to the yellow ball.")
    assert ep.success
    assert [c["skill"] for c in ep.plan] == ["walk_toward"]

    body, ep = _run("Kick the ball.")
    assert ep.success
    assert "kicked:yellow_ball" in body.world.events

    body, ep = _run("Look at the yellow ball.")
    assert ep.success
    assert [c["skill"] for c in ep.plan] == ["look_at"]


def test_grabbable_targets_unaffected():
    body, ep = _run("Pick up the red block.")
    assert ep.success
    assert body.world.holding is not None and body.world.holding.name == "red_block"

    body, ep = _run("Grab the blue one.")
    assert ep.success
    assert body.world.holding is not None and body.world.holding.name == "blue_block"

    body, ep = _run("Put the red block in the box.")
    assert ep.success
    assert "in_box:red_block" in body.world.events
