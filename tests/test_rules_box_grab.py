"""The rules planner must not plan to grab the box.

The sim's grab primitive refuses the box by construction (its reachability
predicate excludes the box, like the ball): the hand cannot close on it.
Planning ``walk_toward(box) > grab`` for "Pick up the box." therefore walks
the robot to the box and fails the grab -- then burns the one replan failing
again with the identical plan. The same doomed plan appears for
"Put the box in the box." (detect > walk_toward > grab > ...). A request
bruhOS cannot perform must produce no plan, the same answer the planner
gives for "Do a backflip." (the eval scores that as ``no_plan``) and for
picking up the ball. The box stays a valid target for the skills that can
act on it: walking to it, facing it and detecting it are unaffected.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody, World


def _run(text, world=None):
    body = SimBody(world or World.tabletop())
    return body, Runtime(body, RulesPlanner()).run(text)


def test_pick_up_box_has_no_plan():
    for text in ("Pick up the box.", "Take the box.", "Grab the box.", "Pick up the red box."):
        body, ep = _run(text)
        assert not ep.success, text
        assert ep.end_reason == "no plan for instruction", text
        assert ep.plan == [], text
        assert ep.steps == [], text
        assert (body.world.x, body.world.y) == (0.0, 0.0), text
        assert body.world.events == [], text


def test_put_box_in_box_has_no_plan():
    for text in ("Put the box in the box.", "Move the box into the box."):
        body, ep = _run(text)
        assert not ep.success, text
        assert ep.end_reason == "no plan for instruction", text
        assert ep.plan == [], text
        assert ep.steps == [], text
        assert body.world.events == [], text


def test_box_stays_valid_for_non_grab_skills():
    _, ep = _run("Walk to the box.")
    assert ep.success
    assert [c["skill"] for c in ep.plan] == ["walk_toward"]

    _body, ep = _run("Face the box.")
    assert ep.success
    assert [c["skill"] for c in ep.plan] == ["look_at"]

    _body, ep = _run("Find the box.")
    assert ep.success
    assert [c["skill"] for c in ep.plan] == ["detect"]


def test_grabbable_targets_unaffected():
    body, ep = _run("Pick up the red block.")
    assert ep.success
    assert body.world.holding is not None and body.world.holding.name == "red_block"

    body, ep = _run("Put the red block in the box.")
    assert ep.success
    assert "in_box:red_block" in body.world.events
