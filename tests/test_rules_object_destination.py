"""A non-box object destination must not be silently dropped from a plan.

Failing-test-first for a reproduced defect: the rules planner only
recognises the box as a destination (the ``into_box`` branch). Every other
branch claimed its verb and ignored a "to <object>" destination, then
certified success for work that never happened:

- ``Get the red block to the blue block.`` planned walk + grab and reported
  ``success=True`` with the block merely held, never delivered.
- ``Drop the ball to the red block.`` planned a bare release and reported
  success with the ball dropped at the robot's feet, not at the red block.
- ``Kick the yellow ball to the red block.`` planned walk + kick and
  reported success with the ball 1.64 m from the red block.
- ``Walk the red block to the blue block.`` planned a lone walk and
  reported success with the block never moved.

The skill set has no place-at-object primitive — the box is special
because release into it is a modeled containment — so the honest outcome
for a delivery to a non-box object is no plan, the same answer the
planner gives for a bare "move <object>" (open PR #224) and for walking
an object to the box (open PR #78). Pure navigation ("Walk to the box.",
"Go to the green block.") has no object before the "to", so it still
walks; "next to" is a spatial relation, not a destination, so
"Walk the red block next to the blue block." still walks to the block.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody, World


def _obs(holding=None):
    return Observation(t=0.0, frame=0, objects=[], holding=holding)


def _plan_text(plan):
    return [(c.skill, dict(c.args)) for c in plan]


def _run(text, world=None):
    body = SimBody(world or World.tabletop())
    return body, Runtime(body, RulesPlanner()).run(text)


def test_no_plan_for_take_get_to_an_object():
    for text in ("Take the red block to the blue block.",
                 "Get the red block to the blue block.",
                 "Take the ball to the red block.",
                 "Take the ball to the green block.",
                 "Take the ball to the line.",
                 "Take the ball over to the red block."):
        assert _plan_text(RulesPlanner().plan(text, _obs(), [])) == [], text


def test_no_plan_for_drop_to_an_object():
    for text in ("Drop the ball to the red block.",
                 "Drop the red block to the blue block."):
        assert _plan_text(RulesPlanner().plan(text, _obs(holding="yellow_ball"), [])) == [], text


def test_no_plan_for_kick_to_an_object():
    assert _plan_text(RulesPlanner().plan("Kick the yellow ball to the red block.", _obs(), [])) == []


def test_no_plan_for_walk_object_to_an_object():
    assert _plan_text(RulesPlanner().plan("Walk the red block to the blue block.", _obs(), [])) == []


def test_take_to_object_does_not_deliver_end_to_end():
    body, ep = _run("Get the red block to the blue block.")
    assert not ep.success
    assert ep.end_reason == "no plan for instruction"
    assert ep.plan == []
    assert ep.steps == []
    assert body.world.holding is None
    red = body.world.get("red_block")
    assert (red.x, red.y) == (0.6, 0.25)


def test_kick_to_object_does_not_kick_end_to_end():
    body, ep = _run("Kick the yellow ball to the red block.")
    assert not ep.success
    assert ep.plan == []
    assert body.world.events == []
    ball = body.world.get("yellow_ball")
    assert (ball.x, ball.y) == (0.4, -0.6)


def test_walk_object_to_object_does_not_move_it_end_to_end():
    body, ep = _run("Walk the red block to the blue block.")
    assert not ep.success
    assert ep.plan == []
    red = body.world.get("red_block")
    assert (red.x, red.y) == (0.6, 0.25)


def test_drop_to_object_keeps_holding_end_to_end():
    world = World.tabletop()
    world.holding = world.get("yellow_ball")
    body, ep = _run("Drop the ball to the red block.", world)
    assert not ep.success
    assert ep.plan == []
    assert body.world.holding is not None and body.world.holding.name == "yellow_ball"
    assert "drop:yellow_ball" not in body.world.events


def test_unaffected_instructions_keep_their_plans():
    assert _plan_text(RulesPlanner().plan("Take the red block.", _obs(), [])) == [
        ("walk_toward", {"target": "red_block"}),
        ("grab", {}),
    ]
    # On base "take" is not an into_box verb yet (open PRs #63/#201 add it),
    # so take-to-box still plans pickup only; the fix must not change that.
    assert _plan_text(RulesPlanner().plan("Take the red block to the box.", _obs(), [])) == [
        ("walk_toward", {"target": "red_block"}),
        ("grab", {}),
    ]
    assert _plan_text(RulesPlanner().plan("Drop the ball.", _obs(holding="yellow_ball"), [])) == [
        ("release", {}),
    ]
    assert _plan_text(RulesPlanner().plan("Kick the ball.", _obs(), [])) == [
        ("walk_toward", {"target": "ball", "stop_distance_m": 0.12}),
        ("kick", {"foot": "right"}),
    ]
    assert _plan_text(RulesPlanner().plan("Walk to the box.", _obs(), [])) == [
        ("walk_toward", {"target": "box"}),
    ]
    assert _plan_text(RulesPlanner().plan("Go to the green block.", _obs(), [])) == [
        ("walk_toward", {"target": "green_block"}),
    ]
    assert _plan_text(RulesPlanner().plan("Walk the red block next to the blue block.", _obs(), [])) == [
        ("walk_toward", {"target": "red_block"}),
    ]
    assert _plan_text(RulesPlanner().plan("Take the ball to me.", _obs(), [])) == [
        ("walk_toward", {"target": "ball"}),
        ("grab", {}),
    ]
    assert _plan_text(RulesPlanner().plan("Take the red block to the blue block and stop.", _obs(), [])) == [
        ("stop", {}),
    ]


def test_bring_to_box_still_delivers_end_to_end():
    body, ep = _run("Bring the red block to the box.")
    assert ep.success, ep.end_reason
    assert "in_box:red_block" in body.world.events
    assert body.world.holding is None
