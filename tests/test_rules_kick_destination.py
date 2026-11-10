"""Kicking a ball toward (or at, onto, on) the box must not certify a blind kick as success.

Failing-test-first for a reproduced defect: the rules planner's kick branch
claimed any kick instruction, even when the instruction also named a box
destination for the ball. Open #79 covers the ``to`` / ``into`` / ``in`` /
``inside`` phrasings through the ``into_box`` matcher, but that matcher never
sees the other destination prepositions, so ``Kick the ball toward the box.``
planned only ``[walk_toward(ball, 0.12), kick]`` and the episode reported
``success=True`` while the ball was kicked away from the box (it ended
~1.32 m from the box, which never moved). The same silent drop happens for
``towards``, ``at``, ``onto`` and ``on`` with any box noun (box / bin /
basket). Kicking cannot aim at a destination, so a kick instruction that
names a box destination is unsatisfiable and must yield no plan, the same
honest outcome as the ``to``-destination kick in open #79. Destination-less
kicks, and ``to the left`` (a foot direction, not a destination), are
unchanged.
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


_DESTINATION_KICKS = [
    "Kick the ball toward the box.",
    "Kick the ball towards the box.",
    "Kick the ball toward the big box.",
    "Kick the ball towards the bin.",
    "Kick the ball toward the basket.",
    "Kick the ball at the box.",
    "Kick the ball onto the box.",
    "Kick the ball on the box.",
    "Kick the ball to the box.",
    "Kick the ball into the box.",
    "Kick the yellow ball toward the box.",
    "Kick the ball toward the box with your left foot.",
]


def test_kick_ball_to_a_box_destination_has_no_plan():
    for text in _DESTINATION_KICKS:
        assert _plan_text(RulesPlanner().plan(text, _obs(), [])) == [], text


def test_kick_ball_toward_box_runtime_kicks_nothing():
    body, ep = _run("Kick the ball toward the box.")
    assert not ep.success
    assert ep.end_reason == "no plan for instruction"
    assert ep.plan == []
    assert ep.steps == []
    ball = body.world.get("yellow_ball")
    assert (ball.x, ball.y) == (0.4, -0.6)
    assert body.world.events == []


def test_destinationless_kick_unchanged():
    body, ep = _run("Kick the ball.")
    assert ep.success, ep.end_reason
    assert "kicked:yellow_ball" in body.world.events

    body, ep = _run("Kick the ball with your left foot.")
    assert ep.success, ep.end_reason
    assert "kick:left" in body.world.events
    assert "kicked:yellow_ball" in body.world.events

    body, ep = _run("Kick.")
    assert ep.success, ep.end_reason
    assert [c["skill"] for c in ep.plan] == ["kick"]


def test_kick_to_the_left_is_a_foot_direction_not_a_destination():
    # "to the left" names which foot kicks, not where the ball should go.
    body, ep = _run("Kick the ball to the left.")
    assert ep.success, ep.end_reason
    assert "kick:left" in body.world.events
    assert "kicked:yellow_ball" in body.world.events
