"""Kicking a ball to the box must not certify a blind kick as success.

Failing-test-first for a reproduced defect: the rules planner's kick branch
claimed any kick instruction, even when the instruction also named a box
destination for the ball. So ``Kick the ball to the box.`` planned only
``[walk_toward(ball, 0.12), kick]`` and the episode reported
``success=True`` while the ball was kicked away from the box (it ended
~1.32 m from the box, which never moved). Kicking cannot aim at a
destination, so a kick instruction that names a box destination is
unsatisfiable and must yield no plan, the same honest outcome as a walk
to the box or a next-to delivery. Aiming before kicking does not rescue
it: after ``look_at(box)`` the ball falls outside the kick skill's 30
degree cone, so the kick whiffs while the skill still reports OK — a
second false success. Destination-less kicks are unchanged.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody, World


def _obs(holding=None):
    return Observation(t=0.0, frame=0, objects=[], holding=holding)


def _plan_text(plan):
    return [(c.skill, dict(c.args)) for c in plan]


def test_kick_ball_to_box_has_no_plan():
    plan = RulesPlanner().plan("Kick the ball to the box.", _obs(), [])
    assert _plan_text(plan) == []


def test_kick_ball_into_box_has_no_plan():
    plan = RulesPlanner().plan("Kick the ball into the box.", _obs(), [])
    assert _plan_text(plan) == []


def test_kick_ball_to_bin_has_no_plan():
    # "bin" maps to the same box destination, so it must not be dropped either.
    plan = RulesPlanner().plan("Kick the ball to the bin.", _obs(), [])
    assert _plan_text(plan) == []


def test_kick_ball_to_box_runtime_does_not_succeed():
    body = SimBody(World.tabletop())
    ep = Runtime(body, RulesPlanner()).run("Kick the ball to the box.")
    assert not ep.success
    assert ep.end_reason == "no plan for instruction"
    assert body.world.events == []


def test_kick_ball_unchanged():
    # Guard: a destination-less kick still walks to the ball and kicks.
    plan = RulesPlanner().plan("Kick the ball.", _obs(), [])
    assert _plan_text(plan) == [
        ("walk_toward", {"target": "ball", "stop_distance_m": 0.12}),
        ("kick", {"foot": "right"}),
    ]


def test_kick_ball_left_unchanged():
    plan = RulesPlanner().plan("Kick the ball left.", _obs(), [])
    assert _plan_text(plan) == [
        ("walk_toward", {"target": "ball", "stop_distance_m": 0.12}),
        ("kick", {"foot": "left"}),
    ]


def test_kick_red_ball_unchanged():
    plan = RulesPlanner().plan("Kick the red ball.", _obs(), [])
    assert _plan_text(plan) == [
        ("walk_toward", {"target": "red_ball", "stop_distance_m": 0.12}),
        ("kick", {"foot": "right"}),
    ]


def test_kick_block_unchanged():
    # Guard: bare kick behavior (no ball involved) is untouched by this fix.
    plan = RulesPlanner().plan("Kick the block.", _obs(), [])
    assert _plan_text(plan) == [("kick", {"foot": "right"})]
