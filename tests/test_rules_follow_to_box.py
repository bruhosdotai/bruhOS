"""Following a line to the box must not certify following the line as success.

Failing-test-first for a reproduced defect: the rules planner's follow
branch claimed any follow instruction, even when the instruction also
named a box destination. So ``Follow the line to the box.`` planned only
``[follow_line]`` and, with a line in view, reported ``success=True``
even when the robot faced away from the box and walked farther from it
— the destination was silently dropped and success certified without
ever checking the box. Following a line cannot verify arrival at a
destination, so a follow instruction that names a box destination must
yield no plan, the same honest outcome as a walk or kick to the box.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody, World


def _obs(holding=None):
    return Observation(t=0.0, frame=0, objects=[], holding=holding)


def _plan_text(plan):
    return [(c.skill, dict(c.args)) for c in plan]


def test_follow_line_to_box_has_no_plan():
    plan = RulesPlanner().plan("Follow the line to the box.", _obs(), [])
    assert _plan_text(plan) == []


def test_follow_tape_to_box_has_no_plan():
    plan = RulesPlanner().plan("Follow the tape to the box.", _obs(), [])
    assert _plan_text(plan) == []


def test_follow_line_into_box_has_no_plan():
    plan = RulesPlanner().plan("Follow the line into the box.", _obs(), [])
    assert _plan_text(plan) == []


def test_follow_line_next_to_box_has_no_plan():
    # Consistent with #71: "next to the box" still matches the into_box
    # destination, so it must not be claimed as a follow either.
    plan = RulesPlanner().plan("Follow the line next to the box.", _obs(), [])
    assert _plan_text(plan) == []


def test_follow_line_to_box_runtime_does_not_succeed():
    # Line in view, robot facing AWAY from the box: on base this walked
    # away from the box and still reported success.
    w = World.tabletop(line_m=2.0)
    w.heading_deg = 180.0
    body = SimBody(w)
    ep = Runtime(body, RulesPlanner()).run("Follow the line to the box.")
    assert not ep.success
    assert ep.end_reason == "no plan for instruction"
    assert body.world.events == []
    assert (body.world.x, body.world.y) == (0.0, 0.0)


def test_follow_line_unchanged():
    # Guard: a destination-less follow still follows the line.
    plan = RulesPlanner().plan("Follow the line.", _obs(), [])
    assert _plan_text(plan) == [("follow_line", {})]


def test_follow_line_episode_unchanged():
    # Guard: destination-less following still succeeds with a line in view.
    body = SimBody(World.tabletop(line_m=2.0))
    ep = Runtime(body, RulesPlanner()).run("Follow the line.")
    assert ep.success, ep.end_reason


def test_follow_line_to_block_unchanged():
    # Documented limit: a non-box destination still drops, because only
    # box/bin/basket destinations produce an into_box signal.
    plan = RulesPlanner().plan("Follow the line to the red block.", _obs(), [])
    assert _plan_text(plan) == [("follow_line", {})]


def test_follow_line_to_box_and_stop_unchanged():
    # Guard: stop-first preempt is untouched by this fix.
    plan = RulesPlanner().plan("Follow the line to the box and stop.", _obs(), [])
    assert _plan_text(plan) == [("stop", {})]
