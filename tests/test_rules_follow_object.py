"""Following a non-line object must not plan line-following.

Failing-test-first for a reproduced defect: the rules planner mapped ANY
"follow ..." instruction to ``follow_line``, regardless of what was to be
followed. So ``Follow the red block.`` planned ``follow_line`` and, with a
tape present, reported ``success=True`` ("end of line") while the block was
never followed — a wrong action certified as success. The skill set has no
follow-object or follow-person primitive; the only followable thing in this
world is the floor line (the world's tape). A follow instruction that does
not name the line therefore has no honest plan, exactly as a next-to
instruction has no place-near plan.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody, World


def _obs(holding=None):
    return Observation(t=0.0, frame=0, objects=[], holding=holding)


def _plan_text(plan):
    return [(c.skill, dict(c.args)) for c in plan]


def test_follow_block_has_no_plan():
    plan = RulesPlanner().plan("Follow the red block.", _obs(), [])
    assert _plan_text(plan) == []


def test_follow_person_has_no_plan():
    plan = RulesPlanner().plan("Follow me.", _obs(), [])
    assert _plan_text(plan) == []


def test_follow_green_block_has_no_plan():
    plan = RulesPlanner().plan("Follow the green block.", _obs(), [])
    assert _plan_text(plan) == []


def test_follow_block_to_box_has_no_plan():
    # No wrong delivery either: following a block to the box is not a
    # delivery instruction the planner may claim.
    plan = RulesPlanner().plan("Follow the red block to the box.", _obs(), [])
    assert _plan_text(plan) == []


def test_follow_line_unchanged():
    plan = RulesPlanner().plan("Follow the line.", _obs(), [])
    assert _plan_text(plan) == [("follow_line", {})]


def test_follow_tape_unchanged():
    # The world's line is tape (see line_m in kinematic.py and the
    # perception docstring), so "tape" still means the line.
    plan = RulesPlanner().plan("Follow the tape.", _obs(), [])
    assert _plan_text(plan) == [("follow_line", {})]


def test_follow_line_for_steps_unchanged():
    # Step-count parsing is out of scope: the plan shape is unchanged.
    plan = RulesPlanner().plan("Follow the line for 10 steps.", _obs(), [])
    assert _plan_text(plan) == [("follow_line", {})]


def test_follow_block_runtime_does_not_succeed():
    body = SimBody(World.tabletop(line_m=1.0))
    ep = Runtime(body, RulesPlanner()).run("Follow the red block.")
    assert not ep.success
    assert ep.end_reason == "no plan for instruction"


def test_follow_line_runtime_still_succeeds():
    body = SimBody(World.tabletop(line_m=1.0))
    ep = Runtime(body, RulesPlanner()).run("Follow the line.")
    assert ep.success, ep.end_reason
