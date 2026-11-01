"""Rules planner: an explicit step count in a follow-line instruction must reach
the follow_line skill instead of silently falling back to the 20-step default."""

import pytest

from bruhos.brain import make_planner
from bruhos.runtime import Runtime
from bruhos.schemas import SkillCall
from bruhos.sim.kinematic import SimBody, World


def _plan(text):
    return make_planner("rules").plan(text, SimBody().observe(), [])


@pytest.mark.parametrize("text,steps", [
    ("Follow the line for 10 steps.", 10),
    ("Follow the line for 5 steps.", 5),
    ("Follow the line 3 steps.", 3),
    ("Follow the line for 1 step.", 1),
])
def test_follow_line_count_reaches_skill(text, steps):
    assert _plan(text) == [SkillCall("follow_line", {"steps": steps})]


def test_follow_line_without_count_keeps_default():
    # No explicit count: the plan stays bare so the skill default (20) applies.
    assert _plan("Follow the line.") == [SkillCall("follow_line", {})]
    assert _plan("Follow the line for twenty steps.") == [SkillCall("follow_line", {})]


def test_follow_line_negative_count_passes_through_unsilenced():
    # A negative count is passed through, not silently flipped to positive;
    # what the skill/validation layer does with it is that layer's contract.
    assert _plan("Follow the line for -5 steps.") == [SkillCall("follow_line", {"steps": -5})]


def _run_follow(text, line_m):
    body = SimBody(World.tabletop(line_m=line_m))
    ep = Runtime(body, make_planner("rules")).run(text)
    assert ep.success, ep.end_reason
    step = next(s for s in ep.steps if s.call["skill"] == "follow_line")
    return body, step.outcome["data"]["steps"]


def test_follow_line_five_steps_on_long_line():
    body, done = _run_follow("Follow the line for 5 steps.", line_m=2.0)
    assert done == 5
    assert body.world.x == pytest.approx(0.25)
    assert body.world.line_m == pytest.approx(1.75)


def test_follow_line_default_unchanged_on_long_line():
    body, done = _run_follow("Follow the line.", line_m=2.0)
    assert done == 20
    assert body.world.x == pytest.approx(1.0)
