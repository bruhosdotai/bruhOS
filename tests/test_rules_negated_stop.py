"""A negated stop must not halt the robot.

Failing-test-first for a reproduced defect: the rules planner's stop branch
claims any instruction that contains a stop word, so "Don't stop." plans
[stop] and the episode reports success with end reason "stopped" -- the
robot does the exact opposite of the instruction. The same happens for
"Do not stop.", "Never stop.", and "Don't stop walking." A negated stop is
a prohibition, not a command: the planner now strips the negated stop
clause and plans the rest of the instruction, and returns no plan when
nothing remains. A plain "Stop." still plans [stop].
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


def test_negated_stop_has_no_plan():
    for text in ("Don't stop.", "Do not stop.", "Never stop.", "Don't ever stop.",
                 "Don't stop walking.", "Do not halt.", "Don't freeze."):
        assert _plan(text) == [], text


def test_negated_stop_is_no_plan_end_to_end():
    body, ep = _run("Don't stop.")
    assert not ep.success
    assert ep.end_reason == "no plan for instruction"
    assert body.world.events == []


def test_negated_stop_keeps_the_rest_of_the_command():
    # The rest is planned exactly as the planner would plan it alone:
    # "walking" alone is not a trigger word, so only the bare verb form
    # carries a command.
    assert [c.skill for c in _plan("Don't stop, walk to the box.")] == ["walk_toward"]
    assert [c.skill for c in _plan("Never stop, follow the line.")] == ["follow_line"]


def test_negated_stop_keeps_the_rest_end_to_end():
    _, ep = _run("Don't stop, walk to the box.")
    assert ep.success, ep.end_reason
    assert [c["skill"] for c in ep.plan] == ["walk_toward"]


def test_plain_stop_still_halts():
    assert [c.skill for c in _plan("Stop.")] == ["stop"]
    _, ep = _run("Stop.")
    assert ep.success and ep.end_reason == "stopped"
