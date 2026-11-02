"""A shorthand call that gives one argument twice must be a parse error, not last-value-wins.

Before this fix ``parse_call`` wrote every segment into a dict, so a repeated argument silently
replaced the earlier value: ``walk_toward(red_block, box)`` planned a walk to the box and
``kick(foot=left, foot=right)`` kicked right. The brain's plan was ambiguous, but the runtime
executed one reading of it on the body instead of rejecting it.
"""

import pytest

from bruhos.brain.base import Planner, parse_plan
from bruhos.runtime import Runtime
from bruhos.schemas import SkillCall, parse_call
from bruhos.sim.kinematic import SimBody
from bruhos.skills.registry import positional_args

POS = positional_args()


@pytest.mark.parametrize("text", [
    "walk_toward(red_block, box)",            # two positional values
    "walk_toward(red_block, target=box)",     # positional, then the same arg by name
    "walk_toward(target=red_block, box)",     # by name, then positionally
    "kick(foot=left, foot=right)",            # the same keyword twice
    "kick(left, right)",
    "follow_line(5, steps=50)",
])
def test_repeated_argument_is_rejected(text):
    with pytest.raises(ValueError, match="more than once"):
        parse_call(text, POS)


def test_repeated_argument_never_reaches_the_body():
    class TextPlanner(Planner):
        def plan(self, instruction, obs, catalog):
            return parse_plan([instruction])

    body = SimBody()
    ep = Runtime(body, TextPlanner()).run("kick(foot=left, foot=right)")
    assert not ep.success
    assert [s.kind for s in ep.steps] == ["rejected"]
    assert body.world.events == []


def test_distinct_arguments_still_parse():
    assert parse_call("walk_toward(red_block, stop_distance_m=0.2)", POS) == \
        SkillCall("walk_toward", {"target": "red_block", "stop_distance_m": 0.2})
    assert parse_call("walk_toward(target=box, stop_distance_m=0.12)", POS) == \
        SkillCall("walk_toward", {"target": "box", "stop_distance_m": 0.12})
    assert parse_call("kick(foot=left)", POS) == SkillCall("kick", {"foot": "left"})
