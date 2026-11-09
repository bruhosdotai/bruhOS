"""follow_line is for the floor line, not for a named object.

"Follow the red block." planned follow_line, so with a line on the floor the
episode ended "completed" after walking the line while the block was never
approached. A named non-line object has no tracking skill, so it is no plan.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody, World
from bruhos.skills import CATALOG

PLANNER = RulesPlanner()


def _obs():
    return Observation(t=0.0, frame=1, objects=[])


def test_follow_named_object_is_no_plan():
    assert PLANNER.plan("Follow the red block.", _obs(), CATALOG) == []
    assert PLANNER.plan("Follow the yellow ball.", _obs(), CATALOG) == []


def test_follow_the_line_still_plans_follow_line():
    assert [c.skill for c in PLANNER.plan("Follow the line.", _obs(), CATALOG)] == ["follow_line"]
    assert [c.skill for c in PLANNER.plan("Follow the red line.", _obs(), CATALOG)] == ["follow_line"]


def test_follow_without_object_still_plans_follow_line():
    assert [c.skill for c in PLANNER.plan("Follow me.", _obs(), CATALOG)] == ["follow_line"]


def test_follow_object_leaves_line_untouched_end_to_end():
    world = World.tabletop(line_m=0.6)
    body = SimBody(world)
    ep = Runtime(body, PLANNER).run("Follow the red block.")
    assert not ep.success
    assert ep.end_reason == "no plan for instruction"
    assert world.line_m == 0.6
