"""Drop/release must not drop an object the robot is not holding.

"Drop the red block." while the robot holds the blue block planned a bare
release, so the episode ended "completed" after dropping the wrong object.
A named drop only releases when the held object matches the named one.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody, World
from bruhos.skills import CATALOG

PLANNER = RulesPlanner()


def _obs(holding):
    return Observation(t=0.0, frame=1, objects=[], holding=holding)


def test_drop_named_object_held_is_release():
    assert [c.skill for c in PLANNER.plan("Drop the blue block.", _obs("blue_block"), CATALOG)] == ["release"]


def test_drop_named_object_not_held_is_no_plan():
    assert PLANNER.plan("Drop the red block.", _obs("blue_block"), CATALOG) == []
    assert PLANNER.plan("Let go of the red block.", _obs("blue_block"), CATALOG) == []
    assert PLANNER.plan("Release the red block.", _obs("blue_block"), CATALOG) == []


def test_drop_generic_target_while_holding_is_release():
    assert [c.skill for c in PLANNER.plan("Drop the block.", _obs("blue_block"), CATALOG)] == ["release"]
    assert [c.skill for c in PLANNER.plan("Drop it.", _obs("blue_block"), CATALOG)] == ["release"]


def test_drop_named_object_with_empty_hand_still_plans_release():
    assert [c.skill for c in PLANNER.plan("Drop the red block.", _obs(None), CATALOG)] == ["release"]


def test_drop_named_object_with_generic_holding_is_release():
    # TonyPi / eliza bridge report holding="object" for any grab; the held
    # identity is unknown, so it cannot be disproved.
    assert [c.skill for c in PLANNER.plan("Drop the red block.", _obs("object"), CATALOG)] == ["release"]


def test_drop_wrong_object_keeps_holding_end_to_end():
    world = World.tabletop()
    body = SimBody(world)
    blue = world.get("blue_block")
    world.holding = blue
    ep = Runtime(body, PLANNER).run("Drop the red block.")
    assert not ep.success
    assert ep.end_reason == "no plan for instruction"
    assert world.holding is blue
    assert "drop:blue_block" not in world.events
