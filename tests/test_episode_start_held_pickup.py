"""A pickup is already done when the episode starts with that object in the hand.

The rules planner still emits walk_toward + grab. The held object is omitted
from observations, so the walk scans a full revolution and the episode fails
with "not found" even though the hand already holds the named object. The
same fetch is planned after an opening fall recovery: stand-up leaves the
grip intact, then the scan fails. A different object in the hand is not this
case, and a fetch that continues on to a release is not treated as done.
"""

import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.brain.base import Planner
from bruhos.runtime import Runtime
from bruhos.schemas import SkillCall
from bruhos.sim.kinematic import SimBody, World
from bruhos.trace import verify_episode


def _held(name: str, *, fallen: bool = False) -> World:
    world = World.tabletop()
    world.holding = world.get(name)
    if fallen:
        world.fallen = True
        world.fell_forward = True
    return world


@pytest.mark.parametrize("text,name", [
    ("Pick up the red block.", "red_block"),
    ("Grab the blue one.", "blue_block"),
    ("Hold the green block.", "green_block"),
])
def test_episode_starting_with_the_named_object_held_completes_the_pickup(text, name):
    body = SimBody(_held(name))
    ep = Runtime(body, RulesPlanner()).run(text)
    assert ep.success, ep.end_reason
    assert ep.end_reason == f"already holding {name}"
    assert body.world.holding is not None and body.world.holding.name == name
    assert body.n_locomotion == 0
    assert not any(event.startswith("grab:") or event.startswith("drop:") for event in body.world.events)
    assert [step.kind for step in ep.steps] == []
    assert [call["skill"] for call in ep.plan] == ["walk_toward", "grab"]
    assert verify_episode(ep)


def test_episode_starting_fallen_while_holding_stands_up_and_keeps_the_pickup():
    body = SimBody(_held("red_block", fallen=True))
    ep = Runtime(body, RulesPlanner()).run("Pick up the red block.")
    assert ep.success, ep.end_reason
    assert ep.end_reason == "already holding red_block"
    assert ep.recoveries == 1
    assert not body.world.fallen
    assert "stood_up" in body.world.events
    assert body.world.holding is not None and body.world.holding.name == "red_block"
    assert body.n_locomotion == 0
    assert [step.kind for step in ep.steps] == ["recovery"]
    assert [call["skill"] for call in ep.plan] == ["walk_toward", "grab"]
    assert verify_episode(ep)


def test_holding_a_different_object_is_not_a_finished_pickup():
    body = SimBody(_held("blue_block"))
    ep = Runtime(body, RulesPlanner()).run("Pick up the red block.")
    assert not ep.success
    assert ep.end_reason == "grab failed: hand full"
    assert body.world.holding is not None and body.world.holding.name == "blue_block"
    assert "grab:red_block" not in body.world.events


def test_pickup_from_the_table_still_grabs():
    body = SimBody()
    ep = Runtime(body, RulesPlanner()).run("Pick up the red block.")
    assert ep.success, ep.end_reason
    assert ep.end_reason == "completed"
    assert body.world.holding is not None and body.world.holding.name == "red_block"
    assert "grab:red_block" in body.world.events


class _Delivery(Planner):
    name = "delivery"

    def plan(self, instruction, obs, catalog):
        return [SkillCall("walk_toward", {"target": "box"}), SkillCall("release")]


def test_held_object_delivery_plan_still_runs():
    body = SimBody(_held("red_block"))
    ep = Runtime(body, _Delivery()).run("Put it in the box.")
    assert ep.success, ep.end_reason
    assert "in_box:red_block" in body.world.events
    assert body.n_locomotion > 0
