import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody
from bruhos.skills import CATALOG


@pytest.mark.parametrize("instruction", [
    "Put it down.",
    "Put the block down.",
    "Put down the block.",
    "Set it down.",
    "Set down the red block.",
    "Place it down.",
    "Put it on the table.",
    "Put it on the floor.",
    "Put it on the ground.",
])
def test_put_down_instruction_plans_release(instruction):
    planner = RulesPlanner()
    obs = SimBody().observe()
    plan = planner.plan(instruction, obs, CATALOG)
    assert len(plan) == 1
    assert plan[0].skill == "release"
    assert plan[0].args == {}


def test_runtime_put_down_when_holding_drops_object():
    body = SimBody()
    body.world.holding = body.world.get("red_block")
    ep = Runtime(body, RulesPlanner()).run("Put it down.")
    assert ep.success, ep.end_reason
    assert ep.end_reason == "completed"
    assert "drop:red_block" in body.world.events
    assert body.world.holding is None


def test_runtime_put_down_when_empty_hand_fails_honestly():
    body = SimBody()
    ep = Runtime(body, RulesPlanner()).run("Put it down.")
    assert not ep.success
    assert ep.end_reason == "release failed: hand empty"


def test_into_box_and_other_commands_unaffected():
    planner = RulesPlanner()
    obs = SimBody().observe()
    # Destination into box still plans delivery, not in-place release
    delivery_plan = planner.plan("Put the red one in the box.", obs, CATALOG)
    assert [c.skill for c in delivery_plan] == ["detect", "walk_toward", "grab", "walk_toward", "release"]
    # Drop and release controls
    assert [c.skill for c in planner.plan("Drop it.", obs, CATALOG)] == ["release"]
    # Unrelated instructions
    assert planner.plan("Sit down.", obs, CATALOG) == []
    assert planner.plan("Do a backflip.", obs, CATALOG) == []
