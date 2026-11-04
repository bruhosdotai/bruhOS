import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import SkillCall
from bruhos.sim.kinematic import SimBody, World


def _plan(instruction: str) -> list[SkillCall]:
    obs = SimBody().observe()
    return RulesPlanner().plan(instruction, obs, [])


@pytest.mark.parametrize("instruction", [
    "Stand back up.",
    "stand back up",
    "Please stand back up.",
    "Stand back up now.",
    "Stand up.",
    "Get up.",
    "Get back up.",
])
def test_stand_up_phrasing_plans_stand_up(instruction: str):
    plan = _plan(instruction)
    assert len(plan) == 1
    assert plan[0].skill == "stand_up"
    assert plan[0].args == {}


def test_fallen_robot_recovers_on_stand_back_up():
    world = World(fallen=True, fell_forward=False)
    body = SimBody(world)
    ep = Runtime(body, RulesPlanner()).run("Stand back up.")

    assert ep.success is True
    assert ep.end_reason == "completed"
    assert ep.recoveries == 1
    assert ep.plan == [{"skill": "stand_up", "args": {}}]
    assert "stood_up" in body.world.events
    assert not body.world.fallen
