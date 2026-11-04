import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody
from bruhos.skills import CATALOG


@pytest.mark.parametrize("instruction", [
    "What can you see?",
    "What can you see around you?",
    "What can you see here?",
    "What is around?",
    "What is around you?",
    "What is around here?",
    # Existing supported phrasings
    "What do you see?",
    "Look around.",
    "What's around?",
    "Scan.",
])
def test_open_detection_phrasings_plan_targetless_detect(instruction):
    planner = RulesPlanner()
    obs = SimBody().observe()
    plan = planner.plan(instruction, obs, CATALOG)
    assert len(plan) == 1
    assert plan[0].skill == "detect"
    assert plan[0].args == {}


def test_targeted_vision_queries_still_target_object():
    planner = RulesPlanner()
    obs = SimBody().observe()
    # Targeted queries must route to detect(target=...)
    plan = planner.plan("Can you see the red block?", obs, CATALOG)
    assert len(plan) == 1
    assert plan[0].skill == "detect"
    assert plan[0].args == {"target": "red_block"}


@pytest.mark.parametrize("instruction", [
    "What can you see?",
    "What is around you?",
])
def test_runtime_executes_detect_query_successfully(instruction):
    body = SimBody()
    planner = RulesPlanner()
    ep = Runtime(body, planner).run(instruction)
    assert ep.success, ep.end_reason
    assert ep.end_reason == "completed"
    assert len(ep.plan) == 1
    assert ep.plan[0]["skill"] == "detect"
    assert len(ep.steps) == 1
    assert ep.steps[0].kind == "skill"
    assert ep.steps[0].outcome["status"] == "ok"
