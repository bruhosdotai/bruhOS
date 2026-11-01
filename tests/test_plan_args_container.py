import pytest

from bruhos.brain.base import Planner, parse_plan
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody


class JsonPlanner(Planner):
    def __init__(self, call):
        self.call = call

    def plan(self, instruction, obs, catalog):
        return parse_plan({"plan": [self.call]})


@pytest.mark.parametrize("args", [False, 0, [], "", ["unexpected"], "unexpected"])
def test_non_object_args_are_rejected_instead_of_executing_a_default_call(args):
    body = SimBody()
    episode = Runtime(body, JsonPlanner({"skill": "wave", "args": args})).run("Wave hello.")
    assert not episode.success
    assert episode.steps[0].kind == "rejected"
    assert body.world.events == []


@pytest.mark.parametrize("call", [{"skill": "wave"}, {"skill": "wave", "args": None},
                                  {"skill": "wave", "args": {}}])
def test_missing_null_and_object_args_remain_supported(call):
    body = SimBody()
    episode = Runtime(body, JsonPlanner(call)).run("Wave hello.")
    assert episode.success
    assert body.world.events == ["wave"]
