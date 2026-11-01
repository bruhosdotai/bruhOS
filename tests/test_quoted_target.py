import pytest

from bruhos.brain.base import Planner, parse_plan
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody, World, WorldObject


class TextPlanner(Planner):
    def plan(self, instruction, obs, catalog):
        return parse_plan([instruction])


@pytest.mark.parametrize("target", ["001", "1e2", "nan"])
def test_quoted_target_keeps_its_literal_name(target):
    body = SimBody(World(objects=[WorldObject(target, None, 0.05, 0)]))
    episode = Runtime(body, TextPlanner()).run(f"detect('{target}')")
    assert episode.success, episode.end_reason
    assert episode.plan[0]["args"]["target"] == target
    assert episode.steps[0].outcome["data"]["found"] == target
