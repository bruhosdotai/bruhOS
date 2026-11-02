import math

import pytest

from bruhos.brain.base import Planner
from bruhos.runtime import Runtime, RuntimeConfig
from bruhos.schemas import SkillCall
from bruhos.sim.kinematic import SimBody, World, WorldObject


class PickupPlanner(Planner):
    def plan(self, instruction, obs, catalog):
        return [SkillCall("walk_toward", {"target": "red_block"}), SkillCall("grab")]


@pytest.mark.parametrize("bearing", [-30, 30])
def test_nearby_target_is_faced_before_walk_reports_reached(bearing):
    angle = math.radians(bearing)
    world = World(objects=[WorldObject("block", "red", 0.08 * math.cos(angle), 0.08 * math.sin(angle))])
    episode = Runtime(SimBody(world), PickupPlanner(), config=RuntimeConfig(max_replans=0)).run("Pick up red.")
    assert episode.success, episode.end_reason
    assert world.holding is world.objects[0]


def test_already_aligned_target_requires_no_locomotion():
    world = World(objects=[WorldObject("block", "red", 0.08, 0)])
    body = SimBody(world)
    episode = Runtime(body, PickupPlanner(), config=RuntimeConfig(max_replans=0)).run("Pick up red.")
    assert episode.success
    assert body.n_locomotion == 0
