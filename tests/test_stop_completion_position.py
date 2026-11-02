import pytest

from bruhos.brain.base import Planner
from bruhos.runtime import Runtime
from bruhos.schemas import SkillCall
from bruhos.sim.kinematic import SimBody


class FixedPlanner(Planner):
    def __init__(self, names):
        self.names = names

    def plan(self, instruction, obs, catalog):
        return [SkillCall(name) for name in self.names]


def test_early_stop_does_not_claim_later_work_was_completed():
    body = SimBody()
    episode = Runtime(body, FixedPlanner(["stop", "wave", "stop"])).run("Wave, then stop.")
    assert episode.end_reason == "stopped"
    assert body.world.events == []
    assert len(episode.steps) == 1
    assert not episode.success


@pytest.mark.parametrize("names", [["stop"], ["wave", "stop"]])
def test_an_actually_terminal_stop_still_completes_the_plan(names):
    episode = Runtime(SimBody(), FixedPlanner(names)).run("Stop after planned work.")
    assert episode.success
    assert episode.end_reason == "stopped"
    assert len(episode.steps) == len(names)
