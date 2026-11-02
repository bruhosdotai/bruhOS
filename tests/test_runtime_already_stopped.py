from bruhos.brain.base import Planner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody
from bruhos.schemas import SkillCall
from bruhos.trace import verify_episode


class CountingPlanner(Planner):
    name = "counting"

    def __init__(self):
        self.calls = 0

    def plan(self, instruction, obs, catalog):
        self.calls += 1
        return [SkillCall("wave")]


def test_stopped_runtime_skips_planner_until_reset():
    brain, body = CountingPlanner(), SimBody()
    runtime = Runtime(body, brain)
    runtime.stop()
    episode = runtime.run("Wave hello.")
    assert brain.calls == 0
    assert episode.end_reason == "stopped"
    assert not episode.success and not episode.plan and not episode.steps
    assert verify_episode(episode)
    runtime.reset_stop()
    resumed = runtime.run("Wave hello.")
    assert brain.calls == 1
    assert resumed.success
    assert body.world.events == ["wave"]
