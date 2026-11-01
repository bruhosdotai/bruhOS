import pytest

from bruhos.brain.base import Planner
from bruhos.runtime import Runtime
from bruhos.schemas import SkillCall
from bruhos.sim.kinematic import FORWARD_M, SimBody, World
from bruhos.trace import verify_episode


class FiveStepsPlanner(Planner):
    name = "five_steps"

    def plan(self, instruction, obs, catalog):
        return [SkillCall("follow_line", {"steps": 5})]


@pytest.mark.parametrize("falls_at", [set(), {1}, {3}, {5}, {3, 5}])
def test_follow_line_resumes_only_the_unfinished_steps(falls_at):
    body = SimBody(World(line_m=1.0), falls_at=falls_at)
    ep = Runtime(body, FiveStepsPlanner()).run("Follow the line for five steps.")

    assert ep.success, ep.end_reason
    assert ep.falls == ep.recoveries == len(falls_at)
    assert body.world.x == pytest.approx(5 * FORWARD_M)
    assert body.world.line_m == pytest.approx(1.0 - 5 * FORWARD_M)
    assert sum(s.outcome["data"].get("steps", 0) for s in ep.steps if s.kind == "skill") == 5
    assert ep.plan == [{"skill": "follow_line", "args": {"steps": 5}}]
    assert verify_episode(ep)
