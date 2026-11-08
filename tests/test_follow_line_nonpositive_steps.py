"""follow_line must fail a non-positive step count instead of certifying success.

``follow_line(steps=0)`` (or a negative count, which the rules planner passes
through once open #100 lands) returns OK "followed 0 steps" without moving:
the loop breaks at once because ``done >= steps`` and ``done < steps`` is
false, so an episode that did no work is certified a success. A skill that
follows zero steps has done nothing; it must fail.
"""

import pytest

from bruhos.brain.base import Planner
from bruhos.runtime import Runtime
from bruhos.schemas import SkillCall, Status
from bruhos.sim.kinematic import SimBody, World
from bruhos.skills import SkillContext, run_skill


def _body():
    return SimBody(World.tabletop(line_m=1.0))


@pytest.mark.parametrize("steps", [0, -5])
def test_follow_line_nonpositive_steps_fails(steps):
    body = _body()
    out = run_skill(SkillContext(body, max_steps=50), SkillCall("follow_line", {"steps": steps}))
    assert out.status is Status.FAILED
    assert "steps" in out.detail
    assert out.data["steps"] == steps
    assert body.world.line_m == 1.0  # no motion at all


def test_follow_line_zero_steps_fails_end_to_end():
    class ZeroStepPlanner(Planner):
        name = "zero-step"

        def plan(self, instruction, obs, catalog):
            return [SkillCall("follow_line", {"steps": 0})]

    ep = Runtime(_body(), ZeroStepPlanner()).run("Follow the line for 0 steps.")
    assert not ep.success
    assert ep.end_reason.startswith("follow_line failed")


def test_follow_line_one_step_still_succeeds():
    body = _body()
    out = run_skill(SkillContext(body, max_steps=50), SkillCall("follow_line", {"steps": 1}))
    assert out.status is Status.OK
    assert out.data["steps"] == 1
    assert body.world.line_m == pytest.approx(0.95)
