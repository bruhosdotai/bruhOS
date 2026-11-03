"""``kick`` and ``wave`` must not certify a primitive the body refused.

``Body.act`` returns False when the primitive did not run: the bridge is halted or closed, the
robot has no action group configured for that primitive, or the control link errored. Both
skills used to drop that return value and report ``ok``, so a step in the trace claimed
"kicked right" for a robot that never moved — while ``grab``/``release`` already handled it.
"""

import pytest

from bruhos.body.base import Body
from bruhos.runtime import Runtime
from bruhos.schemas import Imu, Observation, SkillCall, Status
from bruhos.skills.registry import SkillContext, run_skill
from bruhos.sim.kinematic import SimBody, World


class _RefusingBody(Body):
    """Upright body that refuses every primitive, like a bridge with no action group."""

    name = "refusing"
    turn_step_deg = 15.0

    def __init__(self):
        self.frame = 0

    def observe(self) -> Observation:
        self.frame += 1
        return Observation(t=float(self.frame), frame=self.frame, imu=Imu())

    def act(self, primitive: str) -> bool:
        return False

    def halt(self) -> None:
        pass


@pytest.mark.parametrize("call", [
    SkillCall("kick", {}),
    SkillCall("kick", {"foot": "left"}),
    SkillCall("wave", {}),
])
def test_refused_primitive_is_not_ok(call):
    out = run_skill(SkillContext(_RefusingBody()), call)
    assert out.status is Status.FAILED
    assert "refused" in out.detail


def test_grab_and_release_already_reported_refusal():
    body = _RefusingBody()
    assert run_skill(SkillContext(body), SkillCall("grab", {})).status is Status.FAILED
    assert run_skill(SkillContext(body), SkillCall("release", {})).status is Status.FAILED


def test_episode_with_a_refused_kick_is_not_a_success():
    class _KickPlanner:
        name, version = "fixed", "1"

        def plan(self, instruction, obs, catalog):
            return [SkillCall("kick", {"foot": "right"})]

        def replan(self, instruction, obs, catalog, history):
            return self.plan(instruction, obs, catalog)

    ep = Runtime(_RefusingBody(), _KickPlanner()).run("Kick the ball.")
    assert ep.success is False
    assert "kick_right" in ep.end_reason


def test_a_real_body_still_reports_ok():
    body = SimBody(world=World.tabletop())
    assert run_skill(SkillContext(body), SkillCall("kick", {"foot": "right"})).status is Status.OK
    assert run_skill(SkillContext(body), SkillCall("wave", {})).status is Status.OK
