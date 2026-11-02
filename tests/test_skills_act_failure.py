from bruhos.body.base import Body
from bruhos.schemas import Imu, Observation, SkillCall, Status
from bruhos.skills import SkillContext, run_skill


class RefusingBody(Body):
    """Body simulator that fails all action primitives."""

    def observe(self):
        return Observation(t=0.0, frame=0, imu=Imu())

    def act(self, primitive: str) -> bool:
        return False

    def halt(self):
        pass


def test_wave_reports_failure_when_body_act_fails():
    ctx = SkillContext(RefusingBody())
    outcome = run_skill(ctx, SkillCall("wave"))
    assert outcome.status is Status.FAILED
    assert "could not wave" in outcome.detail


def test_kick_reports_failure_when_body_act_fails():
    ctx = SkillContext(RefusingBody())
    outcome_r = run_skill(ctx, SkillCall("kick", {"foot": "right"}))
    assert outcome_r.status is Status.FAILED
    assert "could not kick right" in outcome_r.detail

    outcome_l = run_skill(ctx, SkillCall("kick", {"foot": "left"}))
    assert outcome_l.status is Status.FAILED
    assert "could not kick left" in outcome_l.detail
