"""follow_line must not report success when the line is lost before any progress.

Failing-test-first for a reproduced defect: on base, a body that loses sight of
the line right after the first turn (offset beyond the deadband, then the line
gone) made ``follow_line(steps=20)`` return ``Status.OK / "end of line"`` with
``steps == 0`` — success sealed into the trace for line-following that never
happened. Losing the line after real forward progress is still the designed
"end of line" success; only the zero-progress loss is a failure.
"""

from bruhos.body.base import Body
from bruhos.brain.base import Planner
from bruhos.runtime import Runtime
from bruhos.schemas import Imu, LineReading, Observation, SkillCall, Status
from bruhos.sim.kinematic import SimBody, World
from bruhos.skills import SkillContext, run_skill


class LosingLineBody(Body):
    """Line in view at first, then gone as soon as the robot turns."""

    name = "losing-line"
    turn_step_deg = 15.0

    def __init__(self):
        self.visible = True

    def observe(self):
        return Observation(t=0.0, frame=0, imu=Imu(),
                           line=LineReading(visible=self.visible, offset=0.5))

    def act(self, primitive):
        self.visible = False
        return True

    def halt(self):
        pass


class FixedPlanner(Planner):
    name = "fixed"

    def __init__(self, plan):
        self._plan = plan

    def plan(self, instruction, obs, catalog):
        return list(self._plan)


def test_follow_line_lost_before_any_step_fails():
    out = run_skill(SkillContext(LosingLineBody()), SkillCall("follow_line", {"steps": 20}))
    assert out.status is Status.FAILED
    assert out.detail == "lost line"
    assert out.data["steps"] == 0


def test_follow_line_end_of_line_after_progress_still_ok():
    # Short tape: one forward step consumes it, then the line is genuinely ended.
    body = SimBody(World.tabletop(line_m=0.05))
    out = run_skill(SkillContext(body), SkillCall("follow_line", {"steps": 20}))
    assert out.status is Status.OK
    assert out.detail == "end of line"
    assert out.data["steps"] >= 1


def test_follow_line_zero_steps_noop_unchanged():
    body = SimBody(World.tabletop(line_m=0.6))
    out = run_skill(SkillContext(body), SkillCall("follow_line", {"steps": 0}))
    assert out.status is Status.OK
    assert out.detail == "followed 0 steps"


def test_runtime_does_not_report_lost_line_as_success():
    body = LosingLineBody()
    ep = Runtime(body, FixedPlanner([SkillCall("follow_line", {"steps": 20})])).run("Follow the line.")
    assert not ep.success
    # The first executed step must be the lost-line failure itself, not a false
    # success. (The runtime then replans once; the retry fails at entry with
    # "no line in view", which is what the final end_reason reports.)
    assert ep.steps[0].outcome["status"] == "failed"
    assert ep.steps[0].outcome["detail"] == "lost line"
