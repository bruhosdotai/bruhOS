"""A scan must cover a full revolution before reporting a target not found.

``_scan`` turns in ``turn_step_deg`` increments; when the step size does not divide
360, truncating the turn count stops the scan short of a full revolution, and a
target visible only from the uncovered wedge is reported missing even though one
more (partial) step would have found it. A body whose configured step is 100
degrees and a target at a 50-degree bearing with a +/-35-degree field of view
reproduces it: the old scan looks from 0/100/200/300 degrees and never from the
40-degree heading that would see the target.
"""

from bruhos.body.base import Body
from bruhos.schemas import Detection, Imu, LineReading, Observation, SkillCall, Status
from bruhos.skills import SkillContext, run_skill


class StepBody(Body):
    """Turns by ``turn_step_deg`` and sees a fixed target within a narrow FOV."""

    name = "step"
    turn_step_deg = 100.0
    FOV_DEG = 35.0
    TARGET_BEARING = 50.0

    def __init__(self):
        self.heading = 0.0

    def observe(self) -> Observation:
        rel = (self.TARGET_BEARING - self.heading + 180.0) % 360.0 - 180.0
        objects = [Detection("box", None, round(rel, 2), 1.0)] if abs(rel) <= self.FOV_DEG else []
        return Observation(t=0.0, frame=0, objects=objects, imu=Imu(),
                           line=LineReading(), holding=None)

    def act(self, primitive: str) -> bool:
        if primitive == "turn_left":
            self.heading = (self.heading + self.turn_step_deg) % 360.0
        return True

    def halt(self) -> None:
        pass


def test_scan_completes_the_revolution():
    body = StepBody()
    result = run_skill(SkillContext(body), SkillCall("detect", {"target": "box"}))
    assert result.status is Status.OK, result.detail
    assert result.data["found"] == "box"
