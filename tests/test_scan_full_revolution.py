"""_scan must sweep a full revolution when the turn step does not divide 360.

``_scan`` counted its turns with ``int(360 / turn_step_deg)``, which truncates:
with a 100-degree turn step it turned 3 times (300 degrees) and reported
"not found after full scan" for a target at bearing 50 degrees, which a full
revolution would have seen. Coarse steps are in the supported range: load-time
validation allows turn steps up to 180 degrees.
"""

from bruhos.body.base import Body
from bruhos.schemas import Detection, Imu, Observation, SkillCall, Status
from bruhos.skills import SkillContext, run_skill


class CoarseTurnBody(Body):
    """Turns exactly turn_step_deg per turn_left; 35-degree FOV like the sim."""

    name = "coarse"
    turn_step_deg = 100.0

    def __init__(self, target_bearing=50.0):
        self.heading = 0.0
        self.target_bearing = target_bearing
        self.turns = 0

    def observe(self):
        b = (self.target_bearing - self.heading + 180) % 360 - 180
        objs = [Detection("block", "red", round(b, 2), 0.5)] if abs(b) <= 35 else []
        return Observation(t=0, frame=0, objects=objs, imu=Imu())

    def act(self, primitive):
        if primitive == "turn_left":
            self.heading = (self.heading + self.turn_step_deg) % 360
            self.turns += 1
            return True
        return False

    def halt(self):
        pass


def test_scan_finds_target_past_the_truncated_sweep():
    """A target at bearing 50 needs the 4th turn (heading 40) to enter view."""
    body = CoarseTurnBody(target_bearing=50.0)
    out = run_skill(SkillContext(body), SkillCall("detect", {"target": "red_block"}))
    assert out.status is Status.OK
    assert out.detail == "found red_block"
    assert body.turns == 4


def test_scan_still_stops_after_one_revolution_when_target_absent():
    body = CoarseTurnBody(target_bearing=150.0)
    out = run_skill(SkillContext(body), SkillCall("detect", {"target": "red_block"}))
    assert out.status is Status.FAILED
    assert "not found after full scan" in out.detail
    assert body.turns == 4
