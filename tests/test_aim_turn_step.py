"""look_at / walk_toward on a body whose calibrated turn step is wider than twice the aim deadband.

Shipped configs say ``turn_step_deg = 15.0  # calibrate``. A body that turns 20 degrees per
``turn_*`` primitive can never place a target inside the fixed 8 degree ``look_at`` deadband when
it starts 10 degrees off: every turn overshoots to the other side.
"""

from __future__ import annotations

import math

from bruhos.body.base import Body
from bruhos.schemas import Detection, Observation, SkillCall, Status
from bruhos.sim.kinematic import SimBody
from bruhos.skills import SkillContext, run_skill


class WideTurnBody(Body):
    """One target; ``turn_*`` rotates ``turn_step_deg`` degrees, ``forward`` walks 5 cm along the heading."""

    name = "wide_turn"

    def __init__(self, turn_step_deg: float, bearing_deg: float, distance_m: float = 0.5):
        self.turn_step_deg = turn_step_deg
        r = math.radians(bearing_deg)
        self.tx, self.ty = distance_m * math.cos(r), distance_m * math.sin(r)   # robot frame: +x ahead, +y left
        self.acts: list[str] = []
        self.frame = 0

    @property
    def bearing_deg(self) -> float:
        return math.degrees(math.atan2(self.ty, self.tx))

    @property
    def distance_m(self) -> float:
        return math.hypot(self.tx, self.ty)

    def observe(self) -> Observation:
        self.frame += 1
        return Observation(t=0.4 * self.frame, frame=self.frame,
                           objects=[Detection("ball", "yellow", round(self.bearing_deg, 2), round(self.distance_m, 3))])

    def act(self, primitive: str) -> bool:
        self.acts.append(primitive)
        if primitive in {"turn_left", "turn_right"}:
            sign = -1.0 if primitive == "turn_left" else 1.0      # turning left moves the target right
            r = math.radians(sign * self.turn_step_deg)
            self.tx, self.ty = (self.tx * math.cos(r) - self.ty * math.sin(r),
                                self.tx * math.sin(r) + self.ty * math.cos(r))
        elif primitive == "forward":
            self.tx -= 0.05
        return True

    def halt(self) -> None:
        pass


def test_look_at_faces_target_inside_half_a_turn_step():
    body = WideTurnBody(turn_step_deg=20.0, bearing_deg=10.0)
    out = run_skill(SkillContext(body, max_steps=120), SkillCall("look_at", {"target": "yellow_ball"}))
    assert out.status is Status.OK, out
    assert abs(body.bearing_deg) <= 10.0
    assert body.acts == []                      # any turn would overshoot to -10 degrees


def test_look_at_still_turns_when_a_turn_gets_closer():
    body = WideTurnBody(turn_step_deg=20.0, bearing_deg=25.0)
    out = run_skill(SkillContext(body, max_steps=120), SkillCall("look_at", {"target": "yellow_ball"}))
    assert out.status is Status.OK, out
    assert body.acts == ["turn_left"]
    assert abs(body.bearing_deg) <= 10.0


def test_walk_toward_walks_instead_of_oscillating():
    body = WideTurnBody(turn_step_deg=30.0, bearing_deg=13.0, distance_m=0.5)
    out = run_skill(SkillContext(body, max_steps=120), SkillCall("walk_toward", {"target": "yellow_ball"}))
    assert out.status is Status.OK, out
    assert body.distance_m <= 0.1
    assert body.acts[0] == "forward"           # 13 deg is inside half a 30 deg step: walk, do not turn


def test_sim_deadbands_unchanged():
    """TURN_DEG is 15, half of it is below both deadbands, so the kinematic sim behaves as before."""
    body = SimBody()
    body.world.heading_deg = -70.0                     # yellow_ball sits at -56.3 deg: 13.7 deg to the left
    out = run_skill(SkillContext(body), SkillCall("look_at", {"target": "yellow_ball"}))
    assert out.status is Status.OK
    assert abs(body.world.relative(body.world.get("yellow_ball"))[1]) <= 8.0
    assert body.n_locomotion == 1
