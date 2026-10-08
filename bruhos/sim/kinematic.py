"""Deterministic 2D kinematic tabletop world with fall injection."""

from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass, field

from bruhos.body.base import Body
from bruhos.schemas import Detection, Imu, LineReading, Observation

FORWARD_M = 0.05
TURN_DEG = 15.0
FOV_DEG = 35.0
RANGE_M = 2.5
STEP_S = 0.4
GRAB_REACH_M = 0.12
GRAB_CONE_DEG = 20.0
DROP_REACH_M = 0.18
KICK_REACH_M = 0.15
LOCOMOTION = {"forward", "turn_left", "turn_right", "kick_left", "kick_right"}


@dataclass
class WorldObject:
    label: str
    color: str | None
    x: float
    y: float
    in_box: bool = False

    @property
    def name(self) -> str:
        return f"{self.color}_{self.label}" if self.color else self.label


@dataclass
class World:
    objects: list[WorldObject] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0
    heading_deg: float = 0.0
    line_m: float = 0.0                 # tape ahead of the robot, consumed by walking forward
    holding: WorldObject | None = None
    fallen: bool = False
    fell_forward: bool | None = None
    events: list[str] = field(default_factory=list)

    @classmethod
    def tabletop(cls, line_m: float = 0.0) -> "World":
        """Default scene: red/blue/green blocks, a box and a ball."""
        return cls(objects=[
            WorldObject("block", "red", 0.6, 0.25),
            WorldObject("block", "blue", 0.7, -0.3),
            WorldObject("block", "green", -0.5, 0.4),
            WorldObject("box", None, 0.9, 0.0),
            WorldObject("ball", "yellow", 0.4, -0.6),
        ], line_m=line_m)

    def get(self, name: str) -> WorldObject | None:
        return next((o for o in self.objects if o.name == name), None)

    def relative(self, o: WorldObject) -> tuple[float, float]:
        dx, dy = o.x - self.x, o.y - self.y
        dist = math.hypot(dx, dy)
        bearing = math.degrees(math.atan2(dy, dx)) - self.heading_deg
        bearing = (bearing + 180.0) % 360.0 - 180.0
        return dist, bearing


class SimBody(Body):
    name = "sim"
    turn_step_deg = TURN_DEG

    def __init__(self, world: World | None = None, fall_prob: float = 0.0,
                 falls_at: set[int] | None = None, seed: int = 0):
        self.world = world or World.tabletop()
        self.fall_prob = fall_prob
        self.falls_at = set(falls_at or ())
        self.rng = random.Random(seed)
        self.t = 0.0
        self.frame = 0
        self.n_locomotion = 0
        self._halted = threading.Event()

    def observe(self) -> Observation:
        w = self.world
        self.frame += 1
        objs = []
        for o in w.objects:
            if o.in_box or o is w.holding or w.fallen:
                continue
            dist, bearing = w.relative(o)
            if dist <= RANGE_M and abs(bearing) <= FOV_DEG:
                objs.append(Detection(o.label, o.color, round(bearing, 2), round(dist, 3)))
        imu = Imu(pitch_deg=(80.0 if w.fell_forward else -80.0) if w.fallen else 0.0,
                  fallen=w.fallen, fell_forward=w.fell_forward if w.fallen else None)
        line = LineReading(visible=w.line_m > 0 and not w.fallen, offset=0.0)
        return Observation(t=round(self.t, 3), frame=self.frame, objects=objs, imu=imu, line=line,
                           holding=w.holding.name if w.holding else None)

    def act(self, primitive: str) -> bool:
        w = self.world
        self.t += STEP_S
        if self._halted.is_set():
            return False
        if w.fallen and primitive not in {"stand_up_front", "stand_up_back"}:
            return False
        if primitive in LOCOMOTION:
            self.n_locomotion += 1
            if self.n_locomotion in self.falls_at or self.rng.random() < self.fall_prob:
                w.fallen, w.fell_forward = True, self.rng.random() < 0.5
                w.events.append("fell")
                if w.holding:
                    w.holding.x, w.holding.y = w.x, w.y
                    w.holding = None
                return True
        return getattr(self, f"_p_{primitive}")()

    def halt(self) -> None:
        self._halted.set()

    def resume(self) -> None:
        self._halted.clear()

    # ---- primitives ----
    def _p_stand(self) -> bool:
        return True

    def _p_forward(self) -> bool:
        w = self.world
        r = math.radians(w.heading_deg)
        w.x += FORWARD_M * math.cos(r)
        w.y += FORWARD_M * math.sin(r)
        w.line_m = max(0.0, w.line_m - FORWARD_M)
        if w.holding:
            w.holding.x, w.holding.y = w.x, w.y
        return True

    def _p_turn_left(self) -> bool:
        self.world.heading_deg = (self.world.heading_deg + TURN_DEG) % 360.0
        return True

    def _p_turn_right(self) -> bool:
        self.world.heading_deg = (self.world.heading_deg - TURN_DEG) % 360.0
        return True

    def _reachable(self, reach: float, cone: float, pred) -> WorldObject | None:
        w, best = self.world, None
        for o in w.objects:
            if o.in_box or o is w.holding or not pred(o):
                continue
            dist, bearing = w.relative(o)
            if dist <= reach and abs(bearing) <= cone and (best is None or dist < best[0]):
                best = (dist, o)
        return best[1] if best else None

    def _p_grab(self) -> bool:
        w = self.world
        if w.holding:
            return False
        o = self._reachable(GRAB_REACH_M, GRAB_CONE_DEG, lambda o: o.label not in {"box", "ball"})
        if not o:
            return False
        w.holding = o
        w.events.append(f"grab:{o.name}")
        return True

    def _p_release(self) -> bool:
        w = self.world
        if not w.holding:
            return False
        o, w.holding = w.holding, None
        box = self._reachable(DROP_REACH_M, 30.0, lambda b: b.label == "box")
        if box:
            o.in_box = True
            o.x, o.y = box.x, box.y
            w.events.append(f"in_box:{o.name}")
        else:
            r = math.radians(w.heading_deg)
            o.x, o.y = w.x + 0.08 * math.cos(r), w.y + 0.08 * math.sin(r)
            w.events.append(f"drop:{o.name}")
        return True

    def _kick(self, foot: str) -> bool:
        w = self.world
        ball = self._reachable(KICK_REACH_M, 30.0, lambda o: o.label == "ball")
        w.events.append(f"kick:{foot}")
        if ball:
            r = math.radians(w.heading_deg)
            ball.x += math.cos(r)
            ball.y += math.sin(r)
            w.events.append(f"kicked:{ball.name}")
        return True

    def _p_kick_left(self) -> bool:
        return self._kick("left")

    def _p_kick_right(self) -> bool:
        return self._kick("right")

    def _p_wave(self) -> bool:
        self.world.events.append("wave")
        return True

    def _stand_up(self, front: bool) -> bool:
        w = self.world
        if not w.fallen or w.fell_forward is not front:
            return False
        w.fallen, w.fell_forward = False, None
        w.events.append("stood_up")
        return True

    def _p_stand_up_front(self) -> bool:
        return self._stand_up(True)

    def _p_stand_up_back(self) -> bool:
        return self._stand_up(False)
