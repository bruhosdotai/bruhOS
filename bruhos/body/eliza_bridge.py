"""elizaOS robot bridge body: any robot behind ``eliza_robot.bridge.server`` (elizaOS/research).

The bridge fronts a mock backend, a MuJoCo sim or a real robot (first profile: Hiwonder AiNex).
bruhOS only sends ``action.play``, ``walk.set`` and ``walk.command`` — never ``servo.set``, so a
brain still cannot reach raw joints through this body.

The bridge stops motion if no motion command arrives within its deadman timeout (1 s on AiNex),
so walking primitives re-send ``walk.set`` while they run.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from bruhos._net import require_secure
from bruhos.body.base import PRIMITIVES, Body
from bruhos.schemas import Detection, Imu, Observation

log = logging.getLogger(__name__)

WALK_PRIMITIVES = ("forward", "turn_left", "turn_right")
COLORS = {"red", "green", "blue", "yellow", "orange", "purple", "black", "white", "pink"}
TOKEN_ENV = "ELIZA_AINEX_BRIDGE_AUTH_TOKEN"


class BridgeError(RuntimeError):
    pass


def _num(value: Any) -> float | None:
    """Finite float from a bridge field, or ``None`` if it is absent, unparseable or non-finite."""
    try:
        f = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def to_detection(entity: dict[str, Any]) -> Detection | None:
    """Bridge perception entity (robot frame: +x forward, +y left) -> bruhOS detection.

    ``None`` when the entity carries no usable position: an entity placed at the
    origin from a garbage coordinate reads as "in the robot's hand" and makes
    ``walk_toward`` report arrival without moving. An unusable distance stays
    unknown (``None``) rather than zero for the same reason.
    """
    if not isinstance(entity, dict):
        entity = {}
    words = str(entity.get("label", "object")).lower().split()
    color = words[0] if len(words) > 1 and words[0] in COLORS else None
    label = "_".join(words[1:] if color else words) or "object"
    x = _num(entity["x"]) if "x" in entity else 0.0
    y = _num(entity["y"]) if "y" in entity else 0.0
    raw_dist = entity.get("distance")
    dist = _num(raw_dist)
    if x is None or y is None:
        return None
    if "x" not in entity and "y" not in entity and dist is None:
        return None
    if raw_dist is None:
        distance = round(math.hypot(x, y), 3)
    else:
        distance = round(dist, 3) if dist is not None else None
    confidence = _num(entity.get("confidence")) if "confidence" in entity else 1.0
    return Detection(
        label=label,
        color=color,
        bearing_deg=round(math.degrees(math.atan2(y, x)), 1) if (x or y) else 0.0,
        distance_m=distance,
        confidence=confidence if confidence is not None else 1.0,
    )


class ElizaBridgeBody(Body):
    name = "eliza_bridge"

    def __init__(self, cfg: dict[str, Any]):
        body = cfg.get("body", {})
        self.name = body.get("name", "eliza_bridge")
        self.url = body.get("url", "ws://127.0.0.1:9100")
        self.turn_step_deg = float(body.get("turn_step_deg", 15.0))
        self.timeout = float(body.get("timeout_s", 5.0))
        self.fall_deg = float(body.get("fall_deg", 55.0))
        self.forward_sign = int(body.get("forward_sign", 1))
        self.allow_insecure = bool(body.get("allow_insecure", False))
        walk = cfg.get("walk", {})
        self.walk = {
            "speed": int(walk.get("speed", 2)),
            "height": float(walk.get("height", 0.036)),
            "forward_x": float(walk.get("forward_x", 0.02)),
            "turn_yaw": float(walk.get("turn_yaw", 5.0)),
            "forward_s": float(walk.get("forward_s", 1.5)),
            "turn_s": float(walk.get("turn_s", 1.0)),
            "heartbeat_s": float(walk.get("heartbeat_s", 0.25)),
        }
        self.actions: dict[str, str | None] = {p: None for p in PRIMITIVES} | cfg.get("actions", {})
        self._halted = threading.Event()
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._pending: dict[str, tuple[threading.Event, list[dict[str, Any]]]] = {}
        self._telemetry: dict[str, Any] = {}
        self._entities: list[dict[str, Any]] = []
        self._deadman = threading.Event()
        self.frame = 0
        self.holding: str | None = None
        self.hello: dict[str, Any] = {}
        self._connect()
        self.durations = self._action_durations()

    def _connect(self) -> None:
        token = os.environ.get(TOKEN_ENV)
        if token:
            require_secure(self.url, TOKEN_ENV, self.allow_insecure)
        from websockets.sync.client import connect

        headers = {"Authorization": f"Bearer {token}"} if token else None
        self.ws = connect(self.url, additional_headers=headers, open_timeout=self.timeout).__enter__()
        self._reader = threading.Thread(target=self._read, name="eliza-bridge-reader", daemon=True)
        self._reader.start()

    def _read(self) -> None:
        try:
            for raw in self.ws:
                msg = json.loads(raw)
                if not isinstance(msg, dict):
                    continue
                if msg.get("type") == "response":
                    slot = self._pending.get(msg.get("request_id", ""))
                    if slot:
                        slot[1].append(msg)
                        slot[0].set()
                    continue
                self._handle_event(msg.get("event"), msg.get("data"))
        except Exception:
            log.exception("eliza bridge reader stopped; telemetry is frozen until reconnect")
        finally:
            self._closed.set()
            for done, _ in list(self._pending.values()):
                done.set()

    def _handle_event(self, event: Any, data: Any) -> None:
        """Feed one bridge event into body state. Bridge peers are not trusted: malformed
        frames degrade to defaults instead of killing the reader or poisoning observations."""
        if not isinstance(data, dict):
            data = {}
        if event == "session.hello":
            self.hello = data
        elif event == "telemetry.basic":
            self._telemetry = data
        elif event == "telemetry.perception":
            self._entities = [e for e in data.get("entities") or [] if isinstance(e, dict)]
        elif event == "safety.deadman_triggered":
            self._deadman.set()

    def _send(self, command: str, payload: dict[str, Any], preempt: bool = False) -> dict[str, Any]:
        rid = str(uuid.uuid4())
        done: threading.Event = threading.Event()
        box: list[dict[str, Any]] = []
        self._pending[rid] = (done, box)
        try:
            self.ws.send(json.dumps({
                "type": "command", "request_id": rid, "timestamp": datetime.now(UTC).isoformat(),
                "command": command, "payload": payload, "preempt": preempt,
            }))
            if not done.wait(self.timeout) or not box:
                raise BridgeError(f"{command}: no response from bridge")
            return box[0]
        finally:
            self._pending.pop(rid, None)

    def _action_durations(self) -> dict[str, float]:
        try:
            resp = self._send("profile.describe", {})
        except BridgeError:
            return {}
        groups = resp.get("data", {}).get("profile", {}).get("actions", {}).get("groups", {})
        return {k: float(v.get("duration_s", 1.0)) for k, v in groups.items() if isinstance(v, dict)}

    def observe(self) -> Observation:
        self.frame += 1
        tel = self._telemetry
        roll = math.degrees(_num(tel.get("imu_roll", 0.0)) or 0.0)
        pitch = math.degrees(_num(tel.get("imu_pitch", 0.0)) or 0.0)
        fallen = abs(roll) > self.fall_deg or abs(pitch) > self.fall_deg
        imu = Imu(roll_deg=round(roll, 1), pitch_deg=round(pitch, 1), fallen=fallen,
                  fell_forward=(pitch * self.forward_sign > 0) if fallen else None)
        return Observation(t=round(time.monotonic(), 3), frame=self.frame,
                           objects=[d for e in self._entities if (d := to_detection(e)) is not None], imu=imu, holding=self.holding)

    def act(self, primitive: str) -> bool:
        if self._halted.is_set() or self._closed.is_set():
            return False
        with self._lock:
            self._deadman.clear()
            try:
                ok = self._walk(primitive) if primitive in WALK_PRIMITIVES else self._play(primitive)
            except BridgeError:
                return False
        if ok and primitive == "grab":
            self.holding = "object"
        elif ok and primitive == "release":
            self.holding = None
        return ok

    def _play(self, primitive: str) -> bool:
        group = self.actions.get(primitive)
        if not group:
            return False
        if not self._send("action.play", {"name": group}).get("ok"):
            return False
        end = time.monotonic() + self.durations.get(group, 1.0)
        while time.monotonic() < end:
            if self._halted.is_set() or self._deadman.is_set():
                return False
            time.sleep(0.05)
        return True

    def _walk(self, primitive: str) -> bool:
        w = self.walk
        x = w["forward_x"] if primitive == "forward" else 0.0
        yaw = {"turn_left": w["turn_yaw"], "turn_right": -w["turn_yaw"]}.get(primitive, 0.0)
        params = {"speed": w["speed"], "height": w["height"], "x": x, "y": 0.0, "yaw": yaw}
        if not self._send("walk.set", params).get("ok"):
            return False
        if not self._send("walk.command", {"action": "start"}).get("ok"):
            return False
        ok = True
        end = time.monotonic() + (w["forward_s"] if primitive == "forward" else w["turn_s"])
        try:
            while time.monotonic() < end:
                if self._halted.is_set() or self._deadman.is_set():
                    ok = False
                    break
                time.sleep(w["heartbeat_s"])
                if not self._send("walk.set", params).get("ok"):
                    ok = False
                    break
        finally:
            ok = self._send("walk.command", {"action": "stop"}, preempt=True).get("ok", False) and ok
        return ok

    def halt(self) -> None:
        self._halted.set()
        if not self._closed.is_set():
            try:
                self._send("walk.command", {"action": "stop"}, preempt=True)
            except BridgeError:
                pass

    def resume(self) -> None:
        self._halted.clear()

    def close(self) -> None:
        self._closed.set()
        self.ws.close()
