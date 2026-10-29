import json
import threading
import time

import pytest

pytest.importorskip("websockets")
from websockets.sync.server import serve  # noqa: E402

from bruhos.body.eliza_bridge import ElizaBridgeBody, to_detection  # noqa: E402
from bruhos.brain import make_planner  # noqa: E402
from bruhos.runtime import Runtime  # noqa: E402

PROFILE = {"profile": {"actions": {"groups": {"stand": {"duration_s": 0.2}, "wave": {"duration_s": 0.3}}}}}


class FakeBridge:
    """Speaks the eliza_robot command-envelope protocol on a local port."""

    def __init__(self, imu_pitch=0.0, deadman_after=None):
        self.commands: list[tuple[str, dict]] = []
        self.imu_pitch = imu_pitch
        self.deadman_after = deadman_after
        self.server = serve(self._handle, "127.0.0.1", 0)
        self.url = f"ws://127.0.0.1:{self.server.socket.getsockname()[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _event(self, ws, event, data):
        ws.send(json.dumps({"type": "event", "event": event, "timestamp": "t", "backend": "fake", "data": data}))

    def _handle(self, ws):
        self._event(ws, "session.hello", {"capabilities": {}})
        self._event(ws, "telemetry.basic", {"imu_roll": 0.0, "imu_pitch": self.imu_pitch})
        self._event(ws, "telemetry.perception", {"entities": [
            {"label": "red ball", "x": 1.0, "y": 1.0, "confidence": 0.9}]})
        for raw in ws:
            msg = json.loads(raw)
            self.commands.append((msg["command"], msg["payload"]))
            data = PROFILE if msg["command"] == "profile.describe" else {}
            ws.send(json.dumps({"type": "response", "request_id": msg["request_id"], "timestamp": "t",
                                "ok": True, "backend": "fake", "message": "ok", "data": data}))
            if msg["command"] == "action.play" and self.deadman_after is not None:
                time.sleep(self.deadman_after)
                self._event(ws, "safety.deadman_triggered", {"age_sec": 1.0})

    def names(self):
        return [c for c, _ in self.commands]

    def close(self):
        self.server.shutdown()


def _body(bridge, **walk):
    cfg = {"body": {"url": bridge.url, "fall_deg": 28.0},
           "walk": {"forward_s": 0.3, "turn_s": 0.2, "heartbeat_s": 0.05, **walk},
           "actions": {"stand": "stand", "wave": "wave"}}
    return ElizaBridgeBody(cfg)


@pytest.fixture
def bridge():
    b = FakeBridge()
    yield b
    b.close()


def test_detection_from_bridge_entity():
    d = to_detection({"label": "red ball", "x": 1.0, "y": 1.0, "confidence": 0.5})
    assert (d.label, d.color, d.bearing_deg, d.distance_m) == ("ball", "red", 45.0, 1.414)
    assert to_detection({"label": "box", "x": 2.0, "y": 0.0, "distance": 2.1}).name == "box"


def test_forward_heartbeats_then_stops(bridge):
    body = _body(bridge)
    assert body.act("forward")
    names = bridge.names()
    assert names[:3] == ["profile.describe", "walk.set", "walk.command"]
    assert names.count("walk.set") >= 3 and names[-1] == "walk.command"
    assert bridge.commands[-1][1] == {"action": "stop"}
    assert bridge.commands[1][1]["x"] == 0.02
    body.close()


def test_turns_use_yaw_only(bridge):
    body = _body(bridge, turn_yaw=4.0)
    assert body.act("turn_right")
    walk = next(p for c, p in bridge.commands if c == "walk.set")
    assert (walk["x"], walk["yaw"]) == (0.0, -4.0)
    body.close()


def test_action_group_and_unmapped_primitive(bridge):
    body = _body(bridge)
    assert body.act("wave")
    assert ("action.play", {"name": "wave"}) in bridge.commands
    before = len(bridge.commands)
    assert not body.act("grab")
    assert len(bridge.commands) == before
    body.close()


def test_never_sends_raw_servo_commands(bridge):
    body = _body(bridge)
    for p in ("stand", "forward", "turn_left", "wave", "grab", "kick_left"):
        body.act(p)
    body.halt()
    assert "servo.set" not in bridge.names()
    body.close()


def test_deadman_during_action_reports_failure():
    bridge = FakeBridge(deadman_after=0.05)
    body = _body(bridge)
    assert not body.act("wave")
    body.close()
    bridge.close()


def test_halt_stops_walking_and_blocks_motion(bridge):
    body = _body(bridge)
    body.halt()
    assert bridge.commands[-1] == ("walk.command", {"action": "stop"})
    assert not body.act("forward")
    body.resume()
    assert body.act("forward")
    body.close()


def test_fall_detected_from_bridge_imu():
    bridge = FakeBridge(imu_pitch=0.9)
    body = _body(bridge)
    time.sleep(0.1)
    obs = body.observe()
    assert obs.imu.fallen and obs.imu.fell_forward
    assert obs.find("red") is not None
    body.close()
    bridge.close()


def test_runtime_runs_on_bridge(bridge):
    body = _body(bridge)
    ep = Runtime(body, make_planner("rules")).run("Wave hello.")
    assert ep.success
    assert ("action.play", {"name": "wave"}) in bridge.commands
    body.close()
