"""A malformed bridge payload must not kill observe().

Failing-test-first for a reproduced defect: ElizaBridgeBody.observe() calls
float() directly on telemetry.basic imu_roll/imu_pitch and to_detection()
directly on every telemetry.perception entity. One non-numeric value (a
string, null, a list) or one non-dict entity raises ValueError/TypeError/
AttributeError out of observe(). Runtime calls observe() with no guard, so
the whole episode crashes instead of running with the last good reading.

#258 fixed only the non-dict telemetry.basic container; the values inside
the dict and the perception entities are still unguarded.
"""

import json
import threading
import time

import pytest

pytest.importorskip("websockets")
from websockets.sync.server import serve

from bruhos.body.eliza_bridge import ElizaBridgeBody
from bruhos.brain import make_planner
from bruhos.runtime import Runtime

PROFILE = {"profile": {"actions": {"groups": {"stand": {"duration_s": 0.2}, "wave": {"duration_s": 0.3}}}}}


class MalformedBridge:
    """Sends one malformed telemetry.basic + telemetry.perception on connect."""

    def __init__(self, telemetry, entities):
        self.telemetry = telemetry
        self.entities = entities
        self.server = serve(self._handle, "127.0.0.1", 0)
        self.url = f"ws://127.0.0.1:{self.server.socket.getsockname()[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _event(self, ws, event, data):
        ws.send(json.dumps({"type": "event", "event": event, "timestamp": "t",
                            "backend": "fake", "data": data}))

    def _handle(self, ws):
        self._event(ws, "session.hello", {"capabilities": {}})
        self._event(ws, "telemetry.basic", self.telemetry)
        self._event(ws, "telemetry.perception", {"entities": self.entities})
        for raw in ws:
            msg = json.loads(raw)
            data = PROFILE if msg["command"] == "profile.describe" else {}
            ws.send(json.dumps({"type": "response", "request_id": msg["request_id"],
                                "timestamp": "t", "ok": True, "backend": "fake",
                                "message": "ok", "data": data}))

    def close(self):
        self.server.shutdown()


def _body(bridge):
    cfg = {"body": {"url": bridge.url, "fall_deg": 28.0},
           "walk": {"forward_s": 0.3, "turn_s": 0.2, "heartbeat_s": 0.05},
           "actions": {"stand": "stand", "wave": "wave"}}
    body = ElizaBridgeBody(cfg)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not body._telemetry:
        time.sleep(0.02)
    return body


def test_non_numeric_imu_values_do_not_crash_observe():
    bridge = MalformedBridge({"imu_roll": "bad", "imu_pitch": None}, [])
    try:
        obs = _body(bridge).observe()
    finally:
        bridge.close()
    assert obs.imu.roll_deg == 0.0
    assert obs.imu.pitch_deg == 0.0
    assert not obs.imu.fallen


def test_non_finite_imu_values_are_treated_as_zero():
    bridge = MalformedBridge({"imu_roll": "nan", "imu_pitch": float("inf")}, [])
    try:
        obs = _body(bridge).observe()
    finally:
        bridge.close()
    assert obs.imu.roll_deg == 0.0
    assert obs.imu.pitch_deg == 0.0
    assert not obs.imu.fallen


def test_malformed_entities_are_skipped_and_good_ones_kept():
    bridge = MalformedBridge(
        {"imu_roll": 0.0, "imu_pitch": 0.0},
        [{"label": "red ball", "x": 1.0, "y": 1.0, "confidence": 0.9},
         {"label": "blue block", "x": "far", "y": 1.0},
         {"label": "green block", "x": 1.0, "y": 1.0, "confidence": "high"},
         "not-a-dict"])
    try:
        obs = _body(bridge).observe()
    finally:
        bridge.close()
    assert [o.name for o in obs.objects] == ["red_ball"]


def test_non_list_entities_do_not_crash_observe():
    bridge = MalformedBridge({"imu_roll": 0.0, "imu_pitch": 0.0}, "red ball")
    try:
        obs = _body(bridge).observe()
    finally:
        bridge.close()
    assert obs.objects == []


def test_episode_survives_malformed_telemetry_end_to_end():
    bridge = MalformedBridge({"imu_roll": "bad", "imu_pitch": None},
                             [{"label": "red ball", "x": "far", "y": 1.0}])
    try:
        body = _body(bridge)
        ep = Runtime(body, make_planner("rules")).run("Wave hello.")
    finally:
        bridge.close()
    assert ep.success
    assert ep.end_reason == "completed"
