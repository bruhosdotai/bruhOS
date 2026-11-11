"""A non-dict `telemetry.basic` data envelope must not take the control loop down.

Failing-test-first for a reproduced defect: ``ElizaBridgeBody._read`` stores the
``telemetry.basic`` event's ``data`` verbatim, and ``observe()`` reads it with
``tel.get(...)``. When the bridge sends a malformed envelope whose ``data`` is
not an object (a list, a string, a number, or null), ``observe()`` raises
``AttributeError`` on every frame. The runtime calls ``observe()`` with no
guard, so the episode dies with an unhandled exception instead of degrading
to the same upright defaults the missing-key path already uses. Bad field
values inside a dict are already tolerated (open PR #95); a non-dict container
is not.
"""

import json
import threading

import pytest

pytest.importorskip("websockets")
from websockets.sync.server import serve

from bruhos.body.eliza_bridge import ElizaBridgeBody
from bruhos.brain import make_planner
from bruhos.runtime import Runtime

PROFILE = {"profile": {"actions": {"groups": {"stand": {"duration_s": 0.2}, "wave": {"duration_s": 0.3}}}}}


class NonDictTelemetryBridge:
    """Fake bridge that sends a malformed (non-dict) telemetry.basic payload."""

    def __init__(self, telemetry_data):
        self.telemetry_data = telemetry_data
        self.server = serve(self._handle, "127.0.0.1", 0)
        self.url = f"ws://127.0.0.1:{self.server.socket.getsockname()[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _event(self, ws, event, data):
        ws.send(json.dumps({"type": "event", "event": event, "timestamp": "t",
                            "backend": "fake", "data": data}))

    def _handle(self, ws):
        self._event(ws, "session.hello", {"capabilities": {}})
        self._event(ws, "telemetry.basic", self.telemetry_data)
        self._event(ws, "telemetry.perception", {"entities": []})
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
    return ElizaBridgeBody(cfg)


@pytest.mark.parametrize("bad_data", [["bad"], "nope", 42, None])
def test_observe_survives_non_dict_telemetry_basic(bad_data):
    bridge = NonDictTelemetryBridge(bad_data)
    body = _body(bridge)
    try:
        obs = body.observe()
        assert obs.imu.fallen is False
        assert obs.imu.roll_deg == 0.0 and obs.imu.pitch_deg == 0.0
    finally:
        body.close()
        bridge.close()


def test_runtime_survives_non_dict_telemetry_basic():
    bridge = NonDictTelemetryBridge(["bad"])
    body = _body(bridge)
    try:
        ep = Runtime(body, make_planner("rules")).run("Wave hello.")
        assert ep.success, ep.end_reason
    finally:
        body.close()
        bridge.close()
