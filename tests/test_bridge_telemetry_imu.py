"""One malformed IMU telemetry field must not take the control loop down.

Failing-test-first for a reproduced defect: ``observe()`` parsed the streamed
``telemetry.basic`` roll/pitch with bare ``float(...)``, so a string or a null
in either field raised inside the runtime control loop, and a NaN passed the
parse silently and poisoned fall detection. The perception entities streamed
from the same bridge events were made tolerant of exactly this data (see the
tolerant ``_num`` parse guarding ``to_detection``); the IMU fields in the same
``observe()`` were not.
"""

import json
import math
import threading
import time

import pytest

pytest.importorskip("websockets")
from websockets.sync.server import serve  # noqa: E402

from bruhos.body.eliza_bridge import ElizaBridgeBody  # noqa: E402
from bruhos.brain import make_planner  # noqa: E402
from bruhos.runtime import Runtime  # noqa: E402

PROFILE = {"profile": {"actions": {"groups": {"wave": {"duration_s": 0.2}}}}}


class FakeBridge:
    """Speaks the eliza_robot command-envelope protocol with an overridable IMU payload."""

    def __init__(self, imu=None):
        self.imu = imu
        self.server = serve(self._handle, "127.0.0.1", 0)
        self.url = f"ws://127.0.0.1:{self.server.socket.getsockname()[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _event(self, ws, event, data):
        ws.send(json.dumps({"type": "event", "event": event, "timestamp": "t", "backend": "fake", "data": data}))

    def _handle(self, ws):
        self._event(ws, "session.hello", {"capabilities": {}})
        self._event(ws, "telemetry.basic",
                    {"imu_roll": 0.0, "imu_pitch": 0.0} if self.imu is None else self.imu)
        self._event(ws, "telemetry.perception", {"entities": []})
        for raw in ws:
            msg = json.loads(raw)
            data = PROFILE if msg["command"] == "profile.describe" else {}
            ws.send(json.dumps({"type": "response", "request_id": msg["request_id"], "timestamp": "t",
                                "ok": True, "backend": "fake", "message": "ok", "data": data}))

    def close(self):
        self.server.shutdown()


def _body(bridge):
    cfg = {"body": {"url": bridge.url, "fall_deg": 28.0},
           "walk": {"forward_s": 0.3, "turn_s": 0.2, "heartbeat_s": 0.05},
           "actions": {"wave": "wave"}}
    return ElizaBridgeBody(cfg)


@pytest.mark.parametrize("imu", [
    {"imu_roll": "level", "imu_pitch": 0.0},
    {"imu_roll": 0.0, "imu_pitch": None},
    {"imu_roll": 0.0, "imu_pitch": float("nan")},
    {"imu_roll": float("inf"), "imu_pitch": 0.0},
])
def test_observe_tolerates_malformed_imu_telemetry(imu):
    bridge = FakeBridge(imu=imu)
    body = _body(bridge)
    time.sleep(0.1)
    obs = body.observe()          # raised ValueError/TypeError or yielded NaN on base
    assert math.isfinite(obs.imu.roll_deg) and math.isfinite(obs.imu.pitch_deg)
    assert not obs.imu.fallen
    body.close()
    bridge.close()


def test_valid_imu_telemetry_still_drives_fall_detection():
    # Guard: numeric telemetry keeps driving fall detection exactly as before.
    bridge = FakeBridge(imu={"imu_roll": 0.0, "imu_pitch": 0.9})
    body = _body(bridge)
    time.sleep(0.1)
    obs = body.observe()
    assert obs.imu.fallen and obs.imu.fell_forward
    body.close()
    bridge.close()


def test_missing_imu_keys_stay_upright():
    # Guard: the missing-key default (upright at 0.0) is unchanged.
    bridge = FakeBridge(imu={})
    body = _body(bridge)
    time.sleep(0.1)
    obs = body.observe()
    assert (obs.imu.roll_deg, obs.imu.pitch_deg) == (0.0, 0.0)
    assert not obs.imu.fallen
    body.close()
    bridge.close()


def test_runtime_survives_malformed_imu_telemetry():
    bridge = FakeBridge(imu={"imu_roll": "level", "imu_pitch": 0.0})
    body = _body(bridge)
    ep = Runtime(body, make_planner("rules")).run("Wave hello.")
    assert ep.success, ep.end_reason
    body.close()
    bridge.close()
