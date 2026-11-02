"""Tests for fall direction detection on MPU6050 and Eliza bridge IMUs.

A robot falling sideways (roll exceeding fall_deg while pitch remains within
fall_deg) must report fallen=True and fell_forward=None. It must not claim
fell_forward=True or fell_forward=False simply because residual pitch happens
to be slightly positive or negative.
"""

from __future__ import annotations

import math
import sys
from unittest.mock import MagicMock

import pytest

from bruhos.body.imu import ACCEL_SCALE, MPU6050


def _mock_smbus(ax_val: int, ay_val: int, az_val: int):
    """Create a mock SMBus module returning raw 16-bit accelerometer words."""
    mock_module = MagicMock()
    bus_instance = MagicMock()
    mock_module.SMBus.return_value = bus_instance

    def fake_word(addr, reg, n):
        if reg == 0x3B:
            v = ax_val & 0xFFFF
            return [v >> 8, v & 0xFF]
        elif reg == 0x3D:
            v = ay_val & 0xFFFF
            return [v >> 8, v & 0xFF]
        elif reg == 0x3F:
            v = az_val & 0xFFFF
            return [v >> 8, v & 0xFF]
        return [0, 0]

    bus_instance.read_i2c_block_data.side_effect = fake_word
    return mock_module, bus_instance


def _accel_for_angles(roll_deg: float, pitch_deg: float) -> tuple[int, int, int]:
    """Derive approximate accelerometer values (LSB) for given roll and pitch."""
    # roll = atan2(ay, az), pitch = atan2(-ax, hypot(ay, az))
    r_rad = math.radians(roll_deg)
    p_rad = math.radians(pitch_deg)
    # With 1g total magnitude:
    # -ax = sin(pitch) -> ax = -sin(pitch)
    # hypot(ay, az) = cos(pitch)
    # ay = cos(pitch) * sin(roll)
    # az = cos(pitch) * cos(roll)
    ax = -math.sin(p_rad)
    ay = math.cos(p_rad) * math.sin(r_rad)
    az = math.cos(p_rad) * math.cos(r_rad)
    return int(ax * ACCEL_SCALE), int(ay * ACCEL_SCALE), int(az * ACCEL_SCALE)


def test_mpu6050_roll_fall_does_not_claim_pitch_direction(monkeypatch):
    # Robot rolls 70 deg right, residual pitch is slightly positive (+0.5 deg)
    ax, ay, az = _accel_for_angles(roll_deg=70.0, pitch_deg=0.5)
    mock_mod, _ = _mock_smbus(ax, ay, az)
    monkeypatch.setitem(sys.modules, "smbus2", mock_mod)

    sensor = MPU6050(fall_deg=55.0)
    reading = sensor.read()

    assert reading.fallen is True
    # Roll caused the fall, pitch did not exceed threshold: direction must be None
    assert reading.fell_forward is None


def test_mpu6050_roll_fall_with_negative_pitch_noise(monkeypatch):
    # Robot rolls 70 deg left, residual pitch is slightly negative (-0.5 deg)
    ax, ay, az = _accel_for_angles(roll_deg=-70.0, pitch_deg=-0.5)
    mock_mod, _ = _mock_smbus(ax, ay, az)
    monkeypatch.setitem(sys.modules, "smbus2", mock_mod)

    sensor = MPU6050(fall_deg=55.0)
    reading = sensor.read()

    assert reading.fallen is True
    assert reading.fell_forward is None


def test_mpu6050_forward_fall(monkeypatch):
    ax, ay, az = _accel_for_angles(roll_deg=0.0, pitch_deg=70.0)
    mock_mod, _ = _mock_smbus(ax, ay, az)
    monkeypatch.setitem(sys.modules, "smbus2", mock_mod)

    sensor = MPU6050(fall_deg=55.0)
    reading = sensor.read()

    assert reading.fallen is True
    assert reading.fell_forward is True


def test_mpu6050_backward_fall(monkeypatch):
    ax, ay, az = _accel_for_angles(roll_deg=0.0, pitch_deg=-70.0)
    mock_mod, _ = _mock_smbus(ax, ay, az)
    monkeypatch.setitem(sys.modules, "smbus2", mock_mod)

    sensor = MPU6050(fall_deg=55.0)
    reading = sensor.read()

    assert reading.fallen is True
    assert reading.fell_forward is False


def test_mpu6050_upright(monkeypatch):
    ax, ay, az = _accel_for_angles(roll_deg=5.0, pitch_deg=5.0)
    mock_mod, _ = _mock_smbus(ax, ay, az)
    monkeypatch.setitem(sys.modules, "smbus2", mock_mod)

    sensor = MPU6050(fall_deg=55.0)
    reading = sensor.read()

    assert reading.fallen is False
    assert reading.fell_forward is None


# Eliza bridge tests (conditional on websockets)
def test_eliza_bridge_roll_fall_does_not_claim_pitch_direction():
    pytest.importorskip("websockets")
    import json
    import threading
    import time
    from websockets.sync.server import serve
    from bruhos.body.eliza_bridge import ElizaBridgeBody

    class BridgeServer:
        def __init__(self):
            self.server = serve(self._handle, "127.0.0.1", 0)
            self.url = f"ws://127.0.0.1:{self.server.socket.getsockname()[1]}"
            threading.Thread(target=self.server.serve_forever, daemon=True).start()

        def _handle(self, ws):
            ws.send(json.dumps({"type": "event", "event": "session.hello", "timestamp": "t", "backend": "fake", "data": {}}))
            # roll = 1.0 rad (~57.3 deg > 28 deg), pitch = 0.005 rad (~0.29 deg < 28 deg)
            ws.send(json.dumps({"type": "event", "event": "telemetry.basic", "timestamp": "t", "backend": "fake",
                                "data": {"imu_roll": 1.0, "imu_pitch": 0.005}}))
            ws.send(json.dumps({"type": "event", "event": "telemetry.perception", "timestamp": "t", "backend": "fake", "data": {"entities": []}}))
            for raw in ws:
                msg = json.loads(raw)
                ws.send(json.dumps({"type": "response", "request_id": msg["request_id"], "timestamp": "t",
                                    "ok": True, "backend": "fake", "message": "ok", "data": {}}))

        def close(self):
            self.server.shutdown()

    server = BridgeServer()
    body = ElizaBridgeBody({"body": {"url": server.url, "fall_deg": 28.0}})
    time.sleep(0.1)
    obs = body.observe()

    assert obs.imu.fallen is True
    # Roll caused the fall, pitch is well below fall_deg -> fell_forward must be None
    assert obs.imu.fell_forward is None

    body.close()
    server.close()


def test_eliza_bridge_pitch_falls():
    pytest.importorskip("websockets")
    import json
    import threading
    import time
    from websockets.sync.server import serve
    from bruhos.body.eliza_bridge import ElizaBridgeBody

    class BridgeServer:
        def __init__(self, pitch_rad):
            self.pitch_rad = pitch_rad
            self.server = serve(self._handle, "127.0.0.1", 0)
            self.url = f"ws://127.0.0.1:{self.server.socket.getsockname()[1]}"
            threading.Thread(target=self.server.serve_forever, daemon=True).start()

        def _handle(self, ws):
            ws.send(json.dumps({"type": "event", "event": "session.hello", "timestamp": "t", "backend": "fake", "data": {}}))
            ws.send(json.dumps({"type": "event", "event": "telemetry.basic", "timestamp": "t", "backend": "fake",
                                "data": {"imu_roll": 0.0, "imu_pitch": self.pitch_rad}}))
            ws.send(json.dumps({"type": "event", "event": "telemetry.perception", "timestamp": "t", "backend": "fake", "data": {"entities": []}}))
            for raw in ws:
                msg = json.loads(raw)
                ws.send(json.dumps({"type": "response", "request_id": msg["request_id"], "timestamp": "t",
                                    "ok": True, "backend": "fake", "message": "ok", "data": {}}))

        def close(self):
            self.server.shutdown()

    # Forward fall (+0.9 rad ~ +51.6 deg > 28 deg)
    s1 = BridgeServer(0.9)
    b1 = ElizaBridgeBody({"body": {"url": s1.url, "fall_deg": 28.0}})
    time.sleep(0.1)
    obs1 = b1.observe()
    assert obs1.imu.fallen is True
    assert obs1.imu.fell_forward is True
    b1.close()
    s1.close()

    # Backward fall (-0.9 rad ~ -51.6 deg < -28 deg)
    s2 = BridgeServer(-0.9)
    b2 = ElizaBridgeBody({"body": {"url": s2.url, "fall_deg": 28.0}})
    time.sleep(0.1)
    obs2 = b2.observe()
    assert obs2.imu.fallen is True
    assert obs2.imu.fell_forward is False
    b2.close()
    s2.close()
