"""forward_sign is a mount-direction flip: only 1 and -1 describe a mounted IMU.

A 0 zeroes the pitch, so a robot lying face-down reads pitch 0.0 and
``fallen=False`` — the fall is never detected and recovery never runs. Any
other value rescales the pitch (2 doubles it: a 30-degree tilt reads as a
60-degree fall). Both fail silently at run time. The value must be rejected
at load, before the I2C bus opens (MPU6050) or the bridge connects.
"""

import json
import math
import sys
import threading
import types

import pytest

pytest.importorskip("websockets")
from websockets.sync.server import serve  # noqa: E402

from bruhos.body.base import parse_forward_sign  # noqa: E402
from bruhos.body.imu import ACCEL_SCALE, ACCEL_XOUT_H, MPU6050  # noqa: E402
from bruhos.body.eliza_bridge import ElizaBridgeBody  # noqa: E402


@pytest.mark.parametrize("value,expected", [
    (1, 1), (-1, -1), (1.0, 1), (-1.0, -1), ("1", 1), ("-1", -1),
    ("1.0", 1), ("-1.0", -1),
])
def test_accepts_only_plus_or_minus_one(value, expected):
    got = parse_forward_sign(value)
    assert got == expected and type(got) is int


@pytest.mark.parametrize("value", [
    0, 2, -2, 0.5, 1.5, -0.5, 10, float("nan"), float("inf"), float("-inf"),
    True, False, None, "0", "2", "abc", "", "1.5", [1], {"sign": 1},
])
def test_rejects_anything_but_plus_or_minus_one(value):
    with pytest.raises(ValueError, match="forward_sign"):
        parse_forward_sign(value)


class _FakeSMBus:
    """Minimal smbus2 stand-in: serves one accelerometer sample."""

    def __init__(self, bus, sample=(0.0, 0.0, 1.0)):
        self.bus = bus
        self.sample = sample

    def write_byte_data(self, addr, reg, val):
        pass

    def read_i2c_block_data(self, addr, reg, n):
        i = (reg - ACCEL_XOUT_H) // 2
        raw = int(self.sample[i] * ACCEL_SCALE) & 0xFFFF
        return [(raw >> 8) & 0xFF, raw & 0xFF]

    def close(self):
        pass


@pytest.fixture
def fake_smbus(monkeypatch):
    mod = types.ModuleType("smbus2")
    mod.SMBus = _FakeSMBus
    monkeypatch.setitem(sys.modules, "smbus2", mod)
    return mod


@pytest.mark.parametrize("value", [0, 2, -2, 0.5, float("nan"), True, "abc"])
def test_mpu6050_rejects_bad_forward_sign_at_construction(fake_smbus, value):
    with pytest.raises(ValueError, match="forward_sign"):
        MPU6050(bus=1, address=0x68, forward_sign=value)


def test_mpu6050_face_down_fall_is_detected_with_valid_sign(fake_smbus):
    # Face-down: ax = -1g reads as pitch +90 in the sensor frame.
    imu = MPU6050(bus=1, address=0x68, forward_sign=1)
    imu.bus.sample = (-1.0, 0.0, 0.0)
    reading = imu.read()
    assert reading.pitch_deg == 90.0
    assert reading.fallen and reading.fell_forward


def test_mpu6050_reversed_mount_still_detects_the_fall(fake_smbus):
    imu = MPU6050(bus=1, address=0x68, forward_sign=-1)
    imu.bus.sample = (-1.0, 0.0, 0.0)
    reading = imu.read()
    assert reading.pitch_deg == -90.0
    assert reading.fallen and not reading.fell_forward


class _FakeBridge:
    """Speaks the eliza_robot command-envelope protocol on a local port."""

    def __init__(self):
        self.server = serve(self._handle, "127.0.0.1", 0)
        self.url = f"ws://127.0.0.1:{self.server.socket.getsockname()[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _handle(self, ws):
        for raw in ws:
            msg = json.loads(raw)
            ws.send(json.dumps({"type": "response", "request_id": msg["request_id"],
                                "timestamp": "t", "ok": True, "backend": "fake",
                                "message": "ok", "data": {}}))

    def close(self):
        self.server.shutdown()


@pytest.fixture
def bridge():
    b = _FakeBridge()
    yield b
    b.close()


@pytest.mark.parametrize("value", [0, 2, -2, 0.5, float("nan"), True, "abc"])
def test_bridge_rejects_bad_forward_sign_before_connecting(value):
    # No server listens on port 1: a ValueError proves the check runs
    # before _connect(), a connection error would prove it does not.
    with pytest.raises(ValueError, match="forward_sign"):
        ElizaBridgeBody({"body": {"url": "ws://127.0.0.1:1", "forward_sign": value}})


def test_bridge_accepts_valid_forward_sign(bridge):
    body = ElizaBridgeBody({"body": {"url": bridge.url, "forward_sign": -1}})
    assert body.forward_sign == -1
    body.close()
