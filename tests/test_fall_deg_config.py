"""Fall-detection threshold must be validated at load time.

``fall_deg`` is the tilt that counts as a fall. On ``main`` it is accepted
without any check in ``MPU6050`` and in ``ElizaBridgeBody.observe``'s
comparison, so a bad value silently breaks fall recovery either way:

- NaN / inf: ``abs(roll) > fall_deg`` is always False, so a robot rolled 70
  degrees reads ``fallen=False`` and ``stand_up`` never runs.
- 0 / negative: any micro-tilt counts as a fall, so an upright robot reads
  ``fallen=True`` and burns the recovery budget standing up from nothing.

Both fail silently at run time; the config must be rejected at construction,
the same way ``parse_turn_step`` (open PR #120) rejects an unusable turn step.
"""

from __future__ import annotations

import math
import sys
import types
from unittest.mock import MagicMock

import pytest

from bruhos.body.base import parse_fall_deg
from bruhos.body.imu import ACCEL_SCALE, MPU6050

BAD = [0, 0.0, -10.0, 180.0, 400.0, math.nan, math.inf, "fast", True]
GOOD = [1.0, 28.0, 55.0, "55.0", 179.9]


@pytest.mark.parametrize("value", BAD)
def test_parse_fall_deg_rejects(value):
    with pytest.raises(ValueError, match="fall_deg"):
        parse_fall_deg(value)


@pytest.mark.parametrize("value", GOOD)
def test_parse_fall_deg_accepts(value):
    assert parse_fall_deg(value) == float(value)


def _mock_smbus(ax_val: int, ay_val: int, az_val: int):
    mock_module = MagicMock()
    bus_instance = MagicMock()
    mock_module.SMBus.return_value = bus_instance

    def fake_word(addr, reg, n):
        v = {0x3B: ax_val, 0x3D: ay_val, 0x3F: az_val}[reg] & 0xFFFF
        return [v >> 8, v & 0xFF]

    bus_instance.read_i2c_block_data.side_effect = fake_word
    return mock_module


def _accel_for_angles(roll_deg: float, pitch_deg: float) -> tuple[int, int, int]:
    r_rad, p_rad = math.radians(roll_deg), math.radians(pitch_deg)
    ax = -math.sin(p_rad)
    ay = math.cos(p_rad) * math.sin(r_rad)
    az = math.cos(p_rad) * math.cos(r_rad)
    return int(ax * ACCEL_SCALE), int(ay * ACCEL_SCALE), int(az * ACCEL_SCALE)


@pytest.mark.parametrize("value", BAD)
def test_mpu6050_rejects_bad_fall_deg_before_touching_the_bus(monkeypatch, value):
    ax, ay, az = _accel_for_angles(70.0, 0.0)
    mock_mod = _mock_smbus(ax, ay, az)
    mock_mod.SMBus.side_effect = AssertionError(
        "bus must not be opened for a bad fall_deg"
    )
    monkeypatch.setitem(sys.modules, "smbus2", mock_mod)
    with pytest.raises(ValueError, match="fall_deg"):
        MPU6050(fall_deg=value)


def test_mpu6050_accepts_good_fall_deg_and_still_detects_falls(monkeypatch):
    ax, ay, az = _accel_for_angles(70.0, 0.0)
    monkeypatch.setitem(sys.modules, "smbus2", _mock_smbus(ax, ay, az))
    sensor = MPU6050(fall_deg=55.0)
    reading = sensor.read()
    assert reading.fallen is True

    ax, ay, az = _accel_for_angles(0.5, 0.0)
    monkeypatch.setitem(sys.modules, "smbus2", _mock_smbus(ax, ay, az))
    sensor = MPU6050(fall_deg=55.0)
    assert sensor.read().fallen is False


@pytest.mark.parametrize("value", [0, -10.0, math.nan, math.inf, 180.0])
def test_eliza_bridge_rejects_bad_fall_deg_before_connecting(value):
    from bruhos.body.eliza_bridge import ElizaBridgeBody

    cfg = {"body": {"fall_deg": value, "url": "ws://127.0.0.1:9", "timeout_s": 0.2}}
    with pytest.raises(ValueError, match="fall_deg"):
        ElizaBridgeBody(cfg)


@pytest.mark.parametrize("value", [0, math.nan, math.inf])
def test_tonypi_rejects_bad_imu_fall_deg(monkeypatch, value):
    from bruhos.body.tonypi import TonyPiBody

    sdk = types.ModuleType("ActionGroupControl")
    sdk.runActionGroup = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "ActionGroupControl", sdk)
    ax, ay, az = _accel_for_angles(70.0, 0.0)
    monkeypatch.setitem(sys.modules, "smbus2", _mock_smbus(ax, ay, az))
    cfg = {
        "body": {"action_path": "/tmp"},
        "camera": {"enabled": False},
        "imu": {"enabled": True, "fall_deg": value},
        "actions": {},
    }
    with pytest.raises(ValueError, match="fall_deg"):
        TonyPiBody(cfg)
