"""Config validation for turn_step_deg: rejected at body load, not in _scan."""

import pytest

from bruhos.body.base import parse_turn_step_deg
from bruhos.body.eliza_bridge import ElizaBridgeBody
from bruhos.body.tonypi import TonyPiBody


@pytest.mark.parametrize("value", [15.0, 0.5, 180.0])
def test_parse_turn_step_deg_accepts(value):
    assert parse_turn_step_deg(value) == value


@pytest.mark.parametrize(
    "value",
    [0, -15, float("nan"), float("inf"), "abc", True, None],
)
def test_parse_turn_step_deg_rejects(value):
    with pytest.raises(ValueError, match=r"turn_step_deg must be a finite angle in \(0, 180\] degrees"):
        parse_turn_step_deg(value)


def test_tonypi_body_rejects_zero_turn_step_deg(monkeypatch):
    monkeypatch.setattr("bruhos.body.tonypi._load_agc", lambda sdk_path: object())
    cfg = {
        "body": {"turn_step_deg": 0},
        "camera": {"enabled": False},
        "imu": {"enabled": False},
    }
    with pytest.raises(ValueError, match=r"turn_step_deg must be a finite angle in \(0, 180\] degrees"):
        TonyPiBody(cfg)


def test_eliza_bridge_body_rejects_zero_turn_step_deg(monkeypatch):
    monkeypatch.setattr(ElizaBridgeBody, "_connect", lambda self: None)
    monkeypatch.setattr(ElizaBridgeBody, "_action_durations", lambda self: {})
    cfg = {"body": {"turn_step_deg": 0}}
    with pytest.raises(ValueError, match=r"turn_step_deg must be a finite angle in \(0, 180\] degrees"):
        ElizaBridgeBody(cfg)
