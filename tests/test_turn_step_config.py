"""Bodies must reject a ``turn_step_deg`` that the skills cannot steer with, at load time.

``_scan`` computes ``int(360 / turn_step_deg)`` and ``look_at`` / ``walk_toward`` turn one step at a
time. On ``main`` a config value of 0 raised ZeroDivisionError and NaN raised ValueError mid-skill,
while a negative or > 360 value made ``detect`` report "not found after full scan" without turning.
"""

from __future__ import annotations

import math
import sys
import types

import pytest

from bruhos.body.eliza_bridge import ElizaBridgeBody
from bruhos.body.tonypi import TonyPiBody

BAD = [0, 0.0, -15.0, 181.0, 400.0, math.nan, math.inf, "fast", True]
GOOD = [1.0, 15, "15.0", 30.0, 180.0]


@pytest.mark.parametrize("value", BAD)
def test_parse_turn_step_rejects(value):
    from bruhos.body.base import parse_turn_step

    with pytest.raises(ValueError, match="turn_step_deg"):
        parse_turn_step(value)


@pytest.mark.parametrize("value", GOOD)
def test_parse_turn_step_accepts(value):
    from bruhos.body.base import parse_turn_step

    assert parse_turn_step(value) == float(value)


@pytest.mark.parametrize("value", [0, -15.0, math.nan])
def test_tonypi_rejects_before_loading_the_sdk(monkeypatch, value):
    sdk = types.ModuleType("ActionGroupControl")
    sdk.runActionGroup = lambda *a, **k: pytest.fail("SDK must not be touched")
    monkeypatch.setitem(sys.modules, "ActionGroupControl", sdk)
    cfg = {"body": {"turn_step_deg": value}, "camera": {"enabled": False}, "imu": {"enabled": False}}
    with pytest.raises(ValueError, match="turn_step_deg"):
        TonyPiBody(cfg)


@pytest.mark.parametrize("value", [0, 400.0, math.inf])
def test_eliza_bridge_rejects_before_connecting(value):
    # The URL points nowhere: on main this reached _connect() and failed there (or on the import).
    cfg = {"body": {"turn_step_deg": value, "url": "ws://127.0.0.1:9", "timeout_s": 0.2}}
    with pytest.raises(ValueError, match="turn_step_deg"):
        ElizaBridgeBody(cfg)
