"""Torque packets match Hiwonder's unload, and limb groups stay apart."""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SDK = Path("/home/pi/TonyPi/HiwonderSDK")


def load_limbs():
    path = ROOT / "scripts" / "limbs.py"
    spec = importlib.util.spec_from_file_location("limbs_script", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


limbs = load_limbs()


def test_each_hand_is_its_own_group():
    assert "hands" not in limbs.GROUPS
    assert limbs.GROUPS["left-hand"] == (8,)
    assert limbs.GROUPS["right-hand"] == (16,)
    assert set(limbs.GROUPS["left-arm"]) == {6, 7, 8}
    assert set(limbs.GROUPS["right-arm"]) == {14, 15, 16}


def test_legs_do_not_include_arms_or_head():
    legs = set(limbs.GROUPS["legs"])
    assert legs == {1, 2, 3, 4, 5, 9, 10, 11, 12, 13}
    assert legs.isdisjoint(limbs.GROUPS["arms"])
    assert 17 not in legs and 18 not in legs


def test_ids_for_keeps_each_hand_and_dedupes_into_the_arm():
    assert limbs.ids_for(["left-hand", "right-hand"]) == [8, 16]
    assert limbs.ids_for(["left-hand", "left-arm"]) == [8, 6, 7]


def test_unload_packet_matches_sdk_buf_write():
    if not (SDK / "hiwonder" / "ros_robot_controller_sdk.py").is_file():
        pytest.skip("Hiwonder SDK not on this machine")
    sys.path.insert(0, str(SDK))
    import hiwonder.ros_robot_controller_sdk as rrc

    class Sink:
        def write(self, data):
            self.buf = bytes(data)

    board = rrc.Board.__new__(rrc.Board)
    board.port = Sink()
    board.buf_write(rrc.PacketFunction.PACKET_FUNC_BUS_SERVO, struct.pack("<BB", 0x0B, 8))
    assert limbs.packet(limbs.UNLOAD, 8) == board.port.buf
    board.buf_write(rrc.PacketFunction.PACKET_FUNC_BUS_SERVO, struct.pack("<BB", 0x0C, 16))
    assert limbs.packet(limbs.LOAD, 16) == board.port.buf
    assert limbs.packet(limbs.UNLOAD, 8) != limbs.packet(limbs.LOAD, 8)
