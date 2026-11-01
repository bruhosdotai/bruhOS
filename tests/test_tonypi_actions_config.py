"""TonyPi [actions] config: unknown primitive names must not be silently dropped."""

import tomllib
from pathlib import Path

import pytest

from bruhos.body.base import PRIMITIVES
from bruhos.body.tonypi import TonyPiBody

ROOT = Path(__file__).resolve().parents[1]

FAKE_SDK = """
calls = []


def runActionGroup(name, path=None):
    calls.append((name, path))
"""


def _cfg(tmp_path, actions):
    (tmp_path / "ActionGroupControl.py").write_text(FAKE_SDK)
    return {
        "body": {"sdk_path": str(tmp_path), "action_path": "/robot/ActionGroups"},
        "actions": actions,
        "camera": {"enabled": False},
        "imu": {"enabled": False},
    }


def test_unknown_action_key_is_rejected_at_load(tmp_path):
    cfg = _cfg(tmp_path, {"forward": "go_forward", "stand_up_fron": "stand_up_front"})
    with pytest.raises(ValueError, match="stand_up_fron"):
        TonyPiBody(cfg)


def test_valid_action_keys_load_and_dispatch(tmp_path):
    body = TonyPiBody(_cfg(tmp_path, {"forward": "go_forward", "stand_up_front": "stand_up_front"}))
    assert body.actions["stand_up_front"] == "stand_up_front"
    assert body.act("forward") is True


def test_shipped_configs_only_use_known_primitives():
    for name in ("tonypi.toml", "tonypi_pro.toml"):
        with open(ROOT / "configs" / name, "rb") as f:
            actions = tomllib.load(f)["actions"]
        assert set(actions) <= set(PRIMITIVES), name
