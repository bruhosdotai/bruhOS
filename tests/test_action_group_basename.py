"""Action-group names are file stems, not servo-script paths.

Hiwonder ``runAction`` opens ``path + name + ".d6a"`` and writes every row with
``set_bus_servo_pulse``. A name that contains a path separator escapes the
action-group directory, so a whitelisted skill such as ``wave`` plays whatever
servo script that file holds. The body may only dispatch whitelist primitives,
and only to a single basename.
"""

import sys
import tomllib
import types
from pathlib import Path

import pytest

from bruhos.body.tonypi import TonyPiBody
from bruhos.schemas import SkillCall, Status
from bruhos.skills import SkillContext, run_skill

ROOT = Path(__file__).resolve().parents[1]


class _AGC:
    def __init__(self):
        self.runs: list[tuple[object, object]] = []

    def runActionGroup(self, name, times=1, with_stand=False, lock_servos="", path=None):
        self.runs.append((name, path))

    def stopAction(self):
        pass


class _ImmediateThread:
    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


def _install_agc(monkeypatch):
    agc = _AGC()
    mod = types.ModuleType("hiwonder.ActionGroupControl")
    mod.runActionGroup = agc.runActionGroup
    mod.stopAction = agc.stopAction
    pkg = types.ModuleType("hiwonder")
    pkg.ActionGroupControl = mod
    monkeypatch.setitem(sys.modules, "hiwonder", pkg)
    monkeypatch.setitem(sys.modules, "hiwonder.ActionGroupControl", mod)
    return agc


def _cfg(**actions):
    return {
        "body": {"action_path": "/robot/ActionGroups"},
        "actions": actions,
        "camera": {"enabled": False},
        "imu": {"enabled": False},
    }


@pytest.mark.parametrize("name", [
    "../pulses",
    "foo/../../../tmp/pulses",
    "/tmp/pulses",
    "..",
    ".",
    "foo\\..\\pulses",
    "wave\x00../pulses",
    7,
])
def test_path_action_group_is_rejected_before_the_sdk_loads(name):
    with pytest.raises(ValueError, match="basename"):
        TonyPiBody(_cfg(wave=name))


def test_safe_basename_still_plays(monkeypatch):
    agc = _install_agc(monkeypatch)
    body = TonyPiBody(_cfg(wave="wave", stand="stand"))
    assert body.act("wave") is True
    assert agc.runs == [("wave", "/robot/ActionGroups")]


def test_unmapped_primitive_does_not_guess_a_file(monkeypatch):
    agc = _install_agc(monkeypatch)
    body = TonyPiBody(_cfg(wave="wave"))
    assert body.act("grab") is False
    assert agc.runs == []


def test_servo_name_outside_the_primitive_whitelist_is_not_played(monkeypatch):
    agc = _install_agc(monkeypatch)
    body = TonyPiBody(_cfg(**{"servo.set": "pulses", "wave": "wave"}))
    assert body.act("servo.set") is False
    assert agc.runs == []


def test_wave_skill_does_not_play_a_traversing_action_group(monkeypatch):
    agc = _install_agc(monkeypatch)
    body = TonyPiBody(_cfg(wave="wave"))
    body.actions["wave"] = "foo/../../../tmp/pulses"
    run_skill(SkillContext(body), SkillCall("wave"))
    assert agc.runs == []


def test_halt_does_not_replay_a_traversing_stand_group(monkeypatch):
    import bruhos.body.tonypi as tonypi

    agc = _install_agc(monkeypatch)
    monkeypatch.setattr(tonypi.threading, "Thread", _ImmediateThread)
    body = TonyPiBody(_cfg(stand="stand", wave="wave"))
    body.actions["stand"] = "../pulses"
    body.halt()
    assert agc.runs == []


def test_eliza_rejects_a_path_action_before_connecting(monkeypatch):
    from bruhos.body.eliza_bridge import ElizaBridgeBody

    def explode(self):
        raise AssertionError("bridge connect")

    monkeypatch.setattr(ElizaBridgeBody, "_connect", explode)
    with pytest.raises(ValueError, match="basename"):
        ElizaBridgeBody({"body": {"url": "ws://127.0.0.1:9"}, "actions": {"wave": "../pulses"}})


def _eliza(monkeypatch, **actions):
    from bruhos.body.eliza_bridge import ElizaBridgeBody

    monkeypatch.setattr(ElizaBridgeBody, "_connect", lambda self: None)
    monkeypatch.setattr(ElizaBridgeBody, "_action_durations", lambda self: {})
    body = ElizaBridgeBody({"actions": actions})
    sent: list[tuple] = []
    monkeypatch.setattr(body, "_send", lambda *args, **kwargs: sent.append(args) or {"ok": True})
    return body, sent


def test_eliza_does_not_play_a_path_action(monkeypatch):
    body, sent = _eliza(monkeypatch, wave="wave")
    body.actions["wave"] = "../pulses"
    assert body.act("wave") is False
    assert sent == []


def test_eliza_does_not_send_a_non_primitive(monkeypatch):
    body, sent = _eliza(monkeypatch, **{"wave": "wave", "servo.set": "pulses"})
    assert body.act("servo.set") is False
    assert sent == []


def test_shipped_action_groups_are_basenames():
    for cfg in sorted((ROOT / "configs").glob("*.toml")):
        actions = tomllib.loads(cfg.read_text()).get("actions", {})
        for key, value in actions.items():
            assert isinstance(value, str) and value not in {".", ".."}
            assert "/" not in value and "\\" not in value and "\x00" not in value, (cfg.name, key, value)


def test_wave_outcome_is_unchanged_for_a_real_basename(monkeypatch):
    """A normal wave still reaches the body. Path rejection must not fail safe names."""
    _install_agc(monkeypatch)
    body = TonyPiBody(_cfg(wave="wave"))
    out = run_skill(SkillContext(body), SkillCall("wave"))
    assert out.status is Status.OK
