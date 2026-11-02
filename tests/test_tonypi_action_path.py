"""TonyPiBody must run an action group at most once per primitive.

``act`` used to call ``runActionGroup(group, path=...)`` and, on *any* ``TypeError``, call it again
without ``path`` to support SDKs that lack the keyword. A ``TypeError`` raised from inside the SDK
after the servos had started moving therefore replayed the whole action group on the robot.
"""

from __future__ import annotations

import sys
import types

import pytest

from bruhos.body.tonypi import TonyPiBody

CFG = {"body": {"action_path": "/home/pi/ActionGroups"}, "camera": {"enabled": False},
       "imu": {"enabled": False}, "actions": {"wave": "wave"}}


@pytest.fixture
def sdk(monkeypatch):
    mod = types.ModuleType("ActionGroupControl")
    mod.calls = []
    monkeypatch.setitem(sys.modules, "ActionGroupControl", mod)
    return mod


def test_sdk_type_error_mid_action_does_not_replay_the_group(sdk):
    def runActionGroup(name, path=None):
        sdk.calls.append((name, path))
        raise TypeError("unsupported operand type(s) for -: 'NoneType' and 'int'")   # bad frame in the .d6a

    sdk.runActionGroup = runActionGroup
    body = TonyPiBody(CFG)
    with pytest.raises(TypeError):
        body.act("wave")
    assert sdk.calls == [("wave", "/home/pi/ActionGroups")]          # once, not twice


def test_sdk_without_path_keyword_is_called_once_without_it(sdk):
    def runActionGroup(name):
        sdk.calls.append((name, None))

    sdk.runActionGroup = runActionGroup
    assert TonyPiBody(CFG).act("wave") is True
    assert sdk.calls == [("wave", None)]


def test_sdk_with_path_keyword_gets_the_configured_path(sdk):
    def runActionGroup(name, path=None):
        sdk.calls.append((name, path))

    sdk.runActionGroup = runActionGroup
    assert TonyPiBody(CFG).act("wave") is True
    assert sdk.calls == [("wave", "/home/pi/ActionGroups")]
