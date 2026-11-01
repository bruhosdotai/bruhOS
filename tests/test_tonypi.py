"""TonyPi body: the action-group call convention, against a stubbed Hiwonder SDK."""

import sys
import threading

import pytest

from bruhos.body.tonypi import TonyPiBody

FAKE_SDK = '''\
import threading

calls = []
stand_done = threading.Event()


def runActionGroup(name, path=None):
    calls.append({"group": name, "path": path})
    if name == "stand":
        stand_done.set()


def stopAction():
    calls.append({"group": "__stop__", "path": None})
'''


@pytest.fixture()
def body(tmp_path):
    (tmp_path / "ActionGroupControl.py").write_text(FAKE_SDK)
    made = TonyPiBody({
        "body": {"sdk_path": str(tmp_path), "action_path": str(tmp_path / "ActionGroups")},
        "actions": {"stand": "stand", "wave": "wave"},
        "camera": {"enabled": False},
        "imu": {"enabled": False},
    })
    yield made
    sys.path.remove(str(tmp_path))
    sys.modules.pop("ActionGroupControl", None)


def test_halt_stands_through_the_configured_action_path(body):
    import ActionGroupControl

    assert body.act("wave") is True
    body.halt()
    assert ActionGroupControl.stand_done.wait(2)
    by_group = {c["group"]: c["path"] for c in ActionGroupControl.calls}
    assert by_group["wave"] == body.action_path
    assert by_group["stand"] == body.action_path


def test_halt_stands_when_sdk_has_no_path_parameter(body):
    calls = []
    stand_done = threading.Event()

    class LegacyAGC:
        def stopAction(self):
            calls.append(("stop",))

        def runActionGroup(self, name):          # older SDK: no path keyword
            calls.append((name,))
            if name == "stand":
                stand_done.set()

    body.agc = LegacyAGC()
    body.halt()
    assert stand_done.wait(2)
    assert ("stand",) in calls
