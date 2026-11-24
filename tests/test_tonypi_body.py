"""TonyPi body regression tests (no hardware: the SDK is stubbed)."""

import threading
from unittest.mock import patch

from bruhos.body.tonypi import TonyPiBody


class FakeAGC:
    def __init__(self):
        self.calls = []
        self.done = threading.Event()

    def runActionGroup(self, group, **kw):
        self.calls.append((group, kw))
        self.done.set()

    def stopAction(self):
        self.calls.append(("stop", {}))


def _body(agc, action_path):
    cfg = {
        "body": {"action_path": action_path},
        "actions": {"stand": "stand_group", "wave": "wave_group"},
        "camera": {"enabled": False},
        "imu": {"enabled": False},
    }
    with patch("bruhos.body.tonypi._load_agc", return_value=agc):
        return TonyPiBody(cfg)


def test_halt_runs_stand_through_configured_action_path():
    """halt() must use the same action group path as act(), or the stand pose never plays."""
    agc = FakeAGC()
    body = _body(agc, "/opt/action-groups")
    body.halt()
    assert agc.done.wait(2)
    assert ("stand_group", {"path": "/opt/action-groups"}) in agc.calls
