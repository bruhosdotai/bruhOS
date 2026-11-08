"""TonyPi act() must fail the primitive, not crash the episode, when the servo send fails.

``Body.act`` promises: "Returns False if the body could not run it." A dropped servo
bus raises inside ``runActionGroup``; today that exception escapes through
``run_skill`` and kills the whole ``Runtime.run`` instead of producing a FAILED
outcome the runtime can replan from. (The eliza bridge got the same treatment
in #227; TonyPi never did.)
"""

import sys
import types

import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.body.tonypi import TonyPiBody
from bruhos.runtime import Runtime


class _FailingAgc:
    """ActionGroupControl whose servo bus is down: every send raises."""

    def __init__(self):
        self.calls: list[str] = []

    def runActionGroup(self, group, **kw):
        self.calls.append(group)
        raise RuntimeError("servo bus dropped")

    def stopAction(self):
        pass


@pytest.fixture
def failing_agc(monkeypatch):
    agc = _FailingAgc()
    mod = types.ModuleType("hiwonder.ActionGroupControl")
    mod.runActionGroup = agc.runActionGroup
    mod.stopAction = agc.stopAction
    monkeypatch.setitem(sys.modules, "hiwonder.ActionGroupControl", mod)
    return agc


def _body():
    return TonyPiBody({
        "body": {"name": "tonypi"},
        "camera": {"enabled": False},
        "imu": {"enabled": False},
        "actions": {"wave": "wave", "grab": "grab", "stand": "stand", "turn_left": "turn_left"},
    })


def test_act_returns_false_when_action_group_raises(failing_agc):
    body = _body()
    assert body.act("wave") is False
    assert failing_agc.calls == ["wave"]


def test_act_does_not_report_a_grab_when_the_send_fails(failing_agc):
    body = _body()
    assert body.act("grab") is False
    assert body.holding is None


def test_runtime_survives_a_failed_action_group(failing_agc):
    ep = Runtime(_body(), RulesPlanner()).run("Grab the blue one.")
    assert ep.success is False
    assert "failed" in ep.end_reason
