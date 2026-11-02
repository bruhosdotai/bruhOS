"""TonyPi body: a primitive that halt() interrupts must not be reported as done (stubbed Hiwonder SDK)."""

import sys
import threading
import types

import pytest

from bruhos.body.tonypi import TonyPiBody
from bruhos.schemas import SkillCall, Status
from bruhos.skills import SkillContext, run_skill


@pytest.fixture
def sdk(monkeypatch):
    """ActionGroupControl stand-in: a group blocks until it finishes or stopAction() cuts it short."""
    mod = types.ModuleType("ActionGroupControl")
    mod.calls = []
    mod.started = threading.Event()
    mod.release = threading.Event()

    def runActionGroup(name, path=None):
        mod.calls.append(name)
        if name == "stand":
            return
        mod.started.set()
        mod.release.wait(5)

    def stopAction():
        mod.release.set()

    mod.runActionGroup, mod.stopAction = runActionGroup, stopAction
    monkeypatch.setitem(sys.modules, "ActionGroupControl", mod)
    monkeypatch.delitem(sys.modules, "hiwonder", raising=False)
    monkeypatch.delitem(sys.modules, "HiwonderSDK", raising=False)
    return mod


def _body() -> TonyPiBody:
    return TonyPiBody({"body": {}, "camera": {"enabled": False}, "imu": {"enabled": False},
                       "actions": {"stand": "stand", "grab": "move_up", "forward": "go_forward"}})


def _act_halted_midway(body: TonyPiBody, sdk, primitive: str) -> bool:
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("ok", body.act(primitive)))
    t.start()
    assert sdk.started.wait(5), "action group never started"
    body.halt()                                    # e-stop while the servos are mid-group
    t.join(5)
    assert not t.is_alive()
    return result["ok"]


def test_act_reports_failure_when_halt_interrupts_the_group(sdk):
    body = _body()
    assert _act_halted_midway(body, sdk, "forward") is False


def test_interrupted_grab_does_not_claim_to_hold_anything(sdk):
    body = _body()
    assert _act_halted_midway(body, sdk, "grab") is False
    assert body.holding is None
    assert body.observe().holding is None


def test_grab_skill_is_not_ok_after_an_interrupted_grab(sdk):
    body = _body()
    ctx = SkillContext(body)
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("o", run_skill(ctx, SkillCall("grab"))))
    t.start()
    assert sdk.started.wait(5)
    body.halt()
    t.join(5)
    assert out["o"].status is not Status.OK, out["o"]
    assert out["o"].data.get("holding") is None


def test_halt_while_waiting_for_the_lock_does_not_start_the_group(sdk):
    body = _body()
    body._lock.acquire()                           # a previous primitive is still running
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("ok", body.act("forward")))
    t.start()
    t.join(0.2)
    assert t.is_alive()                            # blocked on the lock, past the first halt check
    body.halt()
    body._lock.release()
    t.join(5)
    assert result["ok"] is False
    assert "go_forward" not in sdk.calls


def test_completed_group_after_resume_still_succeeds(sdk):
    body = _body()
    body.halt()
    body.resume()
    sdk.release.set()                              # group runs to completion, no halt
    assert body.act("grab") is True
    assert body.holding == "object"
    assert "move_up" in sdk.calls
