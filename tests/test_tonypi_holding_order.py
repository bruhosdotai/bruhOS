"""TonyPi holding state must follow the order in which action groups ran.

``act`` serializes ``runActionGroup`` under ``_lock`` but used to publish ``holding`` after the
lock was released. A ``grab`` whose caller was descheduled at that point could then overwrite the
``None`` of a ``release`` that ran after it, so ``observe()`` reported an object in hand after the
hand had opened (and the reverse for ``release`` then ``grab``). Same defect as the Eliza bridge
adapter (#129), on the TonyPi body.
"""

from __future__ import annotations

import sys
import threading
import types

import pytest

from bruhos.body.tonypi import TonyPiBody

CFG = {"body": {}, "camera": {"enabled": False}, "imu": {"enabled": False},
       "actions": {"grab": "grab", "release": "release"}}


class PausingUnlock:
    """Expose a scheduling point right after the real action lock is released."""

    def __init__(self, lock, pause):
        self.lock, self.pause = lock, pause
        self.released, self.proceed = threading.Event(), threading.Event()

    def __enter__(self):
        self.lock.acquire()

    def __exit__(self, *exc):
        self.lock.release()
        if threading.current_thread().name == "first-tonypi-action":
            self.released.set()
            if self.pause:
                assert self.proceed.wait(10), "test did not release the first caller"


@pytest.fixture
def sdk(monkeypatch):
    mod = types.ModuleType("ActionGroupControl")
    mod.played = []
    mod.runActionGroup = lambda name: mod.played.append(name)
    monkeypatch.setitem(sys.modules, "ActionGroupControl", mod)
    return mod


@pytest.mark.parametrize("pause", [False, True], ids=["sequential", "interleaved"])
@pytest.mark.parametrize("first,second,expected",
                         [("grab", "release", None), ("release", "grab", "object")])
def test_holding_follows_serialized_action_order(sdk, first, second, expected, pause):
    body = TonyPiBody(CFG)
    body.holding = "object" if first == "release" else None
    gate = body._lock = PausingUnlock(body._lock, pause)
    results, errors = [], []

    def act_first():
        try:
            results.append(body.act(first))
        except Exception as exc:  # pragma: no cover - surfaced by the assert below
            errors.append(exc)

    worker = threading.Thread(target=act_first, name="first-tonypi-action", daemon=True)
    try:
        worker.start()
        assert gate.released.wait(10)
        if not pause:
            worker.join(10)
            assert not worker.is_alive()
        assert body.act(second)
        gate.proceed.set()
        worker.join(10)
        assert not worker.is_alive() and not errors and results == [True]
        assert sdk.played == [first, second]
        assert body.observe().holding == expected
    finally:
        gate.proceed.set()
        worker.join(10)
