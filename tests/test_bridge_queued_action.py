"""Cancellation while waiting for another action must be checked before dispatch."""
import threading

import pytest

pytest.importorskip("websockets")
from test_eliza_bridge import FakeBridge, _body  # noqa: E402


class WaitingLock:
    """Expose when a worker reaches the real action lock, without timing sleeps."""

    def __init__(self, lock):
        self.lock = lock
        self.waiting = threading.Event()

    def __enter__(self):
        self.waiting.set()
        self.lock.acquire()

    def __exit__(self, *exc):
        self.lock.release()


@pytest.mark.parametrize("primitive", ["wave", "forward"])
@pytest.mark.parametrize("cancel", ["halt", "close"])
def test_queued_action_rechecks_cancellation_before_sending(primitive, cancel):
    bridge = FakeBridge()
    body = _body(bridge)
    lock = body._lock
    waiting = body._lock = WaitingLock(lock)
    results, errors = [], []

    def act():
        try:
            results.append(body.act(primitive))
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=act, daemon=True)
    try:
        # Hold the same lock an in-flight action owns, then queue the next action.
        with lock:
            worker.start()
            assert waiting.waiting.wait(2), "action did not reach the dispatch lock"
            getattr(body, cancel)()
            commands_after_cancel = list(bridge.commands)
        worker.join(2)
        assert not worker.is_alive()
        assert not errors
        assert results == [False]
        assert bridge.commands == commands_after_cancel
        if cancel == "halt":
            assert commands_after_cancel[-1] == ("walk.command", {"action": "stop"})
            body.resume()
            assert body.act(primitive)
            assert len(bridge.commands) > len(commands_after_cancel)
    finally:
        body.close()
        worker.join(2)
        bridge.close()
