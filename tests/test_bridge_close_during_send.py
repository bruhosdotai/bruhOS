"""Graceful close must not escape from a halt queued behind a motion send."""

import threading

import pytest

pytest.importorskip("websockets")
from test_eliza_bridge import FakeBridge, _body  # noqa: E402

from bruhos.brain.adapters.rules import RulesPlanner  # noqa: E402
from bruhos.runtime import Runtime  # noqa: E402


class WatchedSendLock:
    def __init__(self, lock):
        self.lock = lock
        self.stop_waiting = threading.Event()

    def __enter__(self):
        if threading.current_thread().name == "stop-worker":
            self.stop_waiting.set()
        self.lock.acquire()

    def __exit__(self, *args):
        self.lock.release()


def test_close_while_runtime_stop_waits_for_send_lock():
    bridge = FakeBridge()
    body = _body(bridge)
    body.ws.close_timeout = 0.5
    runtime = Runtime(body, RulesPlanner())
    lock = body._send_lock = WatchedSendLock(body._send_lock)
    sent, release = threading.Event(), threading.Event()
    actual_send = body.ws.send
    results, errors, done, workers = {}, {}, {}, []

    def send(raw):
        actual_send(raw)
        if threading.current_thread().name == "action-worker":
            # A scheduling boundary after the real send, before its lock exits.
            sent.set()
            assert release.wait(5), "test did not release the sender"

    def start(name, fn):
        done[name] = threading.Event()
        def run():
            try:
                results[name] = fn()
            except Exception as exc:
                errors[name] = exc
            finally:
                done[name].set()
        worker = threading.Thread(target=run, name=name, daemon=True)
        workers.append(worker)
        worker.start()

    body.ws.send = send
    try:
        start("action-worker", lambda: body.act("wave"))
        assert sent.wait(3)
        start("stop-worker", runtime.stop)
        assert lock.stop_waiting.wait(3)
        assert not done["stop-worker"].is_set()
        start("close-worker", body.close)
        assert done["close-worker"].wait(3)
        assert body._closed.wait(1)
        body._reader.join(1)
        assert not body._reader.is_alive()

        release.set()
        assert done["stop-worker"].wait(3)
        assert done["action-worker"].wait(3)
        assert not errors
        assert results["action-worker"] is False
        assert results["stop-worker"] is None
        assert not body._pending
        assert bridge.commands == [("profile.describe", {}), ("action.play", {"name": "wave"})]
    finally:
        release.set()
        for worker in workers:
            worker.join(3)
        body.close()
        bridge.close()
