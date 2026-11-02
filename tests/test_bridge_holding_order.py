"""Holding observations must follow the order of serialized bridge commands."""
import threading
import pytest

pytest.importorskip('websockets')
from test_eliza_bridge import FakeBridge  # noqa: E402
from bruhos.body.eliza_bridge import ElizaBridgeBody  # noqa: E402


class PausingUnlock:
    """Expose a possible scheduling point after the real dispatch lock releases."""
    def __init__(self, lock, pause):
        self.lock, self.pause = lock, pause
        self.released, self.proceed = threading.Event(), threading.Event()

    def __enter__(self):
        self.lock.acquire()

    def __exit__(self, *exc):
        self.lock.release()
        if threading.current_thread().name == 'first-bridge-action':
            self.released.set()
            if self.pause:
                assert self.proceed.wait(10), 'test did not release the first caller'


@pytest.mark.parametrize('pause', [False, True], ids=['sequential', 'interleaved'])
@pytest.mark.parametrize('first,second,expected', [('grab', 'release', None), ('release', 'grab', 'object')])
def test_holding_follows_serialized_action_order(first, second, expected, pause):
    bridge = FakeBridge()
    body = ElizaBridgeBody({'body': {'url': bridge.url}, 'actions': {'grab': 'grab', 'release': 'release'}})
    body.holding = 'object' if first == 'release' else None
    gate = body._lock = PausingUnlock(body._lock, pause)
    results, errors = [], []

    def act_first():
        try:
            results.append(body.act(first))
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=act_first, name='first-bridge-action', daemon=True)
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
        played = [payload['name'] for command, payload in bridge.commands if command == 'action.play']
        assert played == [first, second]
        assert body.observe().holding == expected
    finally:
        gate.proceed.set()
        worker.join(10)
        body.close()
        bridge.close()
