"""A failed bridge send must fail the primitive, not crash the episode.

``ElizaBridgeBody.act`` only catches ``BridgeError`` around ``_walk`` /
``_play``, and ``_send`` only raised ``BridgeError`` when no response
arrived. But when the connection drops, the transport itself raises while
sending (``ConnectionClosed``, ``ConnectionResetError``, ...) — that error
escaped ``act()`` untouched, propagated through ``run_skill`` (the runtime
has no guard around it) and killed the whole episode with a traceback
instead of failing the one skill and letting the runtime replan.

The stub socket below answers requests like the fake bridge when it works
and raises a transport error when it is told to fail, so the failure is
deterministic instead of depending on a close/send race.
"""

import json

import pytest

pytest.importorskip("websockets")

from websockets.exceptions import ConnectionClosed

from bruhos.body.eliza_bridge import BridgeError, ElizaBridgeBody


class StubSocket:
    """Answers bridge requests in-process; raises when told to fail."""

    def __init__(self, body):
        self.body = body
        self.fail = False
        self.sent: list[str] = []

    def send(self, raw):
        if self.fail:
            raise ConnectionClosed(None, None)
        msg = json.loads(raw)
        self.sent.append(msg["command"])
        slot = self.body._pending.get(msg["request_id"])
        if slot is not None:
            slot[1].append({"ok": True, "data": {}})
            slot[0].set()

    def close(self):
        pass


def _body(monkeypatch):
    # Construction must not need a live bridge for this test.
    monkeypatch.setattr(ElizaBridgeBody, "_connect", lambda self: None)
    monkeypatch.setattr(ElizaBridgeBody, "_action_durations", lambda self: {})
    body = ElizaBridgeBody({
        "body": {"url": "ws://127.0.0.1:1"},
        "walk": {"forward_s": 0.05, "turn_s": 0.05, "heartbeat_s": 0.05},
        "actions": {"stand": "stand", "wave": "wave"},
    })
    body.ws = StubSocket(body)
    body.durations = {"wave": 0.01}
    return body


def test_send_transport_failure_is_bridge_error(monkeypatch):
    body = _body(monkeypatch)
    body.ws.fail = True
    with pytest.raises(BridgeError, match="action.play"):
        body._send("action.play", {"name": "wave"})
    assert body._pending == {}


def test_act_returns_false_when_send_fails(monkeypatch):
    body = _body(monkeypatch)
    body.ws.fail = True
    assert body.act("wave") is False
    assert body.act("forward") is False
    assert body._pending == {}


def test_act_recovers_after_send_failure(monkeypatch):
    body = _body(monkeypatch)
    body.ws.fail = True
    assert body.act("wave") is False
    body.ws.fail = False
    assert body.act("wave") is True
    assert body.act("forward") is True
    assert body._pending == {}
