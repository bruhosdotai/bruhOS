"""Action durations come from the bridge's profile.describe reply and must be validated.

``_play`` waits ``durations[group]`` seconds after ``action.play`` before reporting the action done.
On ``main`` each ``duration_s`` went through a bare ``float()``: a non-numeric value or a non-object
reply crashed the constructor, NaN or a negative value reported completion instantly while the
robot was still moving, and ``Infinity`` (which Python's json accepts) blocked the action forever.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

serve = pytest.importorskip("websockets.sync.server").serve

from bruhos.body.eliza_bridge import ElizaBridgeBody  # noqa: E402


class ProfileBridge:
    def __init__(self, profile_data):
        self.profile_data = profile_data
        self.server = serve(self._handle, "127.0.0.1", 0)
        self.url = f"ws://127.0.0.1:{self.server.socket.getsockname()[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _handle(self, ws):
        for raw in ws:
            msg = json.loads(raw)
            data = self.profile_data if msg["command"] == "profile.describe" else {}
            # allow_nan=True (the default) so Infinity / NaN go over the wire exactly as a peer could send them
            ws.send(json.dumps({"type": "response", "request_id": msg["request_id"], "ok": True, "data": data}))

    def close(self):
        self.server.shutdown()


def _body(profile_data):
    bridge = ProfileBridge(profile_data)
    body = ElizaBridgeBody({"body": {"url": bridge.url, "timeout_s": 1.0}, "actions": {"wave": "wave"}})
    return bridge, body


def _groups(**groups):
    return {"profile": {"actions": {"groups": groups}}}


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), -1.0, 0.0, "slow", None, [1]])
def test_unusable_duration_falls_back_to_the_default(bad):
    bridge, body = _body(_groups(wave={"duration_s": bad}, stand={"duration_s": 0.2}))
    try:
        assert "wave" not in body.durations
        assert body.durations["stand"] == 0.2              # a good neighbour entry still counts
        start = time.monotonic()
        result = {}
        t = threading.Thread(target=lambda: result.setdefault("ok", body.act("wave")), daemon=True)
        t.start()
        t.join(3.0)
        assert not t.is_alive(), "act() never returned (infinite duration)"
        assert result["ok"] is True
        assert time.monotonic() - start >= 0.9             # waited the 1.0 s default, not 0 s
    finally:
        body.halt()
        body.close()
        bridge.close()


@pytest.mark.parametrize("data", [["not", "a", "dict"], {"profile": "x"}, {"profile": {"actions": {"groups": [1]}}}])
def test_malformed_profile_reply_does_not_crash_the_constructor(data):
    bridge, body = _body(data)
    try:
        assert body.durations == {}
    finally:
        body.close()
        bridge.close()


def test_valid_durations_are_kept():
    bridge, body = _body(_groups(wave={"duration_s": 0.3}, stand={"duration_s": 2}, kick={"duration_s": "0.5"},
                                 bow={}))
    try:
        # numeric strings were accepted on main and still are; a missing duration_s keeps main's 1.0
        assert body.durations == {"wave": 0.3, "stand": 2.0, "kick": 0.5, "bow": 1.0}
    finally:
        body.close()
        bridge.close()
