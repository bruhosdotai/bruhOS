"""One malformed frame from the bridge must not stop the body from seeing the next telemetry.

On ``main`` the reader thread did ``json.loads(raw)`` and ``msg.get(...)`` with no per-frame guard,
so a non-JSON frame, a JSON array or ``null`` raised inside the loop, the thread exited, ``_closed``
was set, every later ``act`` returned False and ``observe`` kept reporting the last telemetry: a
fall that arrived right after the bad frame was never seen.
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from bruhos.body import eliza_bridge

BAD_FRAMES = ["not json {", "[1, 2, 3]", "null", '"telemetry"', "42", ""]


@pytest.mark.parametrize("raw", BAD_FRAMES)
def test_parse_frame_rejects_non_object_frames(raw):
    from bruhos.body.eliza_bridge import parse_frame

    assert parse_frame(raw) is None


def test_parse_frame_accepts_an_object():
    from bruhos.body.eliza_bridge import parse_frame

    assert parse_frame('{"type": "event", "event": "telemetry.basic", "data": {"imu_pitch": 1.4}}') == {
        "type": "event", "event": "telemetry.basic", "data": {"imu_pitch": 1.4}}


# ---- end to end, needs the optional websockets extra ----

PROFILE = {"profile": {"actions": {"groups": {"wave": {"duration_s": 0.05}}}}}


class BadFrameBridge:
    """Answers commands; after the first action.play sends one bad frame, then a fall."""

    def __init__(self, bad_frame: str):
        serve = pytest.importorskip("websockets.sync.server").serve
        self.bad_frame = bad_frame
        self.commands: list[str] = []
        self.server = serve(self._handle, "127.0.0.1", 0)
        self.url = f"ws://127.0.0.1:{self.server.socket.getsockname()[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _event(self, ws, event, data):
        ws.send(json.dumps({"type": "event", "event": event, "timestamp": "t", "backend": "fake", "data": data}))

    def _handle(self, ws):
        self._event(ws, "session.hello", {})
        self._event(ws, "telemetry.basic", {"imu_roll": 0.0, "imu_pitch": 0.0})
        for raw in ws:
            msg = json.loads(raw)
            self.commands.append(msg["command"])
            data = PROFILE if msg["command"] == "profile.describe" else {}
            ws.send(json.dumps({"type": "response", "request_id": msg["request_id"], "timestamp": "t",
                                "ok": True, "backend": "fake", "message": "ok", "data": data}))
            if self.commands.count("action.play") == 1 and msg["command"] == "action.play":
                ws.send(self.bad_frame)
                self._event(ws, "telemetry.basic", {"imu_roll": 0.0, "imu_pitch": 1.4})   # ~80 deg: fallen

    def close(self):
        self.server.shutdown()


@pytest.mark.parametrize("raw", ["not json {", "[1, 2, 3]", "null"])
def test_fall_after_a_bad_frame_is_still_seen(raw):
    bridge = BadFrameBridge(raw)
    body = eliza_bridge.ElizaBridgeBody({"body": {"url": bridge.url, "timeout_s": 1.0}, "actions": {"wave": "wave"}})
    try:
        assert body.act("wave") is True
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not body.observe().imu.fallen:
            time.sleep(0.02)
        assert body._reader.is_alive(), "reader thread died on a malformed frame"
        assert body.observe().imu.fallen, "telemetry after the bad frame was never read"
        assert body.act("wave") is True                      # the body is still usable
        assert bridge.commands.count("action.play") == 2
    finally:
        body.close()
        bridge.close()
