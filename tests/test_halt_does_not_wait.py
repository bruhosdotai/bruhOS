"""An emergency stop must not wait out the bridge timeout.

``ElizaBridgeBody.halt`` waited for the bridge's reply to the stop command with the full command
timeout (5 s by default). The bridge stops motion on its own after 1 s without a command, so a
halt that blocks for 5 s is worse than doing nothing: ``Runtime.stop`` and the ``stop`` skill
both call it, and the robot keeps walking while they wait.
"""

import json
import threading
import time

import pytest

pytest.importorskip("websockets")
from websockets.sync.server import serve  # noqa: E402

from bruhos.body.eliza_bridge import ElizaBridgeBody  # noqa: E402


class SlowBridge:
    """A busy bridge: every command is answered only after ``delay_s``."""

    def __init__(self, delay_s: float):
        self.delay = delay_s
        self.commands: list[tuple[str, dict]] = []
        self.server = serve(self._handle, "127.0.0.1", 0)
        self.url = f"ws://127.0.0.1:{self.server.socket.getsockname()[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _handle(self, ws):
        ws.send(json.dumps({"type": "event", "event": "telemetry.basic",
                            "data": {"imu_roll": 0.0, "imu_pitch": 0.0}}))
        for raw in ws:
            msg = json.loads(raw)
            self.commands.append((msg["command"], msg["payload"]))
            time.sleep(self.delay)
            ws.send(json.dumps({"type": "response", "request_id": msg["request_id"],
                                "ok": True, "data": {}}))

    def close(self):
        self.server.shutdown()


def test_halt_does_not_wait_out_the_bridge_timeout():
    bridge = SlowBridge(delay_s=2.0)
    body = ElizaBridgeBody({"body": {"url": bridge.url, "timeout_s": 5.0},
                            "actions": {"stand": "stand", "wave": "wave"}})
    try:
        t0 = time.monotonic()
        body.halt()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.6, f"halt blocked for {elapsed:.2f}s on a busy bridge"
        assert body._halted.is_set()
    finally:
        body.close()
        bridge.close()

