"""A halt must not be followed by another motion command from an in-flight walk."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

pytest.importorskip("websockets")
from websockets.sync.server import serve  # noqa: E402

from bruhos.body.eliza_bridge import ElizaBridgeBody  # noqa: E402
from bruhos.brain.adapters.rules import RulesPlanner  # noqa: E402
from bruhos.runtime import Runtime  # noqa: E402


class DelayedBridge:
    """Real loopback protocol peer that can hold the first walk.set response."""

    def __init__(self, *, delay_setup=True, target_y=0.0):
        self.commands = []
        self.waiting = threading.Event()
        self.pending = None
        self.delay_setup = delay_setup
        self.target_y = target_y
        self.server = serve(self.handle, "127.0.0.1", 0)
        self.url = f"ws://127.0.0.1:{self.server.socket.getsockname()[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @staticmethod
    def respond(ws, msg):
        ws.send(json.dumps({"type": "response", "request_id": msg["request_id"],
                            "ok": True, "data": {}}))

    def handle(self, ws):
        ws.send(json.dumps({"type": "event", "event": "telemetry.perception",
                            "data": {"entities": [{"label": "red ball", "x": 1,
                                                    "y": self.target_y}]}}))
        for raw in ws:
            msg = json.loads(raw)
            self.commands.append((msg["command"], msg["payload"]))
            if self.delay_setup and msg["command"] == "walk.set" and self.pending is None:
                self.pending = (ws, msg)
                self.waiting.set()
                continue
            self.respond(ws, msg)

    def release(self):
        self.respond(*self.pending)

    def close(self):
        self.server.shutdown()
        self.thread.join(2)


def make_body(bridge):
    return ElizaBridgeBody({"body": {"url": bridge.url, "timeout_s": 3},
                            "walk": {"forward_s": 0.04, "turn_s": 0.04,
                                     "heartbeat_s": 0.01}})


@pytest.mark.parametrize(("instruction", "target_y", "walk_key"), [
    ("Walk toward the red ball.", 0.0, "x"),
    ("Look at the red ball.", 1.0, "yaw"),
])
def test_runtime_stop_during_walk_setup_does_not_restart(instruction, target_y, walk_key):
    bridge = DelayedBridge(target_y=target_y)
    body = make_body(bridge)
    rt = Runtime(body, RulesPlanner())
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(rt.run, instruction)
            assert bridge.waiting.wait(2), "real skill did not reach walk.set"
            assert bridge.pending[1]["payload"][walk_key] > 0
            rt.stop()
            boundary = len(bridge.commands)
            assert bridge.commands[-1] == ("walk.command", {"action": "stop"})
            # The initial response arrives only after Runtime.stop() has returned.
            bridge.release()
            ep = future.result(timeout=3)
        assert not any(c == "walk.set" or (c == "walk.command" and p.get("action") == "start")
                       for c, p in bridge.commands[boundary:])
        assert not ep.success
        assert ep.end_reason == "stopped"

        rt.reset_stop()
        assert body.act("forward"), "an explicit resume must allow a new action"
        assert ("walk.command", {"action": "start"}) in bridge.commands[boundary:]
        assert bridge.commands[-1] == ("walk.command", {"action": "stop"})
    finally:
        body.close()
        bridge.close()


def test_uninterrupted_walk_still_heartbeats_and_stops():
    bridge = DelayedBridge(delay_setup=False)
    body = make_body(bridge)
    try:
        assert body.act("forward")
        assert bridge.commands[:3] == [
            ("profile.describe", {}),
            ("walk.set", {"speed": 2, "height": 0.036, "x": 0.02, "y": 0.0, "yaw": 0.0}),
            ("walk.command", {"action": "start"}),
        ]
        assert sum(c == "walk.set" for c, _ in bridge.commands) >= 2
        assert bridge.commands[-1] == ("walk.command", {"action": "stop"})
    finally:
        body.close()
        bridge.close()
