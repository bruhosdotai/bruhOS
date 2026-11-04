import json
import threading

import pytest

pytest.importorskip("websockets")
from websockets.sync.server import serve  # noqa: E402

from bruhos.body.eliza_bridge import ElizaBridgeBody  # noqa: E402


@pytest.mark.parametrize("primitive", ["forward", "turn_left"])
@pytest.mark.parametrize("drop_start_response", [True, False])
def test_walk_stops_after_start_even_if_its_response_is_lost(primitive, drop_start_response):
    messages = []
    walking = threading.Event()

    def handle(ws):
        for raw in ws:
            message = json.loads(raw)
            messages.append(message)
            if message["command"] == "walk.command":
                if message["payload"]["action"] == "start":
                    walking.set()
                    if drop_start_response and sum(
                        m["command"] == "walk.command" and m["payload"]["action"] == "start"
                        for m in messages
                    ) == 1:
                        continue
                else:
                    walking.clear()
            ws.send(json.dumps({"type": "response", "request_id": message["request_id"],
                                "ok": True, "data": {}}))

    with serve(handle, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        body = ElizaBridgeBody({
            "body": {"url": f"ws://127.0.0.1:{server.socket.getsockname()[1]}", "timeout_s": 0.2},
            "walk": {"forward_s": 0.05, "turn_s": 0.05, "heartbeat_s": 0.01},
        })
        try:
            assert body.act(primitive) is (not drop_start_response)
            # Inspect before close(), whose halt would otherwise hide the missing stop.
            commands = [m for m in messages if m["command"] == "walk.command"]
            assert [m["payload"]["action"] for m in commands] == ["start", "stop"]
            assert commands[-1]["preempt"] is True
            assert not walking.is_set()
            assert not body._pending
            if drop_start_response:
                assert body.act(primitive) is True
                assert not walking.is_set()
                assert not body._pending
        finally:
            body.close()
