import json

from websockets.sync.server import serve

from bruhos.body.eliza_bridge import ElizaBridgeBody


def test_connection_loss_during_action_is_not_reported_as_completion():
    def handle(ws):
        for raw in ws:
            message = json.loads(raw)
            data = {"profile": {"actions": {"groups": {"wave": {"duration_s": 0.2}}}}}
            ws.send(json.dumps({"type": "response", "request_id": message["request_id"],
                                "ok": True, "data": data if message["command"] == "profile.describe" else {}}))
            if message["command"] == "action.play":
                ws.close()
                return

    import threading

    with serve(handle, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        body = ElizaBridgeBody({"body": {"url": f"ws://127.0.0.1:{server.socket.getsockname()[1]}"},
                                "actions": {"wave": "wave"}})
        try:
            assert body.act("wave") is False
            assert body._closed.wait(1)
        finally:
            body.close()
