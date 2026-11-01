"""Eliza bridge [actions] config: unknown primitive names must not be silently dropped."""

import json
import threading
import tomllib
from pathlib import Path

import pytest

from bruhos.body.base import PRIMITIVES
from bruhos.body.eliza_bridge import ElizaBridgeBody

ROOT = Path(__file__).resolve().parents[1]

PROFILE = {"profile": {"actions": {"groups": {"stand": {"duration_s": 0.2}, "wave": {"duration_s": 0.3}}}}}


def test_unknown_action_key_is_rejected_at_load():
    cfg = {"body": {"url": "ws://127.0.0.1:1"}, "actions": {"stand_up_fron": "stand_up_front"}}
    with pytest.raises(ValueError, match="stand_up_fron"):
        ElizaBridgeBody(cfg)


def test_shipped_config_only_uses_known_primitives():
    with open(ROOT / "configs" / "eliza_ainex.toml", "rb") as f:
        actions = tomllib.load(f)["actions"]
    assert set(actions) <= set(PRIMITIVES)


def test_valid_action_keys_load_and_dispatch():
    pytest.importorskip("websockets")
    from websockets.sync.server import serve

    commands = []

    def handle(ws):
        ws.send(json.dumps({"type": "event", "event": "session.hello", "data": {}}))
        for raw in ws:
            msg = json.loads(raw)
            commands.append(msg["command"])
            data = PROFILE if msg["command"] == "profile.describe" else {}
            ws.send(json.dumps({"type": "response", "request_id": msg["request_id"],
                                "ok": True, "data": data}))

    server = serve(handle, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
        body = ElizaBridgeBody({"body": {"url": url}, "actions": {"wave": "wave"}})
        try:
            assert body.actions["wave"] == "wave"
            assert body.act("wave") is True
            assert "action.play" in commands
        finally:
            body.close()
    finally:
        server.shutdown()
