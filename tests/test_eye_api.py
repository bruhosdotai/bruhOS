import json
import threading
import urllib.request

from bruhos.eye.state import EyeState, write_frame, write_state
from bruhos.speech.base import SpeechConfig
from bruhos.web.server import BruhOSServer


def test_api_eye_and_frame(tmp_path, monkeypatch):
    write_frame(b"\xff\xd8\xff\xd9" + b"x" * 80, tmp_path)
    write_state(
        EyeState(caption="a cup", objects=["cup"], mean=55.0, frame_seq=1, understood_seq=1, updated=1.0),
        tmp_path,
    )
    monkeypatch.setattr("bruhos.eye.state.DEFAULT_DIR", tmp_path)

    srv = BruhOSServer(
        "127.0.0.1", 0, tmp_path,
        SpeechConfig(asr="openai", base_url="http://unused/v1", tts="print"),
        speak_replies=False,
    )
    host, port = srv.server_address[:2]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/eye", timeout=5) as r:
            st = json.loads(r.read().decode())
        assert st["ok"] and st["caption"] == "a cup"
        assert st["frame"] == "/eye/frame.jpg"
        with urllib.request.urlopen(f"http://{host}:{port}/eye/frame.jpg", timeout=5) as r:
            assert r.headers.get("Content-Type", "").startswith("image/")
            assert len(r.read()) > 50
    finally:
        srv.shutdown()
        srv.server_close()
