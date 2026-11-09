import io
import json
import wave
from pathlib import Path

import pytest

from bruhos.listen.chat import ChatReply, clean_reply
from bruhos.listen.normalize import is_wav, normalize_to_wav
from bruhos.speech import SpeechConfig
from bruhos.web.server import BruhOSServer


def _silent_wav(seconds: float = 0.1, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    n = int(rate * seconds)
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * n)
    return buf.getvalue()


def test_is_wav_and_normalize_passthrough():
    data = _silent_wav()
    assert is_wav(data)
    path = normalize_to_wav(data, suffix=".wav")
    try:
        assert path.suffix == ".wav" and path.stat().st_size > 44
    finally:
        path.unlink(missing_ok=True)


def test_clean_reply_strips_leaks():
    raw = "Assistant: You'll be the judge. 🐾\nJudge: Order in the court!"
    out = clean_reply(raw)
    assert "Assistant:" not in out and "🐾" not in out
    assert "Order in the court" in out


def test_chat_reply_request_shape():
    captured = []

    def transport(url, headers, body, timeout):
        captured.append((url, json.loads(body)))
        return json.dumps({"choices": [{"message": {"content": "Hello there."}}]}).encode()

    text = ChatReply(base_url="http://h/v1", model="m", transport=transport).reply("hi")
    assert text == "Hello there."
    assert captured[0][0] == "http://h/v1/chat/completions"
    assert captured[0][1]["model"] == "m"
    assert captured[0][1]["messages"][-1] == {"role": "user", "content": "hi"}


def test_ptt_keeps_conversation_context(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    seen = []

    class FakeASR:
        def __init__(self):
            self.n = 0
        def transcribe(self, path: str) -> str:
            self.n += 1
            return "my name is Sam" if self.n == 1 else "what is my name"

    def transport(url, headers, body, timeout):
        payload = json.loads(body)
        seen.append(payload["messages"])
        # echo last user; on second turn prove history was sent
        user = [m for m in payload["messages"] if m["role"] == "user"][-1]["content"]
        reply = "Nice to meet you, Sam." if "Sam" in user else "Your name is Sam."
        return json.dumps({"choices": [{"message": {"content": reply}}]}).encode()

    asr = FakeASR()
    chat = ChatReply(base_url="http://h/v1", model="m", transport=transport)
    srv = BruhOSServer(
        "127.0.0.1",
        0,
        audio_dir,
        SpeechConfig(asr="openai", base_url="http://unused/v1", tts="print"),
        chat,
        speak_replies=False,
        asr_factory=lambda: asr,
    )
    host, port = srv.server_address[:2]

    import threading
    import urllib.request

    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        def ptt():
            req = urllib.request.Request(
                f"http://{host}:{port}/api/ptt",
                data=_silent_wav(),
                headers={"Content-Type": "audio/wav", "X-Filename": "ptt.wav"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as r:
                return json.loads(r.read().decode())

        b1 = ptt()
        assert b1["turns"] == 1 and "Sam" in b1["reply"]
        b2 = ptt()
        assert b2["turns"] == 2
        # second request must include prior user+assistant
        roles = [m["role"] for m in seen[1]]
        assert roles.count("user") >= 2 and "assistant" in roles
        assert any(m.get("content") == "my name is Sam" for m in seen[1])

        req = urllib.request.Request(f"http://{host}:{port}/api/reset", data=b"{}", method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            assert json.loads(r.read().decode())["turns"] == 0
        assert srv.conversation.turns == 0
    finally:
        srv.shutdown()
        srv.server_close()


def test_normalize_missing_ffmpeg(monkeypatch):
    from bruhos.listen import normalize as n

    monkeypatch.setattr(n.shutil, "which", lambda _: None)
    with pytest.raises(n.NormalizeError, match="ffmpeg"):
        n.require_ffmpeg()


def _tiny_jpeg(seed: int = 0) -> bytes:
    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)
    img = np.full((48, 64, 3), 40 + (seed % 50), dtype=np.uint8)
    img[:, ::4] = 200
    img[::4, :] = 200
    img = np.clip(img.astype(np.int16) + rng.integers(-3, 4, img.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    assert ok
    return bytes(buf)


def test_see_job_updates_images_and_stages(tmp_path):
    from bruhos.scan.ground import FrameResult, GroundedDet
    from bruhos.web.see import SeeHub

    jpeg = _tiny_jpeg(2)
    calls = {"n": 0}

    def grab():
        calls["n"] += 1
        return jpeg

    class FakeVision:
        def list_objects(self, images):
            assert images and images[0][0] == "see"
            return ["shelf", "tv"]

    def fake_ground(**kwargs):
        return FrameResult(
            image="see", backend="test", seconds=0.01, width=64, height=48,
            detections=[GroundedDet("shelf", 5, 5, 40, 30, conf=0.9, bearing_deg=10.0, elev_deg=-2.0)],
        )

    hub = SeeHub(
        tmp_path,
        {
            "camera": {"enabled": True, "width": 64, "height": 48, "hfov_deg": 62},
            "scan": {
                "retune": False,
                "burst_frames": 1,
                "base_url": "http://example/v1",
                "model": "ministral-3:3b",
            },
        },
        grab=grab,
        vision=FakeVision(),
        ground_fn=fake_ground,
    )
    job = hub.start()
    import time
    for _ in range(100):
        if job.state in ("done", "error"):
            break
        time.sleep(0.05)
    assert job.state == "done", job.error
    d = job.to_dict()
    assert d["images"].get("raw") and d["images"].get("best") and d["images"].get("overlay")
    assert (tmp_path / "see" / job.id / "overlay.jpg").is_file()
    assert d["inventory"] == ["shelf", "tv"]
    assert d["detections"][0]["label"] == "shelf"
    statuses = {s["id"]: s["status"] for s in d["stages"]}
    assert statuses["capture"] == "done"
    assert statuses["inventory"] == "done"
    assert statuses["ground"] == "done"
    assert statuses["retune"] == "skipped"


def test_see_api_post_and_poll(tmp_path):
    import threading
    import time
    import urllib.request

    from bruhos.scan.ground import FrameResult, GroundedDet
    from bruhos.web.see import SeeHub
    from bruhos.web.server import BruhOSServer

    jpeg = _tiny_jpeg(3)

    hub = SeeHub(
        tmp_path,
        {"camera": {"width": 64, "height": 48}, "scan": {"retune": False, "burst_frames": 1, "base_url": "http://x/v1"}},
        grab=lambda: jpeg,
        vision=type("V", (), {"list_objects": lambda self, images: ["door"]})(),
        ground_fn=lambda **kw: FrameResult(
            image="see", backend="t", seconds=0.01, width=64, height=48,
            detections=[GroundedDet("door", 1, 1, 20, 20, bearing_deg=1.0, elev_deg=0.0)],
        ),
    )
    srv = BruhOSServer(
        "127.0.0.1", 0, tmp_path,
        SpeechConfig(asr="openai", base_url="http://unused/v1", tts="print"),
        speak_replies=False, see=hub,
    )
    host, port = srv.server_address[:2]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        req = urllib.request.Request(f"http://{host}:{port}/api/see", data=b"{}", method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            job = json.loads(r.read().decode())
        assert job["id"] and job["state"] in ("queued", "running", "done")
        for _ in range(100):
            with urllib.request.urlopen(f"http://{host}:{port}/api/see/{job['id']}", timeout=5) as r:
                st = json.loads(r.read().decode())
            if st["state"] in ("done", "error"):
                break
            time.sleep(0.05)
        assert st["state"] == "done"
        img = st["images"]["overlay"]
        with urllib.request.urlopen(f"http://{host}:{port}{img}", timeout=5) as r:
            assert r.headers.get("Content-Type", "").startswith("image/")
            assert len(r.read()) > 50
    finally:
        srv.shutdown()
        srv.server_close()


def test_head_control_grid_and_nudge():
    from bruhos.body.head import Head
    from bruhos.web.head_ctrl import HeadControl

    class FakeCtl:
        def __init__(self):
            self.calls = []
        def set_pwm_servo_pulse(self, sid, pulse, ms):
            self.calls.append((sid, pulse, ms))

    ctl = FakeCtl()
    head = Head(ctl, pitch_center=914, yaw_center=1477, yaw_offset_us=400, pitch_max_us=1800)
    hc = HeadControl({"head": {}}, head=head)
    st = hc.status()
    assert st["ok"] and len(st["grid"]) == 3 and len(st["grid"][0]) == 3
    # top row is look-up (row 2)
    assert [c["name"] for c in st["grid"][0]] == ["L2", "C2", "R2"]
    st2 = hc.go_pose("L2")
    assert st2["current"]["name"] == "L2"
    assert any(c[0] == 2 for c in ctl.calls)  # yaw servo
    before = st2["current"]["yaw_us"]
    st3 = hc.nudge("right")
    assert st3["current"]["yaw_us"] > before
    st4 = hc.center()
    assert st4["current"]["yaw_us"] == 1477


def test_head_api(tmp_path):
    import threading
    import urllib.request

    from bruhos.body.head import Head
    from bruhos.web.head_ctrl import HeadControl
    from bruhos.web.see import SeeHub
    from bruhos.web.server import BruhOSServer

    class FakeCtl:
        def set_pwm_servo_pulse(self, sid, pulse, ms):
            pass

    jpeg = _tiny_jpeg(9)
    hc = HeadControl(
        {},
        head=Head(FakeCtl(), pitch_center=900, yaw_center=1500, pitch_max_us=1800, move_ms=1),
    )
    hub = SeeHub(tmp_path, {"camera": {}, "scan": {"retune": False}}, grab=lambda: jpeg)
    srv = BruhOSServer(
        "127.0.0.1", 0, tmp_path,
        SpeechConfig(asr="openai", base_url="http://unused/v1", tts="print"),
        speak_replies=False, head=hc, see=hub,
    )
    host, port = srv.server_address[:2]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/head", timeout=5) as r:
            st = json.loads(r.read().decode())
        assert st["ok"] and st["grid"]
        req = urllib.request.Request(
            f"http://{host}:{port}/api/head",
            data=json.dumps({"pose": "C1", "preview": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            st2 = json.loads(r.read().decode())
        assert st2["current"]["name"] == "C1"
        assert st2.get("preview") in ("/see/live/still.jpg", "/eye/frame.jpg")
        with urllib.request.urlopen(f"http://{host}:{port}{st2['preview']}", timeout=5) as r:
            assert r.headers.get("Content-Type", "").startswith("image/")
            assert len(r.read()) > 50
    finally:
        srv.shutdown()
        srv.server_close()
