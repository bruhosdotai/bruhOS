import base64
import json
import queue
import threading
import time

from bruhos.speech.base import SpeechConfig
from bruhos.speech.realtime import DuplexSession, realtime_ws_url, speech_base


def test_error_text_includes_model_load_failed():
    from bruhos.speech.realtime import error_text

    assert error_text({
        "type": "error",
        "error": {"code": "model_load_failed", "message": "weights missing"},
    }) == "model_load_failed: weights missing"


def test_realtime_url_from_v1_base():
    url = realtime_ws_url(
        "http://127.0.0.1:2083/v1",
        {"pipeline": "duplex", "model": "qwen2.5:7b", "rate": "16000", "turn": "server_vad"},
    )
    assert url.startswith("ws://127.0.0.1:2083/v1/realtime?")
    assert "pipeline=duplex" in url
    assert "turn=server_vad" in url


def test_realtime_url_from_origin():
    url = realtime_ws_url("http://127.0.0.1:2083", {"pipeline": "duplex"})
    assert url.startswith("ws://127.0.0.1:2083/v1/realtime?")


def test_duplex_query_voice_is_piper():
    cfg = SpeechConfig(
        base_url="http://127.0.0.1:2083/v1",
        tts_model="piper-lessac",
        tts_voice="lessac",
        chat_model="qwen2.5:14b",
    )
    query = DuplexSession(cfg).query(16000)
    assert query["voice"] == "piper-lessac"
    assert query["speaker"] == "lessac"
    bare = DuplexSession(SpeechConfig(chat_model="qwen2.5:14b", tts_voice="")).query(16000)
    assert bare["voice"] == "piper-lessac"
    assert "speaker" not in bare


def test_speech_base_is_config_url():
    cfg = SpeechConfig(base_url="http://127.0.0.1:2083/v1")
    assert speech_base(cfg) == "http://127.0.0.1:2083/v1"


def test_duplex_plays_deltas_and_never_commits(monkeypatch):
    pcm = b"\x01\x00" * 20
    events = [
        json.dumps({
            "type": "session.created",
            "session": {"audio": {"input": {"format": {"rate": 16000}}}},
        }),
        json.dumps({
            "type": "response.output_audio.delta",
            "mime": "audio/pcm",
            "rate": 22050,
            "delta": base64.b64encode(pcm).decode(),
        }),
        json.dumps({
            "type": "response.output_text.delta",
            "delta": "Hello.",
        }),
        json.dumps({"type": "response.done", "response": {"status": "cancelled"}}),
    ]
    sent: list[dict] = []

    class FakeWS:
        def __init__(self, url, timeout=30):
            self.url = url
            self._events = iter(events)

        def settimeout(self, timeout):
            return None

        def recv_text(self):
            try:
                nxt = next(self._events)
            except StopIteration:
                return None
            if "output_audio" in nxt:
                deadline = time.monotonic() + 1
                while time.monotonic() < deadline and not any(
                    m.get("type") == "input_audio_buffer.append" for m in sent
                ):
                    time.sleep(0.01)
            return nxt

        def send_text(self, text):
            sent.append(json.loads(text))

        def close(self):
            return None

    class Mic:
        def __init__(self):
            self.q: queue.Queue = queue.Queue()
            self.q.put(b"\x00\x00" * 480)

        def close(self):
            self.q.put(None)

    class Player:
        def __init__(self):
            self.writes: list[tuple[int, bytes]] = []
            self.flushed = False

        def write(self, data, rate):
            self.writes.append((rate, data))

        def flush(self):
            self.flushed = True

    monkeypatch.setattr("bruhos.speech.realtime.WebSocket", FakeWS)
    cfg = SpeechConfig(
        base_url="http://127.0.0.1:2083/v1",
        asr_model="whisper-base:latest",
        tts_model="piper-lessac:latest",
        chat_model="qwen2.5:7b",
        realtime="duplex",
        realtime_turn="server_vad",
    )
    player = Player()
    DuplexSession(cfg).run(threading.Event(), device="plughw:Device,0", player=player, mic=Mic(), once=True)
    kinds = [m["type"] for m in sent]
    assert "input_audio_buffer.commit" not in kinds
    update = next(m for m in sent if m["type"] == "session.update")
    instructions = update["session"]["instructions"]
    vad = update["session"]["audio"]["input"]["turn_detection"]
    assert vad["threshold"] == 0.08
    assert vad["silence_duration_ms"] == 700
    assert "Tony" in instructions
    assert "camera" not in instructions.lower()
    assert "cannot walk" in instructions
    assert "concise" not in instructions.lower()
    assert "sentence" not in instructions.lower()
    assert "input_audio_buffer.append" in kinds
    assert player.writes == [(22050, pcm)]
    assert player.flushed


def test_duplex_logs_debug_status(monkeypatch, capsys):
    events = [
        json.dumps({
            "type": "session.created",
            "session": {
                "audio": {"input": {"format": {"rate": 16000}}},
                "asr": "qwen3-asr:latest",
                "asr_backend": "qwen3-asr",
            },
        }),
        json.dumps({
            "type": "response.status",
            "status": "silence_started",
            "silence_duration_ms": 300,
        }),
        json.dumps({
            "type": "conversation.item.input_audio_transcription.started",
            "item_id": "it_1",
        }),
        json.dumps({"type": "response.status", "status": "transcribing"}),
        json.dumps({"type": "response.status", "status": "generating"}),
        json.dumps({"type": "response.status", "status": "first_token"}),
        json.dumps({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "Hello",
            "asr": "qwen3-asr:latest",
            "asr_backend": "qwen3-asr",
        }),
        json.dumps({"type": "response.output_text.delta", "delta": "Hi there."}),
        json.dumps({"type": "response.status", "status": "audio_done"}),
        json.dumps({"type": "response.done", "response": {"status": "completed"}}),
    ]

    class FakeWS:
        def __init__(self, url, timeout=30):
            self._events = iter(events)

        def settimeout(self, timeout):
            return None

        def recv_text(self):
            try:
                return next(self._events)
            except StopIteration:
                return None

        def send_text(self, text):
            return None

        def close(self):
            return None

    class Mic:
        def __init__(self):
            self.q: queue.Queue = queue.Queue()

        def close(self):
            self.q.put(None)

    class Player:
        def write(self, data, rate):
            return None

        def flush(self):
            return None

    monkeypatch.setattr("bruhos.speech.realtime.WebSocket", FakeWS)
    cfg = SpeechConfig(
        base_url="http://127.0.0.1:2083/v1",
        chat_model="qwen2.5:14b",
        realtime="duplex",
        realtime_turn="server_vad",
    )
    DuplexSession(cfg).run(
        threading.Event(), device="plughw:Device,0", player=Player(), mic=Mic(), once=True,
    )
    out = capsys.readouterr().out
    assert "[ear] STATUS silence_started" in out
    assert "silence_duration_ms=300" in out
    assert "[ear] ASR started" in out
    assert "item_id=it_1" in out
    for name in ("transcribing", "generating", "first_token", "audio_done"):
        assert f"[ear] STATUS {name}" in out
    assert "asr=qwen3-asr:latest" in out
    assert "asr_backend=qwen3-asr" in out
    assert "[ear] HEARD 'Hello'" in out
    assert "[ear] SAY 'Hi there.'" in out
    assert "[ear] DONE completed" in out
    assert "audio_bytes=0" in out


def test_duplex_logs_whisper_fallback(monkeypatch, capsys):
    events = [
        json.dumps({
            "type": "session.created",
            "session": {
                "audio": {"input": {"format": {"rate": 16000}}},
                "asr": "qwen3-asr:latest",
                "asr_backend": "qwen3-asr",
            },
        }),
        json.dumps({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "Hi",
            "asr": "whisper-base:latest",
            "asr_backend": "whisper_fallback",
        }),
        json.dumps({"type": "response.created"}),
        json.dumps({"type": "response.done", "response": {"status": "completed"}}),
    ]

    class FakeWS:
        def __init__(self, url, timeout=30):
            self._events = iter(events)
            self.sent = []

        def settimeout(self, timeout):
            return None

        def recv_text(self):
            try:
                return next(self._events)
            except StopIteration:
                return None

        def send_text(self, text):
            self.sent.append(text)

        def close(self):
            return None

    class Mic:
        def __init__(self):
            self.q: queue.Queue = queue.Queue()

        def close(self):
            self.q.put(None)

    class Player:
        def write(self, data, rate):
            return None

        def flush(self):
            return None

        def release(self):
            return None

    monkeypatch.setattr("bruhos.speech.realtime.WebSocket", FakeWS)
    cfg = SpeechConfig(
        base_url="http://127.0.0.1:2083/v1",
        chat_model="qwen2.5:14b",
        realtime="duplex",
        realtime_turn="server_vad",
    )
    DuplexSession(cfg).run(
        threading.Event(), device="plughw:Device,0", player=Player(), mic=Mic(), once=True,
    )
    out = capsys.readouterr().out
    assert "[ear] ASR whisper_fallback" in out
    assert "asr_backend=whisper_fallback" in out
    assert "[ear] HEARD 'Hi'" in out


def test_mic_gate_stays_quiet_until_playback_ends():
    from bruhos.speech.realtime import MicGate

    gate = MicGate()
    loud = b"\x10\x00" * 8
    assert gate.outbound(loud) == loud
    gate.hold()
    assert gate.outbound(loud) == bytes(len(loud))
    gate.note_audio(2205, 22050)  # 50 ms
    gate.release(0.05)
    assert gate.outbound(loud) == bytes(len(loud))
    time.sleep(0.12)
    assert gate.outbound(loud) == loud


def test_duplex_holds_mic_through_playback(monkeypatch):
    script: queue.Queue = queue.Queue()
    seen: queue.Queue = queue.Queue()

    class FakeWS:
        def __init__(self, url, timeout=30):
            return None

        def settimeout(self, timeout):
            return None

        def recv_text(self):
            item = script.get()
            return None if item is None else item

        def send_text(self, text):
            msg = json.loads(text)
            if msg.get("type") == "input_audio_buffer.append":
                seen.put(base64.b64decode(msg["audio"]))
            else:
                seen.put(msg)

        def close(self):
            script.put(None)

    class Mic:
        def __init__(self):
            self.q: queue.Queue = queue.Queue()

        def close(self):
            self.q.put(None)

    class Player:
        def __init__(self):
            self.writes: list[tuple[int, bytes]] = []

        def write(self, data, rate):
            self.writes.append((rate, data))

        def flush(self):
            return None

    def appends():
        while True:
            item = seen.get(timeout=2)
            if isinstance(item, bytes):
                return item

    monkeypatch.setattr("bruhos.speech.realtime.WebSocket", FakeWS)
    mic = Mic()
    player = Player()
    script.put(json.dumps({
        "type": "session.created",
        "session": {"audio": {"input": {"format": {"rate": 16000}}}},
    }))
    cfg = SpeechConfig(
        base_url="http://127.0.0.1:2083/v1",
        chat_model="qwen2.5:14b",
        realtime="duplex",
        realtime_turn="server_vad",
    )
    worker = threading.Thread(
        target=DuplexSession(cfg).run,
        args=(threading.Event(),),
        kwargs={"device": "plughw:Device,0", "player": player, "mic": mic, "once": True},
        daemon=True,
    )
    worker.start()
    loud = b"\x10\x00" * 480
    mic.q.put(loud)
    assert appends() == loud

    script.put(json.dumps({
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": "Hello",
    }))
    script.put(json.dumps({"type": "response.created"}))
    deadline = time.monotonic() + 2
    held = False
    while time.monotonic() < deadline:
        mic.q.put(loud)
        if appends() == bytes(len(loud)):
            held = True
            break
    assert held

    pcm = b"\x01\x00" * 20
    script.put(json.dumps({
        "type": "response.output_audio.delta",
        "rate": 24000,
        "delta": base64.b64encode(pcm).decode(),
    }))
    deadline = time.monotonic() + 2
    still_held = False
    while time.monotonic() < deadline:
        mic.q.put(loud)
        if appends() == bytes(len(loud)):
            still_held = True
            break
    assert still_held
    deadline = time.monotonic() + 2
    while not player.writes and time.monotonic() < deadline:
        time.sleep(0.01)
    assert player.writes == [(24000, pcm)]

    script.put(json.dumps({"type": "response.done", "response": {"status": "completed"}}))
    worker.join(timeout=2)
    assert not worker.is_alive()


def test_duplex_skips_reply_with_no_words(monkeypatch):
    events = [
        json.dumps({
            "type": "session.created",
            "session": {"audio": {"input": {"format": {"rate": 16000}}}},
        }),
        json.dumps({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "[BLANK_AUDIO]",
        }),
        json.dumps({"type": "response.created"}),
        json.dumps({
            "type": "response.output_audio.delta",
            "rate": 22050,
            "delta": base64.b64encode(b"\x01\x00" * 20).decode(),
        }),
        json.dumps({"type": "response.output_text.delta", "delta": "Got it. What do you need, briefly?"}),
        json.dumps({"type": "response.done", "response": {"status": "cancelled"}}),
    ]
    sent: list[dict] = []

    class FakeWS:
        def __init__(self, url, timeout=30):
            self._events = iter(events)

        def settimeout(self, timeout):
            return None

        def recv_text(self):
            try:
                return next(self._events)
            except StopIteration:
                return None

        def send_text(self, text):
            sent.append(json.loads(text))

        def close(self):
            return None

    class Mic:
        def __init__(self):
            self.q: queue.Queue = queue.Queue()

        def close(self):
            self.q.put(None)

    class Player:
        def __init__(self):
            self.writes: list[tuple[int, bytes]] = []

        def write(self, data, rate):
            self.writes.append((rate, data))

        def flush(self):
            return None

    monkeypatch.setattr("bruhos.speech.realtime.WebSocket", FakeWS)
    player = Player()
    cfg = SpeechConfig(
        base_url="http://127.0.0.1:2083/v1",
        chat_model="qwen2.5:14b",
        realtime="duplex",
    )
    DuplexSession(cfg).run(
        threading.Event(), device="plughw:Device,0", player=player, mic=Mic(), once=True,
    )
    assert any(m.get("type") == "response.cancel" for m in sent)
    assert player.writes == []
