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
    assert query["language"] == "en"
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
    assert vad["threshold"] == 0.04
    assert vad["silence_duration_ms"] == 2000
    assert "You are Tony." in instructions
    assert "one or two short sentences of plain English" in instructions
    assert "No Chinese" in instructions
    assert "Mention the camera only" in instructions
    assert "The camera shows:" not in instructions
    assert update["session"]["audio"]["input"]["transcription"]["language"] == "en"
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


def test_pardon_line_is_english_by_default():
    from bruhos.speech.realtime import pardon_line

    assert pardon_line("en") == "Pardon?"
    assert pardon_line("zh") == "什么？"


def test_reconnect_after_a_live_socket_asks_pardon(monkeypatch):
    connections = {"n": 0}
    said: list[str] = []

    class FakeWS:
        def __init__(self, url, timeout=30):
            connections["n"] += 1
            self.n = connections["n"]

        def settimeout(self, timeout):
            return None

        def recv_text(self):
            if self.n == 1:
                if not hasattr(self, "_opened"):
                    self._opened = True
                    return json.dumps({
                        "type": "session.created",
                        "session": {"audio": {"input": {"format": {"rate": 16000}}}},
                    })
                return None
            return json.dumps({
                "type": "session.created",
                "session": {"audio": {"input": {"format": {"rate": 16000}}}},
            })

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

    def _say(self, gate):
        said.append("pardon")
        stop.set()

    monkeypatch.setattr("bruhos.speech.realtime.WebSocket", FakeWS)
    monkeypatch.setattr("bruhos.speech.realtime.DuplexSession._say_pardon", _say)
    stop = threading.Event()
    cfg = SpeechConfig(
        base_url="http://127.0.0.1:2083/v1",
        chat_model="qwen2.5:14b",
        realtime="duplex",
    )
    DuplexSession(cfg).run(stop, device="plughw:Device,0", player=Player(), mic=Mic())
    assert said == ["pardon"]
    assert connections["n"] >= 2


def test_failed_connect_does_not_ask_pardon(monkeypatch):
    said: list[str] = []

    class FakeWS:
        def __init__(self, url, timeout=30):
            return None

        def settimeout(self, timeout):
            return None

        def recv_text(self):
            return json.dumps({"type": "error", "error": {"message": "nope"}})

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
        def flush(self):
            return None

    monkeypatch.setattr("bruhos.speech.realtime.WebSocket", FakeWS)
    monkeypatch.setattr(
        "bruhos.speech.realtime.DuplexSession._say_pardon",
        lambda self, gate: said.append("pardon"),
    )
    stop = threading.Event()

    def _wait(timeout=None):
        stop.set()
        return True

    stop.wait = _wait  # type: ignore[method-assign]
    cfg = SpeechConfig(
        base_url="http://127.0.0.1:2083/v1",
        chat_model="qwen2.5:14b",
        realtime="duplex",
    )
    DuplexSession(cfg).run(stop, device="plughw:Device,0", player=Player(), mic=Mic())
    assert said == []


def test_playback_open_does_not_unmute_a_newer_hold():
    from bruhos.speech.realtime import MicGate

    gate = MicGate()
    loud = b"\x10\x00" * 8
    gate.hold()
    token = gate.begin_playback_wait()
    gate.hold()
    gate.finish_playback(token, 0.0)
    assert gate.outbound(loud) is None
    gate.finish_playback(gate.begin_playback_wait(), 0.0)
    assert gate.outbound(loud) == loud


def test_mic_gate_stays_quiet_until_playback_ends():
    from bruhos.speech.realtime import MicGate

    gate = MicGate()
    loud = b"\x10\x00" * 8
    assert gate.outbound(loud) == loud
    gate.hold()
    assert gate.outbound(loud) is None
    gate.note_audio(2205, 22050)  # 50 ms
    gate.release(0.05)
    assert gate.outbound(loud) is None
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

    for status in ("silence_started", "transcribing", "generating", "first_token"):
        script.put(json.dumps({"type": "response.status", "status": status}))
    script.put(json.dumps({"type": "conversation.item.input_audio_transcription.started"}))
    script.put(json.dumps({
        "type": "conversation.item.input_audio_transcription.completed",
        "transcript": "Hello",
        "asr_backend": "qwen3-asr",
    }))
    script.put(json.dumps({"type": "response.created"}))
    time.sleep(0.05)
    deadline = time.monotonic() + 2
    still_open = False
    while time.monotonic() < deadline:
        mic.q.put(loud)
        if appends() == loud:
            still_open = True
            break
    assert still_open

    script.put(json.dumps({"type": "input_audio_buffer.speech_stopped"}))
    deadline = time.monotonic() + 2
    stopped = False
    while time.monotonic() < deadline:
        mic.q.put(loud)
        try:
            item = seen.get(timeout=0.15)
        except queue.Empty:
            stopped = True
            break
        if isinstance(item, bytes) and item == loud:
            continue
        if not isinstance(item, bytes):
            continue
    assert stopped

    pcm = b"\x01\x00" * 20
    script.put(json.dumps({
        "type": "response.output_audio.delta",
        "rate": 24000,
        "delta": base64.b64encode(pcm).decode(),
    }))
    deadline = time.monotonic() + 2
    stopped = False
    while time.monotonic() < deadline:
        mic.q.put(loud)
        try:
            item = seen.get(timeout=0.15)
        except queue.Empty:
            stopped = True
            break
        if isinstance(item, bytes) and item == loud:
            continue
        if not isinstance(item, bytes):
            continue
    assert stopped
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


def test_collapsed_reply_is_not_played_and_drops_the_socket(monkeypatch, capsys):
    from bruhos.speech.realtime import _reply_collapsed

    assert _reply_collapsed("Benjamin Franklin is on it.") is False
    assert _reply_collapsed("unique, 橘, 股东") is True
    assert _reply_collapsed("orangeopenharrison") is True

    stop = threading.Event()
    sent: list[dict] = []
    opens = {"n": 0}

    class FakeWS:
        def __init__(self, url, timeout=30):
            opens["n"] += 1
            if opens["n"] > 1:
                stop.set()
            events = [
                json.dumps({
                    "type": "session.created",
                    "session": {"audio": {"input": {"format": {"rate": 16000}}}},
                }),
                json.dumps({
                    "type": "response.output_audio.delta",
                    "rate": 22050,
                    "delta": base64.b64encode(b"\x01\x00" * 20).decode(),
                }),
                json.dumps({"type": "response.output_text.delta", "delta": "unique, 橘"}),
            ]
            self._events = iter(events if opens["n"] == 1 else [])

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
            self.writes = 0
            self.flushed = False

        def write(self, data, rate):
            self.writes += 1

        def flush(self):
            self.flushed = True

    monkeypatch.setattr("bruhos.speech.realtime.WebSocket", FakeWS)
    player = Player()
    cfg = SpeechConfig(
        base_url="http://127.0.0.1:2083/v1",
        chat_model="qwen2.5:14b",
        realtime="duplex",
    )
    DuplexSession(cfg).run(stop, device="plughw:Device,0", player=player, mic=Mic())
    out = capsys.readouterr().out
    assert player.flushed
    assert any(m.get("type") == "response.cancel" for m in sent)
    assert "SAY drop" in out
    assert "reply left English" in out


def test_blank_clip_does_not_mute_the_next_reply(monkeypatch):
    """A blank transcript with no reply open must not drop the next sentence."""
    pcm = b"\x02\x00" * 40
    events = [
        json.dumps({
            "type": "session.created",
            "session": {"audio": {"input": {"format": {"rate": 16000}}}},
        }),
        json.dumps({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "",
        }),
        json.dumps({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "What's on the table?",
            "asr_backend": "qwen3-asr",
        }),
        json.dumps({"type": "response.created"}),
        json.dumps({
            "type": "response.output_audio.delta",
            "rate": 22050,
            "delta": base64.b64encode(pcm).decode(),
        }),
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
    assert player.writes == [(22050, pcm)]


def test_speech_boost_lifts_a_quiet_voice_and_leaves_the_room():
    from bruhos.speech.realtime import SpeechBoost

    boost = SpeechBoost()
    room = b"\x40\x00" * 480  # peak about 0.002
    for _ in range(8):
        assert boost.apply(room) == room
    # Peak 0.03: above the room, under the 0.04 server gate.
    quiet = array_pcm(0.03)
    lifted = boost.apply(quiet)
    assert _peak(lifted) > 0.08
    assert _peak(boost.apply(room)) < 0.01


def array_pcm(peak: float) -> bytes:
    sample = max(-32768, min(32767, int(peak * 32767)))
    return array_bytes([sample] * 480)


def array_bytes(samples: list[int]) -> bytes:
    import array

    raw = array.array("h", samples)
    return raw.tobytes()


def _peak(pcm: bytes) -> float:
    import array

    samples = array.array("h")
    samples.frombytes(pcm)
    return max(abs(sample) for sample in samples) / 32768.0
