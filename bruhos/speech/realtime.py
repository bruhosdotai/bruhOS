"""Hearing: one duplex WebSocket on zerollama.

Owns ``qwen2.5:14b`` only. Mic PCM16 LE mono at 16 kHz goes out as
``input_audio_buffer.append``. With ``turn=server_vad`` this client never
sends ``input_audio_buffer.commit`` or ``response.create``. It does not call
``/api/chat``, ``/v1/chat/completions``, or send images. Vision runs in
``bruhos-eye`` on the other GPU.

``conversation.item.input_audio_transcription.completed`` is the heard text.
An empty transcript is silence. ``asr_backend=whisper_fallback`` means Whisper
heard the clip because Qwen ASR was down — keep the text and log it.
Play every ``response.output_audio.delta`` at the event rate before the turn
is finished. From ``response.created`` until playback ends, the mic uplink is
silence so the hold and the answer do not barge in on themselves.
"""

from __future__ import annotations

import base64
import json
import queue
import shutil
import subprocess
import threading
import time
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

from bruhos.listen.sight import vision_context_item
from bruhos.speech.base import SpeechConfig
from bruhos.speech.ws import WebSocket, WebSocketError

INSTRUCTIONS = (
    "You are Tony, a small humanoid robot. "
    "Talk with the person in front of you about what they just said. "
    "Your legs are not available, so you cannot walk or move."
)
# 30 ms at 16 kHz mono PCM16.
CHUNK_BYTES = 960
# How long to keep the mic silent after the speaker should have gone quiet.
ECHO_TAIL_S = 0.5
# Room noise was crossing the default 0.02 level and restarting ASR forever.
VAD_THRESHOLD = 0.08
VAD_SILENCE_MS = 700


def _no_words(text: str) -> bool:
    """True when a clip has nothing the model should answer."""
    return text.strip().lower() in {"", "[blank_audio]"}


def speech_base(cfg: SpeechConfig) -> str:
    """Speech base URL from config, with no trailing slash."""
    return (cfg.base_url or "").rstrip("/")


def realtime_ws_url(base_url: str, query: dict[str, str]) -> str:
    """Origin or ``.../v1`` → ``ws://host/v1/realtime?...``."""
    parsed = urlparse(base_url if "://" in base_url else "http://" + base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = parsed.path.rstrip("/")
    if path.endswith("/realtime"):
        pass
    elif path.endswith("/v1"):
        path = path + "/realtime"
    elif not path:
        path = "/v1/realtime"
    else:
        path = path + "/realtime"
    return urlunparse((scheme, parsed.netloc, path, "", urlencode(query), ""))


class PcmSpeaker:
    """Stream PCM16 mono to aplay. ``flush`` kills playback for barge-in."""

    def __init__(self, device: str | None = None):
        self.device = device
        self.proc: subprocess.Popen[bytes] | None = None
        self.rate: int | None = None
        self._lock = threading.Lock()

    def write(self, pcm: bytes, rate: int) -> None:
        if not pcm:
            return
        with self._lock:
            if self.proc is None or self.rate != rate or self.proc.poll() is not None:
                self._kill()
                player = shutil.which("aplay")
                if not player:
                    return
                cmd = [player, "-f", "S16_LE", "-r", str(rate), "-c", "1", "-t", "raw", "-q"]
                if self.device:
                    cmd[1:1] = ["-D", self.device]
                self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                self.rate = rate
            assert self.proc.stdin is not None
            try:
                self.proc.stdin.write(pcm)
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self._kill()

    def flush(self) -> None:
        with self._lock:
            self._kill()

    def release(self) -> None:
        """Close aplay's input so this reply can finish and the process can exit."""
        with self._lock:
            proc = self.proc
            self.proc = None
            self.rate = None
        if proc is None:
            return
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass

        def _wait() -> None:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass

        threading.Thread(target=_wait, name="bruhos-aplay-drain", daemon=True).start()

    def _kill(self) -> None:
        proc = self.proc
        self.proc = None
        self.rate = None
        if proc is None or proc.poll() is not None:
            return
        proc.kill()
        try:
            proc.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass


class LiveMic:
    """arecord until ``close``. Yields ~30 ms PCM16 chunks on ``q``."""

    def __init__(self, device: str, rate: int = 16000, channels: int = 1, chunk: int = CHUNK_BYTES):
        arecord = shutil.which("arecord")
        if not arecord:
            raise RuntimeError("arecord not found")
        self.q: queue.Queue[bytes | None] = queue.Queue()
        self.proc = subprocess.Popen(
            [arecord, "-D", device, "-f", "S16_LE", "-r", str(rate), "-c", str(channels), "-t", "raw"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        self._chunk = chunk
        threading.Thread(target=self._pump, name="bruhos-mic", daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        try:
            while True:
                chunk = self.proc.stdout.read(self._chunk)
                if not chunk:
                    break
                self.q.put(chunk)
        finally:
            self.q.put(None)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass


def error_text(event: dict) -> str:
    """``type: error`` message, including ``model_load_failed``."""
    err = event.get("error")
    if isinstance(err, dict):
        code = str(err.get("code") or err.get("type") or "")
        message = str(err.get("message") or "")
        if code and message:
            return f"{code}: {message}"
        return code or message or "error"
    return str(err or event.get("type") or "error")


def _scalar(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        text = " ".join(value.split())
        if not text:
            return None
        if len(text) > 160:
            text = text[:157] + "..."
        return text
    return None


def _event_bits(event: dict, *, skip: frozenset[str] = frozenset()) -> str:
    """One-line scalars from a realtime event. Audio payloads stay out."""
    parts: list[str] = []
    for key, value in event.items():
        if key in skip or key in ("type", "delta", "audio"):
            continue
        shown = _scalar(value)
        if shown is not None:
            parts.append(f"{key}={shown}")
            continue
        if isinstance(value, dict):
            inner = []
            for inner_key, inner_value in value.items():
                if inner_key in ("delta", "audio"):
                    continue
                inner_shown = _scalar(inner_value)
                if inner_shown is not None:
                    inner.append(f"{inner_key}={inner_shown}")
            if inner:
                parts.append(f"{key}{{{' '.join(inner)}}}")
    return " ".join(parts)


def _status_name(event: dict) -> str:
    status = event.get("status")
    if isinstance(status, str) and status:
        return status
    response = event.get("response")
    if isinstance(response, dict):
        nested = response.get("status")
        if isinstance(nested, str):
            return nested
    return ""


def _log_progress(kind: str, event: dict, say_parts: list[str], audio_bytes: int) -> None:
    """Print duplex debug events. PCM deltas are counted, not dumped."""
    if kind == "response.status":
        name = _status_name(event) or "unknown"
        extra = _event_bits(event, skip=frozenset({"status"}))
        if name == "audio_done":
            extra = f"{extra} audio_bytes={audio_bytes}".strip()
        print(f"[ear] STATUS {name} {extra}".rstrip(), flush=True)
        return
    if kind == "conversation.item.input_audio_transcription.started":
        extra = _event_bits(event)
        print(f"[ear] ASR started {extra}".rstrip(), flush=True)
        return
    if kind == "response.output_text.done":
        text = str(event.get("text") or "").strip() or "".join(say_parts).strip()
        say_parts.clear()
        if text:
            print(f"[ear] SAY {text!r}", flush=True)
        return
    if kind == "response.done":
        text = "".join(say_parts).strip()
        say_parts.clear()
        if text:
            print(f"[ear] SAY {text!r}", flush=True)
        status = _status_name(event) or "unknown"
        extra = _event_bits(event, skip=frozenset({"status", "response"}))
        print(f"[ear] DONE {status} audio_bytes={audio_bytes} {extra}".rstrip(), flush=True)
        return
    if kind in (
        "conversation.item.input_audio_transcription.completed",
        "response.cancel",
        "response.cancelled",
    ):
        return
    extra = _event_bits(event)
    print(f"[ear] EVT {kind} {extra}".rstrip(), flush=True)


class DuplexSession:
    """One duplex socket: hear, chat, and speak. History stays on the server."""

    def __init__(self, cfg: SpeechConfig, chat_model: str | None = None, sight: Any | None = None):
        self.cfg = cfg
        self.chat_model = chat_model or cfg.chat_model or ""
        self.base_url = speech_base(cfg)
        self.timeout = cfg.timeout
        self.sight = sight

    def _instructions(self) -> str:
        text = INSTRUCTIONS
        seen = ""
        if self.sight is not None:
            seen = str(getattr(self.sight, "text", "") or "").strip()
        if seen:
            text += " In front of you: " + seen
        return text

    def query(self, rate: int) -> dict[str, str]:
        q = {
            "pipeline": "duplex",
            "model": self.chat_model,
            "voice": self.cfg.tts_model or "piper-lessac",
            "asr": self.cfg.asr_model or "qwen3-asr:latest",
            "rate": str(rate),
            "turn": self.cfg.realtime_turn or "server_vad",
            "language": self.cfg.language or "en",
        }
        if self.cfg.tts_voice:
            q["speaker"] = self.cfg.tts_voice
        return q

    def run(
        self,
        stop: threading.Event,
        *,
        device: str,
        player: Any | None = None,
        mic: Any | None = None,
        once: bool = False,
    ) -> None:
        if not self.base_url or not self.chat_model:
            raise WebSocketError("duplex needs a base url and a chat model")
        backoff = 0.5
        while not stop.is_set():
            try:
                self._once(stop, device=device, player=player, mic=mic, once=once)
                backoff = 0.5
                if once:
                    return
            except Exception as e:  # noqa: BLE001
                print(f"[ear] DUPLEX_DOWN {e}", flush=True)
            if stop.is_set():
                return
            print(f"[ear] DUPLEX reconnect in {backoff:.1f}s", flush=True)
            if stop.wait(backoff):
                return
            backoff = min(backoff * 2, 5.0)

    def _once(
        self,
        stop: threading.Event,
        *,
        device: str,
        player: Any | None,
        mic: Any | None,
        once: bool,
    ) -> None:
        query = self.query(16000)
        url = realtime_ws_url(self.base_url, query)
        print(
            f"[ear] DUPLEX_START {url.split('?', 1)[0]} model={self.chat_model} "
            f"voice={query['voice']} speaker={query.get('speaker', '')}",
            flush=True,
        )
        ws = WebSocket(url, timeout=min(20.0, self.timeout))
        owned_mic = mic is None
        owned_player = player is None
        if owned_mic:
            mic = LiveMic(device)
        if owned_player:
            player = PcmSpeaker(self.cfg.aplay_device)
        unsub = None
        try:
            ws.settimeout(None)
            opened = json.loads(ws.recv_text() or "{}")
            if opened.get("type") != "session.created":
                raise WebSocketError(str(opened.get("error") or opened.get("type") or "no session.created"))
            rate = _input_rate(opened)
            asr, asr_backend = _session_asr(opened)
            print(
                f"[ear] DUPLEX session input={rate}Hz asr={asr or '?'} "
                f"asr_backend={asr_backend or '?'}",
                flush=True,
            )
            if self.sight is not None:
                def _on_sight(text: str) -> None:
                    try:
                        ws.send_text(json.dumps(vision_context_item(text)))
                    except (OSError, WebSocketError):
                        return

                unsub = self.sight.subscribe(_on_sight)
            ws.send_text(json.dumps({
                "type": "session.update",
                "session": {
                    "instructions": self._instructions(),
                    "audio": {
                        "input": {
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": VAD_THRESHOLD,
                                "silence_duration_ms": VAD_SILENCE_MS,
                                "prefix_padding_ms": 300,
                            }
                        }
                    },
                },
            }))
            mic_gate = MicGate()
            threading.Thread(
                target=_pump_mic, args=(ws, mic, stop, mic_gate), name="bruhos-duplex-mic", daemon=True,
            ).start()
            say_parts: list[str] = []
            audio_bytes = 0
            heard = ""
            skipped = False
            while not stop.is_set():
                raw = ws.recv_text()
                if raw is None:
                    raise WebSocketError("socket closed")
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    print("[ear] EVT bad json", flush=True)
                    continue
                kind = str(event.get("type") or "")
                if kind == "response.output_audio.delta":
                    if skipped:
                        continue
                    pcm, play_rate = _pcm_delta(event, self.cfg.tts_rate)
                    if pcm and play_rate:
                        if audio_bytes == 0:
                            print(
                                f"[ear] AUDIO start rate={play_rate} bytes={len(pcm)}",
                                flush=True,
                            )
                        audio_bytes += len(pcm)
                        mic_gate.note_audio(len(pcm), play_rate)
                        player.write(pcm, play_rate)
                    continue
                if kind == "response.output_text.delta":
                    if not skipped:
                        say_parts.append(str(event.get("delta") or ""))
                    continue
                if kind == "error":
                    text = error_text(event)
                    print(f"[ear] DUPLEX error {text}", flush=True)
                    raise WebSocketError(text)
                if kind == "conversation.item.input_audio_transcription.completed":
                    heard = str(event.get("transcript") or "").strip()
                    asr, asr_backend = _session_asr(event)
                    if _no_words(heard):
                        skipped = True
                        print(
                            f"[ear] HEARD '' asr={asr or '?'} asr_backend={asr_backend or '?'}",
                            flush=True,
                        )
                    else:
                        note = ""
                        if asr_backend == "whisper_fallback":
                            note = " whisper_fallback"
                            print("[ear] ASR whisper_fallback", flush=True)
                        print(
                            f"[ear] HEARD {heard!r} asr={asr or '?'} "
                            f"asr_backend={asr_backend or '?'}{note}",
                            flush=True,
                        )
                elif kind == "response.created":
                    mic_gate.hold()
                    if _no_words(heard):
                        skipped = True
                        try:
                            ws.send_text(json.dumps({"type": "response.cancel"}))
                        except (OSError, WebSocketError):
                            pass
                        print("[ear] SKIP no words", flush=True)
                elif kind == "response.status" and _status_name(event) == "generating" and audio_bytes == 0:
                    mic_gate.hold()
                elif kind in ("response.cancel", "response.cancelled"):
                    player.flush()
                    mic_gate.drop(ECHO_TAIL_S)
                    if not skipped:
                        print("[ear] CANCEL barge-in", flush=True)
                elif kind == "response.done":
                    if _status_name(event) == "cancelled":
                        player.flush()
                        mic_gate.drop(ECHO_TAIL_S)
                        if not skipped:
                            print("[ear] CANCEL barge-in", flush=True)
                    else:
                        mic_gate.release(ECHO_TAIL_S)
                        finish = getattr(player, "release", None)
                        if finish is not None:
                            finish()
                    heard = ""
                    skipped = False
                elif kind == "response.status" and _status_name(event) == "silence_started":
                    audio_bytes = 0
                    say_parts.clear()
                _log_progress(kind, event, say_parts, audio_bytes)
                if kind == "response.done":
                    audio_bytes = 0
                    if once:
                        stop.set()
                        return
        finally:
            if unsub is not None:
                unsub()
            if owned_mic:
                mic.close()
            if owned_player:
                player.flush()
            ws.close()


class MicGate:
    """Uplink silence while a reply is playing, plus a short tail after it ends."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._quiet_until = 0.0
        self._play_until = 0.0
        self._holding = False
        self._live_announced = True

    def hold(self) -> None:
        with self._lock:
            self._quiet_until = time.monotonic() + 3600.0
            self._play_until = 0.0
            if self._holding:
                return
            self._holding = True
            self._live_announced = False
        print("[ear] MIC hold", flush=True)

    def note_audio(self, nbytes: int, rate: int) -> None:
        if nbytes <= 0 or rate <= 0:
            return
        dur = nbytes / (rate * 2)
        with self._lock:
            now = time.monotonic()
            self._play_until = max(self._play_until, now) + dur

    def release(self, pad: float) -> None:
        with self._lock:
            now = time.monotonic()
            self._quiet_until = max(self._play_until, now) + pad
            self._holding = False

    def drop(self, pad: float) -> None:
        with self._lock:
            now = time.monotonic()
            self._play_until = now
            self._quiet_until = now + pad
            self._holding = False

    def outbound(self, chunk: bytes) -> bytes:
        with self._lock:
            quiet = time.monotonic() < self._quiet_until
            announce = not quiet and not self._live_announced
            if announce:
                self._live_announced = True
        if announce:
            print("[ear] MIC live", flush=True)
        if quiet:
            return bytes(len(chunk))
        return chunk


def _pump_mic(ws: WebSocket, mic: Any, stop: threading.Event, gate: MicGate) -> None:
    """Append PCM for the life of the socket. Silence while a reply is on the speaker."""
    while not stop.is_set():
        try:
            chunk = mic.q.get(timeout=0.2)
        except queue.Empty:
            continue
        if chunk is None or stop.is_set():
            return
        payload = gate.outbound(chunk)
        try:
            ws.send_text(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(payload).decode(),
            }))
        except (OSError, WebSocketError):
            return


def _session_asr(event: dict) -> tuple[str, str]:
    """``(asr, asr_backend)`` from ``session.created`` or a transcription event."""
    session = event.get("session") if isinstance(event.get("session"), dict) else {}
    asr = str(event.get("asr") or session.get("asr") or "")
    backend = str(event.get("asr_backend") or session.get("asr_backend") or "")
    return asr, backend


def _input_rate(event: dict) -> int:
    try:
        return int(event["session"]["audio"]["input"]["format"]["rate"])
    except (KeyError, TypeError, ValueError):
        return 16000


def _pcm_delta(event: dict, default_rate: int = 22050) -> tuple[bytes, int]:
    raw = event.get("delta")
    if not isinstance(raw, str) or not raw:
        return b"", 0
    try:
        pcm = base64.b64decode(raw)
    except (ValueError, TypeError):
        return b"", 0
    try:
        rate = int(event.get("rate") or default_rate)
    except (TypeError, ValueError):
        rate = default_rate
    return pcm, rate


def play_pcm(pcm: bytes, rate: int, device: str | None = None) -> None:
    speaker = PcmSpeaker(device)
    speaker.write(pcm, rate)
    # Let aplay drain, then close stdin by killing after a short wait is wrong.
    # One-shot playback waits for the process.
    proc = speaker.proc
    if proc is None or proc.stdin is None:
        return
    proc.stdin.close()
    proc.wait(timeout=30)
