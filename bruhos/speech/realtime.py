"""Hearing: one duplex WebSocket on zerollama.

Owns ``qwen2.5:14b`` only. Mic PCM16 LE mono at 16 kHz goes out as
``input_audio_buffer.append``. With ``turn=server_vad`` this client never
sends ``input_audio_buffer.commit`` or ``response.create``. It does not call
``/api/chat``, ``/v1/chat/completions``, or send images. Vision runs in
``bruhos-eye`` on the other GPU.

``conversation.item.input_audio_transcription.completed`` is the heard text.
The field is ``transcript``. An empty transcript means no words: stay on the
mic and do not play a reply. ``asr_backend=qwen3-asr`` means Qwen heard it.
``asr_backend=whisper_fallback`` means Whisper heard the clip because Qwen
was down — keep the text and log it.
Play ``response.output_audio.delta`` after ``response.created``. A reply with
``audio_bytes`` 0 is a failed voice turn, not a transcript.
Keep appending real PCM through ``silence_started`` until
``input_audio_buffer.speech_stopped``. That first quiet dip is not the end of
the sentence, and this client never commits. At ``speech_stopped`` send
nothing until the reply has finished playing, plus a short tail. Audio that
keeps arriving after the turn closes starts a new one and the transcript
never lands. A loud chunk while the speaker is playing is the same problem.
"""

from __future__ import annotations

import array
import base64
import json
import queue
import shutil
import subprocess
import threading
import time
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

from bruhos.speech.base import SpeechConfig, clean_transcript
from bruhos.speech.ws import WebSocket, WebSocketError

# Asked once after a live socket drops, so the person can repeat the lost sentence.
PARDON = {
    "en": "Pardon?",
    "zh": "什么？",
    "ms": "Maaf?",
    "th": "อะไรนะ?",
    "vi": "Xin lỗi?",
}

# Spoken style for qwen2.5:14b. A loose prompt plus the 48-token cap
# collapses into mixed characters, and that reply stays in the next turns.
_SPOKEN = (
    "You are Tony. Answer the person out loud in one or two short sentences of plain English.\n"
    "Use everyday words. No Chinese, no lists, no symbols, no stage directions.\n"
    "If you do not know, say so in one sentence."
)
_CAMERA_RULE = (
    "Mention the camera only when they ask what you see or what something looks like."
)
# 30 ms at 16 kHz mono PCM16.
CHUNK_BYTES = 960
# After aplay exits, the speaker is still in the room for a moment.
# A chunk louder than the room during playback cancels the reply.
ECHO_TAIL_S = 0.20
# The server drops the quiet part before recognition. The wait is how long
# a dip can last before the turn closes. 700 ms was closing on a breath,
# and the mic then stops, so the last words were never sent.
# Do not go below 700 ms.
VAD_THRESHOLD = 0.04
VAD_SILENCE_MS = 2000
VAD_SILENCE_MIN_MS = 700
# Used by tests that check a quiet voice can be lifted. The live uplink does
# not use it: a boosted speaker tail is louder than the room and becomes
# the next thing the person "said."
UPLINK_TARGET_PEAK = 0.10
UPLINK_GAIN_MAX = 4.0


def _pcm_peak(pcm: bytes) -> float:
    """Peak amplitude of PCM16 LE, from 0 to 1."""
    count = len(pcm) // 2
    if count == 0:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm[: count * 2])
    return max(abs(sample) for sample in samples) / 32768.0


def _apply_gain(pcm: bytes, gain: float) -> bytes:
    count = len(pcm) // 2
    if count == 0 or gain == 1.0:
        return pcm
    samples = array.array("h")
    samples.frombytes(pcm[: count * 2])
    for index, sample in enumerate(samples):
        value = int(sample * gain)
        if value > 32767:
            value = 32767
        elif value < -32768:
            value = -32768
        samples[index] = value
    return samples.tobytes()


class SpeechBoost:
    """Lift a quiet voice above the server gate without lifting the room."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._noise = 0.01
        self._peak = 0.0

    def _note(self, pcm: bytes) -> tuple[float, float]:
        peak = _pcm_peak(pcm)
        with self._lock:
            if peak < self._noise:
                self._noise = 0.8 * self._noise + 0.2 * peak
            else:
                self._noise += (peak - self._noise) * 0.001
            if self._noise < 0.004:
                self._noise = 0.004
            self._peak = max(self._peak, peak)
            return peak, self._noise * 2.5

    def observe(self, pcm: bytes) -> None:
        """Remember the level. The uplink keeps the microphone's own gain."""
        self._note(pcm)

    def apply(self, pcm: bytes) -> bytes:
        peak, gate = self._note(pcm)
        if peak < gate or peak >= UPLINK_TARGET_PEAK:
            return pcm
        gain = min(UPLINK_GAIN_MAX, UPLINK_TARGET_PEAK / peak)
        if gain <= 1.05:
            return pcm
        return _apply_gain(pcm, gain)

    def levels(self) -> tuple[float, float]:
        """Peak since the last call, and the current noise estimate."""
        with self._lock:
            peak = self._peak
            self._peak = 0.0
            return peak, self._noise


def pardon_line(language: str) -> str:
    """Short prompt spoken after a dropped duplex socket."""
    return PARDON.get(language or "en", PARDON["en"])


def _play_wav(wav: bytes, device: str | None) -> None:
    player = shutil.which("aplay")
    if not player or not wav:
        return
    cmd = [player, "-q"]
    if device:
        cmd[1:1] = ["-D", device]
    subprocess.run(
        cmd, input=wav, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )


def _reply_collapsed(text: str) -> bool:
    """True when a spoken reply left English and should not be played or kept.

    qwen2.5:14b at temperature 0.8 sometimes spends its 48 tokens on the same
    mix of Chinese, digits, and fragments. The prompt does not stop that, and
    the socket keeps the reply for the next turns.
    """
    for ch in text:
        code = ord(ch)
        if (
            0x3400 <= code <= 0x9FFF
            or 0x3040 <= code <= 0x30FF
            or 0xAC00 <= code <= 0xD7AF
        ):
            return True
    folded = text.lower()
    return any(mark in folded for mark in ("openh", "agged", "mailto"))


def _no_words(text: str) -> bool:
    """True when a clip has nothing the model should answer."""
    cleaned = " ".join(str(text or "").split()).strip().lower()
    if cleaned in {"", "[blank_audio]", ".", "..", "...", "?", "!"}:
        return True
    # Punctuation-only crumbs from noisy VAD closes.
    if all(ch in ".,!?;:'\"-–—" for ch in cleaned):
        return True
    return False


def _plain_caption(text: str, *, max_chars: int = 180) -> str:
    """One plain English sentence. Chinese, JSON, and OCR junk are dropped.

    A caption in that style gets continued by the voice model.
    """
    raw = " ".join(str(text or "").split()).strip()
    if not raw or raw[0] in "{[" or any(ch in raw for ch in "{}[]"):
        return ""
    kept: list[str] = []
    for ch in raw:
        code = ord(ch)
        if (
            0x3400 <= code <= 0x9FFF
            or 0x3040 <= code <= 0x30FF
            or 0xAC00 <= code <= 0xD7AF
        ):
            continue
        if code < 128 and (ch.isalnum() or ch in " .,!?'-"):
            kept.append(ch)
    cleaned = " ".join("".join(kept).split()).strip(" -")
    if not cleaned:
        return ""
    buf = ""
    for ch in cleaned:
        buf += ch
        if ch in ".!?" and len(buf.strip()) > 8:
            break
    out = buf.strip()
    if len(out) > max_chars:
        out = out[: max_chars - 3].rstrip() + "..."
    words = [word for word in out.replace(".", " ").split() if any(c.isalpha() for c in word)]
    letters = sum(c.isalpha() for c in out)
    digits = sum(c.isdigit() for c in out)
    if len(words) < 2 or letters < 8 or digits > letters:
        return ""
    if not out.endswith((".", "!", "?")):
        out += "."
    return out


def build_instructions(caption: str = "", *, when: float | None = None, language: str = "en") -> str:
    """Whole ``session.update`` instructions string. Camera changes replace it."""
    del when, language
    lines = [_SPOKEN]
    seen = _plain_caption(caption)
    if seen:
        lines.append(f"The camera shows: {seen}")
    lines.append(_CAMERA_RULE)
    return "\n".join(lines)


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
    """Stream PCM16 mono to aplay without blocking the hearing socket.

    ``write`` only queues. A stuck speaker must not stop the mic uplink.
    ``release`` finishes the queued audio and signals when aplay has exited.
    ``flush`` kills playback for barge-in.
    """

    def __init__(self, device: str | None = None):
        self.device = device
        self._q: queue.Queue[tuple] = queue.Queue()
        self._proc: subprocess.Popen[bytes] | None = None
        self._rate: int | None = None
        self._proc_lock = threading.Lock()
        self._gen = 0
        threading.Thread(target=self._loop, name="bruhos-aplay", daemon=True).start()

    def write(self, pcm: bytes, rate: int) -> None:
        if not pcm:
            return
        self._q.put(("pcm", pcm, rate, self._gen))

    def flush(self) -> None:
        with self._proc_lock:
            self._gen += 1
        self._kill()
        self._q.put(("kill", b"", 0, None))

    def release(self) -> threading.Event:
        """Play what is already queued, then close aplay. The event fires on exit."""
        done = threading.Event()
        self._q.put(("eof", b"", 0, done))
        return done

    def _loop(self) -> None:
        while True:
            kind, pcm, rate, extra = self._q.get()
            if kind == "kill":
                self._kill()
                continue
            if kind == "eof":
                self._close_and_wait()
                if isinstance(extra, threading.Event):
                    extra.set()
                continue
            if extra != self._gen:
                continue
            self._write(pcm, rate)

    def _write(self, pcm: bytes, rate: int) -> None:
        with self._proc_lock:
            proc = self._proc
            if proc is None or self._rate != rate or proc.poll() is not None:
                self._kill_locked()
                player = shutil.which("aplay")
                if not player:
                    return
                cmd = [player, "-f", "S16_LE", "-r", str(rate), "-c", "1", "-t", "raw", "-q"]
                if self.device:
                    cmd[1:1] = ["-D", self.device]
                self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                self._rate = rate
                proc = self._proc
            stdin = proc.stdin
        if stdin is None:
            return
        try:
            stdin.write(pcm)
            stdin.flush()
        except (BrokenPipeError, OSError):
            self._kill()

    def _close_and_wait(self) -> None:
        with self._proc_lock:
            proc = self._proc
            self._proc = None
            self._rate = None
        if proc is None:
            return
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass

    def _kill(self) -> None:
        with self._proc_lock:
            self._kill_locked()

    def _kill_locked(self) -> None:
        proc = self._proc
        self._proc = None
        self._rate = None
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
                # A stalled uplink must not replay a pile of old audio later.
                if self.q.qsize() > 50:
                    dropped = 0
                    while self.q.qsize() > 8:
                        try:
                            self.q.get_nowait()
                        except queue.Empty:
                            break
                        dropped += 1
                    if dropped:
                        print(f"[ear] MIC backlog drop {dropped}", flush=True)
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
        self._live = False
        self._owe_pardon = False
        self._pardon_wav: bytes | None = None

    def _instructions(self, caption: str | None = None, *, when: float | None = None) -> str:
        seen = caption
        if seen is None and self.sight is not None:
            seen = str(getattr(self.sight, "text", "") or "")
        return build_instructions(seen or "", when=when, language=self.cfg.language or "en")

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

    def _say_pardon(self, gate: MicGate) -> None:
        """Ask them to repeat. The mic stays shut so the word is not a new turn."""
        line = pardon_line(self.cfg.language)
        gate.hold()
        try:
            wav = self._pardon_wav or self._fetch_pardon(line)
            if not wav:
                print(f"[ear] PARDON skipped {line!r}", flush=True)
                return
            self._pardon_wav = wav
            print(f"[ear] PARDON {line!r}", flush=True)
            _play_wav(wav, self.cfg.aplay_device)
        finally:
            gate.drop(ECHO_TAIL_S)

    def _fetch_pardon(self, line: str) -> bytes | None:
        from dataclasses import replace

        from bruhos.speech.openai_compat import OpenAICompatTTS

        cfg = replace(self.cfg, timeout=min(8.0, float(self.cfg.timeout or 8.0)))
        try:
            return OpenAICompatTTS(cfg).synthesize(line)
        except Exception as e:  # noqa: BLE001
            print(f"[ear] PARDON fetch fail {e}", flush=True)
            return None

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
                if self._live and not stop.is_set():
                    self._owe_pardon = True
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
            f"voice={query['voice']} speaker={query.get('speaker', '')} "
            f"language={query.get('language', '')}",
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
            self._live = True
            rate = _input_rate(opened)
            asr, asr_backend = _session_asr(opened)
            print(
                f"[ear] DUPLEX session input={rate}Hz asr={asr or '?'} "
                f"asr_backend={asr_backend or '?'}",
                flush=True,
            )
            last_caption = _plain_caption(
                str(getattr(self.sight, "text", "") or "") if self.sight is not None else ""
            )
            if last_caption:
                print(f"[ear] SEE instructions {last_caption!r}", flush=True)
            if self.sight is not None:
                def _on_sight(text: str) -> None:
                    nonlocal last_caption
                    caption = _plain_caption(text)
                    if not caption or caption == last_caption:
                        return
                    last_caption = caption
                    # Only instructions — do not use conversation.item.create.
                    # A reply already in flight keeps its old copy; the next one uses this.
                    try:
                        ws.send_text(json.dumps({
                            "type": "session.update",
                            "session": {"instructions": self._instructions(caption)},
                        }))
                    except (OSError, WebSocketError):
                        return
                    print(f"[ear] SEE session.update {caption!r}", flush=True)

                unsub = self.sight.subscribe(_on_sight)
            ws.send_text(json.dumps({
                "type": "session.update",
                "session": {
                    "instructions": self._instructions(last_caption),
                    "audio": {
                        "input": {
                            "transcription": {"language": self.cfg.language or "en"},
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": VAD_THRESHOLD,
                                "silence_duration_ms": max(VAD_SILENCE_MS, VAD_SILENCE_MIN_MS),
                                "prefix_padding_ms": 300,
                            }
                        }
                    },
                },
            }))
            mic_gate = MicGate()
            boost = SpeechBoost()
            owe_pardon = self._owe_pardon and not stop.is_set()
            if owe_pardon:
                self._owe_pardon = False
                mic_gate.hold()
            threading.Thread(
                target=_pump_mic,
                args=(ws, mic, stop, mic_gate, boost),
                name="bruhos-duplex-mic",
                daemon=True,
            ).start()
            if owe_pardon:
                self._say_pardon(mic_gate)
            say_parts: list[str] = []
            audio_bytes = 0
            heard = ""
            skipped = False
            heard_empty = False
            reply_open = False

            def _cancel_reply() -> None:
                try:
                    ws.send_text(json.dumps({"type": "response.cancel"}))
                except (OSError, WebSocketError):
                    return

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
                            # The speaker is on. A live mic hears Tony and the
                            # server treats that as the person interrupting.
                            mic_gate.hold()
                            print(
                                f"[ear] AUDIO start rate={play_rate} bytes={len(pcm)}",
                                flush=True,
                            )
                        audio_bytes += len(pcm)
                        mic_gate.note_audio(len(pcm), play_rate)
                        player.write(pcm, play_rate)
                    continue
                if kind == "response.output_text.delta":
                    if skipped:
                        continue
                    say_parts.append(str(event.get("delta") or ""))
                    if _reply_collapsed("".join(say_parts)):
                        skipped = True
                        player.flush()
                        _cancel_reply()
                        print(f"[ear] SAY drop {''.join(say_parts)!r}", flush=True)
                        raise WebSocketError("reply left English")
                    continue
                if kind == "error":
                    text = error_text(event)
                    print(f"[ear] DUPLEX error {text}", flush=True)
                    raise WebSocketError(text)
                if kind == "conversation.item.input_audio_transcription.completed":
                    heard = clean_transcript(str(event.get("transcript") or ""))
                    asr, asr_backend = _session_asr(event)
                    if _no_words(heard):
                        # No words. Stay on the mic and do not speak a reply.
                        # A blank clip with no reply open must not mute the next one.
                        heard_empty = True
                        mic_gate.drop(0)
                        player.flush()
                        if reply_open:
                            skipped = True
                            _cancel_reply()
                        else:
                            skipped = False
                        print(
                            f"[ear] HEARD '' asr={asr or '?'} asr_backend={asr_backend or '?'}",
                            flush=True,
                        )
                    else:
                        heard_empty = False
                        if not reply_open:
                            skipped = False
                        note = ""
                        if asr_backend == "whisper_fallback":
                            note = " whisper_fallback"
                            print("[ear] ASR whisper_fallback", flush=True)
                        print(
                            f"[ear] HEARD {heard!r} asr={asr or '?'} "
                            f"asr_backend={asr_backend or '?'}{note}",
                            flush=True,
                        )
                elif kind == "input_audio_buffer.speech_stopped":
                    # The server closed this turn and started ASR. More live
                    # audio now is a new turn, and the transcript never arrives.
                    peak, noise = boost.levels()
                    print(f"[ear] MIC level peak={peak:.3f} noise={noise:.3f}", flush=True)
                    if not heard_empty:
                        mic_gate.hold()
                elif kind == "response.created":
                    reply_open = True
                    audio_bytes = 0
                    say_parts.clear()
                    if heard_empty:
                        skipped = True
                        mic_gate.drop(0)
                        _cancel_reply()
                        print("[ear] SKIP no words", flush=True)
                    else:
                        skipped = False
                elif kind in ("response.cancel", "response.cancelled"):
                    player.flush()
                    mic_gate.drop(ECHO_TAIL_S)
                    if not skipped:
                        print("[ear] CANCEL barge-in", flush=True)
                elif kind == "response.done":
                    if _status_name(event) == "cancelled":
                        player.flush()
                        mic_gate.drop(0 if skipped else ECHO_TAIL_S)
                        if not skipped:
                            print("[ear] CANCEL barge-in", flush=True)
                    else:
                        if audio_bytes == 0 and not skipped:
                            print("[ear] VOICE fail audio_bytes=0", flush=True)
                        _open_mic_after_playback(player, mic_gate, ECHO_TAIL_S)
                    heard = ""
                    skipped = False
                    heard_empty = False
                    reply_open = False
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


def _open_mic_after_playback(player: Any, gate: "MicGate", pad: float) -> None:
    """Open the mic when the speaker has actually finished, not when the socket says done."""
    finish = getattr(player, "release", None)
    done = finish() if finish is not None else None
    if not isinstance(done, threading.Event):
        gate.release(pad)
        return
    token = gate.begin_playback_wait()

    def _wait() -> None:
        done.wait(timeout=30)
        gate.finish_playback(token, pad)

    threading.Thread(target=_wait, name="bruhos-mic-open", daemon=True).start()


class MicGate:
    """Send nothing while a reply is playing, plus a short tail after it ends."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._quiet_until = 0.0
        self._play_until = 0.0
        self._holding = False
        self._live_announced = True
        self._open_token: object = object()
        self._held_at = 0.0

    def hold(self) -> None:
        with self._lock:
            self._open_token = object()
            self._quiet_until = time.monotonic() + 3600.0
            self._play_until = 0.0
            self._held_at = time.monotonic()
            if self._holding:
                return
            self._holding = True
            self._live_announced = False
        print("[ear] MIC hold", flush=True)

    def begin_playback_wait(self) -> object:
        token = object()
        with self._lock:
            self._open_token = token
        return token

    def finish_playback(self, token: object, pad: float) -> None:
        with self._lock:
            if token != self._open_token:
                return
            now = time.monotonic()
            self._quiet_until = now + pad
            self._play_until = now
            self._holding = False

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
            self._open_token = object()
            self._play_until = now
            self._quiet_until = now + pad
            self._holding = False

    def outbound(self, chunk: bytes) -> bytes | None:
        """The live chunk, or None while the speaker owns the room."""
        with self._lock:
            quiet = time.monotonic() < self._quiet_until
            announce = not quiet and not self._live_announced
            held = self._held_at
            if announce:
                self._live_announced = True
        if announce:
            waited = time.monotonic() - held if held else 0.0
            print(f"[ear] MIC live held={waited:.1f}s", flush=True)
        if quiet:
            return None
        return chunk


def _pump_mic(
    ws: WebSocket,
    mic: Any,
    stop: threading.Event,
    gate: MicGate,
    boost: SpeechBoost | None = None,
) -> None:
    """Append real PCM until the speaker starts. Send nothing while it plays."""
    while not stop.is_set():
        try:
            chunk = mic.q.get(timeout=0.2)
        except queue.Empty:
            continue
        if chunk is None or stop.is_set():
            return
        if boost is not None:
            boost.observe(chunk)
        payload = gate.outbound(chunk)
        if payload is None:
            continue
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
    speaker.release().wait(timeout=30)
