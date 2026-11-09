"""WonderEcho wake → record → STT → harness turn → TTS."""

from __future__ import annotations

import queue
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bruhos.listen import audio_debug as adbg
from bruhos.listen.chat import ChatReply
from bruhos.listen.conversation import Conversation
from bruhos.listen.normalize import normalize_file
from bruhos.listen.wonderecho import WonderEcho
from bruhos.speech import SpeechConfig, make_asr, make_tts


@dataclass
class EarConfig:
    port: str = "/dev/ttyUSB0"
    device: str = "plughw:Device,0"
    seconds: float = 5.0
    rate: int = 16000
    channels: int = 1
    ack: bool = True
    ack_text: str = "Yes?"
    sleep_text: str = "Okay, standing by."
    debug: bool = False
    keep_wav_dir: str = "/tmp/bruhos-ear-wav"
    # After a duplex reply, listen again without another wake word.
    follow_seconds: float = 0.0
    follow_turns: int = 0


def _phase(step: str, msg: str = "", *, t0: float | None = None) -> float:
    """Always-on lifecycle line for journalctl. Returns monotonic now."""
    now = time.monotonic()
    elapsed = f" +{now - t0:.2f}s" if t0 is not None else ""
    detail = f" {msg}" if msg else ""
    print(f"[ear] {step}{elapsed}{detail}", flush=True)
    return now


class _Mic:
    """arecord → queue. ``close`` kills the recorder so a pause ends the turn."""

    def __init__(self, device: str, seconds: float, rate: int, channels: int):
        arecord = shutil.which("arecord")
        if not arecord:
            raise RuntimeError("arecord not found")
        self.q: queue.Queue[bytes | None] = queue.Queue()
        cmd = [
            arecord, "-D", device, "-f", "S16_LE", "-r", str(rate),
            "-c", str(channels), "-t", "raw", "-d", str(max(1, int(round(seconds)))),
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        threading.Thread(target=self._pump, name="bruhos-mic", daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        try:
            while True:
                chunk = self.proc.stdout.read(3200)
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


def _record(device: str, seconds: float, rate: int, channels: int, *, debug: bool = False) -> Path:
    arecord = shutil.which("arecord")
    if not arecord:
        raise RuntimeError("arecord not found")
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    f.close()
    path = Path(f.name)
    cmd = [
        arecord,
        "-D", device,
        "-f", "S16_LE",
        "-r", str(rate),
        "-c", str(channels),
        "-d", str(max(1, int(round(seconds)))),
        str(path),
    ]
    if debug:
        adbg.log(f"arecord cmd: {' '.join(cmd)}")
    t0 = time.monotonic()
    r = subprocess.run(cmd, check=False, capture_output=True, text=True)
    dt = time.monotonic() - t0
    if debug:
        err = (r.stderr or "").strip()
        adbg.log(
            f"arecord exit={r.returncode} {dt:.2f}s stderr={err!r} "
            f"bytes={path.stat().st_size if path.exists() else 0}"
        )
    if r.returncode != 0:
        raise RuntimeError(f"arecord failed ({r.returncode}): {(r.stderr or '').strip()}")
    return path


class _SerialStop:
    """Poll WonderEcho for sleep while a skill runs (mid-gait abort)."""

    def __init__(self, echo: WonderEcho):
        self._echo = echo
        self._flag = threading.Event()

    def arm(self) -> None:
        self._flag.clear()

    def poll(self) -> None:
        ev = self._echo.read_event()
        if ev == "sleep":
            self._flag.set()

    def __call__(self) -> bool:
        self.poll()
        return self._flag.is_set()


def _hear_file(speech, cfg: EarConfig, asr, *, dbg: bool, keep_dir: Path | None, t0: float) -> str:
    _phase("RECORD_START", f"{cfg.seconds}s on {cfg.device}", t0=t0)
    t_rec = time.monotonic()
    wav = _record(cfg.device, cfg.seconds, cfg.rate, cfg.channels, debug=dbg)
    try:
        _phase("RECORD_DONE", f"{time.monotonic() - t_rec:.2f}s", t0=t0)
        if dbg:
            adbg.log_wav(wav, "raw")
            if keep_dir is not None:
                dest = keep_dir / f"raw-{int(time.time())}.wav"
                dest.write_bytes(wav.read_bytes())
                adbg.log(f"kept {dest}")
        _phase("NORM_START", "", t0=t0)
        normed = normalize_file(wav)
        norm = normed if normed != wav else None
        try:
            _phase("NORM_DONE", "", t0=t0)
            _phase("STT_START", f"model={getattr(asr, 'model', speech.asr_model)}", t0=t0)
            t_asr = time.monotonic()
            text = (asr.transcribe(str(normed)) or "").strip()
            _phase("STT_DONE", f"{time.monotonic() - t_asr:.2f}s", t0=t0)
            return text
        finally:
            if norm is not None:
                norm.unlink(missing_ok=True)
    finally:
        wav.unlink(missing_ok=True)


def _hear_duplex(speech, cfg: EarConfig, *, seconds: float, t0: float) -> tuple[str, bytes, int]:
    from bruhos.speech.realtime import RealtimeDuplex

    _phase(
        "DUPLEX_START",
        f"model={speech.chat_model} asr={speech.asr_model} voice={speech.tts_model} "
        f"max {seconds:.0f}s {cfg.rate}Hz",
        t0=t0,
    )
    duplex = RealtimeDuplex(speech, speech.chat_model)
    mic = _Mic(cfg.device, seconds, cfg.rate, cfg.channels)
    result = duplex.capture(mic, rate=cfg.rate, listen_s=seconds)
    if result.error and not result.transcript:
        raise RuntimeError(result.error)
    if result.error:
        _phase("DUPLEX_CHAT", result.error, t0=t0)
    _phase("STT_DONE", f"duplex {len(result.audio)}B pcm", t0=t0)
    return result.transcript, result.audio, result.sample_rate


def _converse(
    speech, cfg, asr, tts, chat, harness, hist, *,
    dbg: bool, keep_dir: Path | None, seconds: float, t0: float,
) -> bool:
    """Hear one utterance and speak the reply. False when nothing was said."""
    text = ""
    duplex_audio = b""
    duplex_rate = speech.tts_rate
    try:
        if speech.realtime == "duplex":
            text, duplex_audio, duplex_rate = _hear_duplex(speech, cfg, seconds=seconds, t0=t0)
        else:
            text = _hear_file(speech, cfg, asr, dbg=dbg, keep_dir=keep_dir, t0=t0)
    except Exception as e:  # noqa: BLE001
        _phase("STT_FAIL", str(e), t0=t0)
        if dbg:
            adbg.alsa_snapshot(card_hint="Device")
        return False

    if not text:
        _phase("HEARD", "(empty)", t0=t0)
        return False

    _phase("HEARD", repr(text), t0=t0)
    acted = False
    try:
        t_turn = time.monotonic()
        if harness is not None:
            turn = harness.turn(text, history=hist.snapshot())
            reply = turn.say
            acted = bool(getattr(turn, "acted", False))
        else:
            _phase("LLM_START", f"turns={hist.turns}", t0=t0)
            reply = chat.reply(text, history=hist.snapshot())
            _phase("LLM_DONE", f"{time.monotonic() - t_turn:.2f}s", t0=t0)
        if reply:
            hist.append(text, reply)
    except Exception as e:  # noqa: BLE001
        _phase("LLM_FAIL", str(e), t0=t0)
        return False

    _phase("REPLY", repr(reply), t0=t0)
    try:
        if duplex_audio and not acted:
            _phase("TTS_START", f"duplex pcm {duplex_rate}Hz {len(duplex_audio)}B", t0=t0)
            t_tts = time.monotonic()
            from bruhos.speech.realtime import play_pcm
            play_pcm(duplex_audio, duplex_rate, speech.aplay_device)
            _phase("TTS_DONE", f"{time.monotonic() - t_tts:.2f}s", t0=t0)
        else:
            _phase("TTS_START", f"playback={speech.playback}", t0=t0)
            t_tts = time.monotonic()
            tts.say(reply)
            _phase("TTS_DONE", f"{time.monotonic() - t_tts:.2f}s", t0=t0)
    except Exception as e:  # noqa: BLE001
        _phase("TTS_FAIL", str(e), t0=t0)
    return True


def run_duplex_ear(
    speech: SpeechConfig,
    ear: EarConfig | None = None,
    *,
    once: bool = False,
    sight: Any | None = None,
) -> None:
    """Wake word, then one duplex socket until the sleep word.

    The server transcribes, replies, and speaks. This process only streams
    the mic and plays PCM. It does not open the robot board.
    """
    from bruhos.speech.realtime import DuplexSession

    cfg = ear or EarConfig()
    echo = WonderEcho(cfg.port, wait_s=120.0, debug=bool(cfg.debug))
    _phase("IDLE", f"duplex waiting for wake on {cfg.port}")
    try:
        while True:
            ev = echo.read_event()
            if ev is None:
                time.sleep(0.02)
                continue
            if ev == "sleep":
                continue
            if ev != "wakeup":
                continue
            cycle = _phase("TRIGGER", "wakeup")
            stop = threading.Event()

            def _watch() -> None:
                while not stop.is_set():
                    got = echo.read_event()
                    if got == "sleep":
                        _phase("TRIGGER", "sleep", t0=cycle)
                        stop.set()
                        return
                    time.sleep(0.02)

            watcher = threading.Thread(target=_watch, name="bruhos-sleep", daemon=True)
            watcher.start()
            try:
                DuplexSession(speech, sight=sight).run(
                    stop, device=cfg.device, once=once,
                )
            except Exception as e:  # noqa: BLE001
                _phase("DUPLEX_FAIL", str(e), t0=cycle)
            stop.set()
            watcher.join(timeout=1)
            _phase("IDLE", "waiting for next wake", t0=cycle)
            if once:
                return
    finally:
        echo.close()
        _phase("STOP", "serial closed")


def run_ear(
    speech: SpeechConfig,
    chat: ChatReply,
    ear: EarConfig | None = None,
    conversation: Conversation | None = None,
    *,
    harness: Any | None = None,
    once: bool = False,
) -> None:
    """Block on WonderEcho wake frames; after each wake, record and reply.

    ``once``: process a single wake→reply cycle then return (for smoke tests).
    When ``harness`` is set, the transcript goes through the agent tool loop
    (skills run before TTS). Otherwise falls back to ChatReply only.
    """
    cfg = ear or EarConfig()
    hist = conversation or Conversation(max_turns=16)
    dbg = bool(cfg.debug)
    keep_dir = Path(cfg.keep_wav_dir) if dbg else None
    if keep_dir is not None:
        keep_dir.mkdir(parents=True, exist_ok=True)

    _phase("BOOT", f"port={cfg.port} device={cfg.device} record={cfg.seconds}s debug={dbg}")
    if dbg:
        adbg.alsa_snapshot(card_hint="Device")
        adbg.log(
            f"speech asr={speech.asr} tts={speech.tts} playback={speech.playback} "
            f"aplay_device={speech.aplay_device!r} base_url={speech.base_url!r}"
        )

    echo = WonderEcho(cfg.port, wait_s=120.0, debug=dbg)
    stopper = _SerialStop(echo)
    # Wire mid-gait abort into the hub if the harness exposes one
    hub = getattr(harness, "hub", None)
    if hub is not None:
        hub.should_stop = stopper

    tts = make_tts(speech)
    asr = make_asr(speech)
    _phase("IDLE", f"waiting for wake on {cfg.port} harness={'on' if harness else 'chat'}")
    try:
        while True:
            ev = echo.read_event()
            if ev is None:
                time.sleep(0.02)
                continue
            if ev == "sleep":
                cycle = _phase("TRIGGER", "sleep")
                stopper.arm()
                stopper._flag.set()
                if cfg.ack:
                    _phase("TTS_START", f"sleep ack {cfg.sleep_text!r}", t0=cycle)
                    try:
                        tts.say(cfg.sleep_text)
                        _phase("TTS_DONE", "sleep ack", t0=cycle)
                    except Exception as e:  # noqa: BLE001
                        _phase("TTS_FAIL", str(e), t0=cycle)
                _phase("IDLE", "back to waiting", t0=cycle)
                if once:
                    return
                continue
            if ev != "wakeup":
                if dbg:
                    adbg.log(f"ignore serial event={ev}")
                continue

            cycle = _phase("TRIGGER", "wakeup")
            stopper.arm()
            if dbg:
                adbg.alsa_snapshot(card_hint="Device")

            # A spoken "Yes?" is a full TTS round trip before the mic opens.
            # Duplex starts listening on the wake word instead.
            if cfg.ack and speech.realtime != "duplex":
                _phase("ACK_START", repr(cfg.ack_text), t0=cycle)
                try:
                    tts.say(cfg.ack_text)
                    _phase("ACK_DONE", "", t0=cycle)
                except Exception as e:  # noqa: BLE001
                    _phase("ACK_FAIL", str(e), t0=cycle)

            extra = cfg.follow_turns if speech.realtime == "duplex" and cfg.follow_seconds > 0 else 0
            turns = 1 + extra
            for n in range(turns):
                limit = cfg.seconds if n == 0 else cfg.follow_seconds
                if n:
                    _phase("FOLLOW", f"listening {limit:.0f}s", t0=cycle)
                if not _converse(
                    speech, cfg, asr, tts, chat, harness, hist,
                    dbg=dbg, keep_dir=keep_dir, seconds=limit, t0=cycle,
                ):
                    break

            _phase("IDLE", "waiting for next wake", t0=cycle)
            if once:
                return
    finally:
        echo.close()
        _phase("STOP", "serial closed")
