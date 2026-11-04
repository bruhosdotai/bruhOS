"""WonderEcho wake → record → STT → harness turn → TTS."""

from __future__ import annotations

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


def _phase(step: str, msg: str = "", *, t0: float | None = None) -> float:
    """Always-on lifecycle line for journalctl. Returns monotonic now."""
    now = time.monotonic()
    elapsed = f" +{now - t0:.2f}s" if t0 is not None else ""
    detail = f" {msg}" if msg else ""
    print(f"[ear] {step}{elapsed}{detail}", flush=True)
    return now


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

            if cfg.ack:
                _phase("ACK_START", repr(cfg.ack_text), t0=cycle)
                try:
                    tts.say(cfg.ack_text)
                    _phase("ACK_DONE", "", t0=cycle)
                except Exception as e:  # noqa: BLE001
                    _phase("ACK_FAIL", str(e), t0=cycle)

            wav: Path | None = None
            norm: Path | None = None
            text = ""
            try:
                _phase("RECORD_START", f"{cfg.seconds}s on {cfg.device}", t0=cycle)
                t_rec = time.monotonic()
                wav = _record(cfg.device, cfg.seconds, cfg.rate, cfg.channels, debug=dbg)
                _phase("RECORD_DONE", f"{time.monotonic() - t_rec:.2f}s", t0=cycle)
                if dbg:
                    adbg.log_wav(wav, "raw")
                    if keep_dir is not None:
                        dest = keep_dir / f"raw-{int(time.time())}.wav"
                        dest.write_bytes(wav.read_bytes())
                        adbg.log(f"kept {dest}")

                _phase("NORM_START", "", t0=cycle)
                normed = normalize_file(wav)
                if normed != wav:
                    norm = normed
                    if dbg:
                        adbg.log_wav(normed, "norm")
                _phase("NORM_DONE", "", t0=cycle)

                _phase("STT_START", f"model={getattr(asr, 'model', speech.asr_model)}", t0=cycle)
                t_asr = time.monotonic()
                text = (asr.transcribe(str(normed)) or "").strip()
                _phase("STT_DONE", f"{time.monotonic() - t_asr:.2f}s", t0=cycle)
            except Exception as e:  # noqa: BLE001
                _phase("STT_FAIL", str(e), t0=cycle)
                if dbg:
                    adbg.alsa_snapshot(card_hint="Device")
                text = ""
            finally:
                if wav is not None:
                    wav.unlink(missing_ok=True)
                if norm is not None:
                    norm.unlink(missing_ok=True)

            if not text:
                _phase("HEARD", "(empty)", t0=cycle)
                _phase("IDLE", "no speech — back to waiting", t0=cycle)
                if once:
                    return
                continue

            _phase("HEARD", repr(text), t0=cycle)

            try:
                t_turn = time.monotonic()
                if harness is not None:
                    turn = harness.turn(text, history=hist.snapshot())
                    reply = turn.say
                else:
                    _phase("LLM_START", f"turns={hist.turns}", t0=cycle)
                    reply = chat.reply(text, history=hist.snapshot())
                    _phase("LLM_DONE", f"{time.monotonic() - t_turn:.2f}s", t0=cycle)
                if reply:
                    hist.append(text, reply)
            except Exception as e:  # noqa: BLE001
                _phase("LLM_FAIL", str(e), t0=cycle)
                _phase("IDLE", "turn failed — back to waiting", t0=cycle)
                if once:
                    return
                continue

            _phase("REPLY", repr(reply), t0=cycle)
            try:
                _phase("TTS_START", f"playback={speech.playback}", t0=cycle)
                t_tts = time.monotonic()
                tts.say(reply)
                _phase("TTS_DONE", f"{time.monotonic() - t_tts:.2f}s", t0=cycle)
            except Exception as e:  # noqa: BLE001
                _phase("TTS_FAIL", str(e), t0=cycle)

            _phase("IDLE", "cycle complete — waiting for next wake", t0=cycle)
            if once:
                return
    finally:
        echo.close()
        _phase("STOP", "serial closed")
