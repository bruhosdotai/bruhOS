"""Voice ingress helpers (normalize + STT). Address gating is roadmap / open-mic."""

from __future__ import annotations

from pathlib import Path

from bruhos.speech import SpeechConfig, make_asr

from .chat import ChatReply, DEFAULT_SYSTEM, clean_reply
from .conversation import Conversation
from .ear import EarConfig, run_ear
from .normalize import NormalizeError, is_wav, normalize_file, normalize_to_wav, require_ffmpeg
from .wonderecho import WonderEcho

__all__ = [
    "ChatReply",
    "Conversation",
    "DEFAULT_SYSTEM",
    "EarConfig",
    "WonderEcho",
    "clean_reply",
    "NormalizeError",
    "is_wav",
    "normalize_file",
    "normalize_to_wav",
    "require_ffmpeg",
    "run_ear",
    "transcribe_audio",
]


def transcribe_audio(data: bytes, cfg: SpeechConfig, *, suffix: str = ".webm") -> str:
    """Normalize bytes if needed, run configured ASR, clean up temps."""
    wav: Path | None = None
    owned = False
    try:
        if is_wav(data) and suffix.lower() in (".wav", ".wave"):
            wav = normalize_to_wav(data, suffix=".wav")  # still write temp for ASR path API
            owned = True
        else:
            wav = normalize_to_wav(data, suffix=suffix)
            owned = True
        return make_asr(cfg).transcribe(str(wav))
    finally:
        if owned and wav is not None:
            wav.unlink(missing_ok=True)
