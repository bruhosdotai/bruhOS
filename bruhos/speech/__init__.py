"""English-first speech. ASR language, TTS voice and planner prompts default to English."""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

from .base import DEFAULT_LANGUAGE, ASR, SpeechConfig, TTS

__all__ = [
    "DEFAULT_LANGUAGE",
    "ASR",
    "TTS",
    "SpeechConfig",
    "WhisperASR",
    "EspeakTTS",
    "PrintTTS",
    "OpenAICompatASR",
    "OpenAICompatTTS",
    "make_asr",
    "make_tts",
]


class WhisperASR(ASR):
    """faster-whisper on CPU (int8). ``base.en`` runs on a Raspberry Pi 5 for short commands."""

    def __init__(self, cfg: SpeechConfig | None = None):
        from faster_whisper import WhisperModel  # optional extra: pip install bruhos[speech]

        self.cfg = cfg or SpeechConfig()
        self.model = WhisperModel(self.cfg.asr_model, device="cpu", compute_type="int8")

    def transcribe(self, wav_path: str) -> str:
        segments, _ = self.model.transcribe(
            wav_path, language=self.cfg.language, beam_size=1, vad_filter=True
        )
        return " ".join(s.text.strip() for s in segments).strip()


class EspeakTTS(TTS):
    def __init__(self, cfg: SpeechConfig | None = None):
        self.cfg = cfg or SpeechConfig()
        self.bin = shutil.which("espeak-ng") or shutil.which("espeak")
        if not self.bin:
            raise RuntimeError("espeak-ng not found (apt install espeak-ng)")

    def say(self, text: str) -> None:
        subprocess.run([self.bin, "-v", self.cfg.tts_voice, text], check=False)


class PrintTTS(TTS):
    def say(self, text: str) -> None:
        print(f"[bruh] {text}")


def make_asr(cfg: SpeechConfig | dict[str, Any] | None = None) -> ASR:
    c = cfg if isinstance(cfg, SpeechConfig) else SpeechConfig(**(cfg or {}))
    if c.asr in ("openai", "remote", "zerollama", "ollama"):
        from .openai_compat import OpenAICompatASR
        return OpenAICompatASR(c)
    if c.asr == "local":
        return WhisperASR(c)
    raise ValueError(f"unknown speech.asr {c.asr!r} (local | openai)")


def make_tts(cfg: SpeechConfig | dict[str, Any] | None = None) -> TTS:
    c = cfg if isinstance(cfg, SpeechConfig) else SpeechConfig(**(cfg or {}))
    if c.tts in ("openai", "remote", "zerollama", "ollama"):
        from .openai_compat import OpenAICompatTTS
        return OpenAICompatTTS(c)
    if c.tts == "print":
        return PrintTTS()
    if c.tts == "piper":
        from .piper import PiperTTS
        return PiperTTS(c)
    if c.tts == "local":
        return EspeakTTS(c)
    raise ValueError(f"unknown speech.tts {c.tts!r} (local | piper | openai | print)")


# re-export remote classes lazily-friendly
def __getattr__(name: str):
    if name in ("OpenAICompatASR", "OpenAICompatTTS"):
        from . import openai_compat
        return getattr(openai_compat, name)
    raise AttributeError(name)
