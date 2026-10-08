"""English-first speech. ASR language, TTS voice and planner prompts default to English."""

from __future__ import annotations

import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass

DEFAULT_LANGUAGE = "en"


@dataclass
class SpeechConfig:
    language: str = DEFAULT_LANGUAGE
    asr_model: str = "base.en"          # faster-whisper English-only checkpoint
    tts_voice: str = "en-us"
    allow_non_english: bool = False

    def __post_init__(self) -> None:
        if self.language != "en" and not self.allow_non_english:
            raise ValueError(f"bruhOS is English-first; language={self.language!r} needs allow_non_english=True")


class ASR(ABC):
    @abstractmethod
    def transcribe(self, wav_path: str) -> str: ...


class TTS(ABC):
    @abstractmethod
    def say(self, text: str) -> None: ...


class WhisperASR(ASR):
    """faster-whisper on CPU (int8). ``base.en`` runs on a Raspberry Pi 5 for short commands."""

    def __init__(self, cfg: SpeechConfig | None = None):
        from faster_whisper import WhisperModel  # optional extra: pip install bruhos[speech]
        self.cfg = cfg or SpeechConfig()
        self.model = WhisperModel(self.cfg.asr_model, device="cpu", compute_type="int8")

    def transcribe(self, wav_path: str) -> str:
        segments, _ = self.model.transcribe(wav_path, language=self.cfg.language, beam_size=1, vad_filter=True)
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
