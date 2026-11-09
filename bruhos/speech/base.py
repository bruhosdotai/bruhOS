"""Speech config and interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

DEFAULT_LANGUAGE = "en"


@dataclass
class SpeechConfig:
    language: str = DEFAULT_LANGUAGE
    asr_model: str = "base.en"          # local whisper checkpoint, or remote model tag
    tts_voice: str = "en-us"            # espeak voice, or remote voice id
    allow_non_english: bool = False
    # ``local`` = faster-whisper / espeak; ``openai`` = OpenAI-compatible remote API
    asr: str = "local"
    tts: str = "local"                  # local | openai | print
    base_url: str | None = None         # e.g. http://host:2083/v1
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    tts_model: str | None = None        # e.g. piper-lessac:latest
    tts_emotion: str | None = None      # zerollama: calm|neutral|happy|… (Kokoro speed / remote-tts)
    tts_rate: int = 22050               # duplex PCM playback rate when session.created omits one
    playback: str = "aplay"             # aplay | http | both | none
    aplay_device: str | None = None     # e.g. plughw:Device,0 (WonderEcho)
    http_dir: str = "/tmp/bruh-audio"
    timeout: float = 60.0
    # ``duplex`` streams the mic on GET /v1/realtime?pipeline=duplex
    realtime: str = "off"               # off | duplex
    realtime_turn: str = "server_vad"   # server_vad | none
    chat_model: str | None = None       # duplex chat model; ear fills this from [brain]

    def __post_init__(self) -> None:
        if self.language != "en" and not self.allow_non_english:
            raise ValueError(
                f"bruhOS is English-first; language={self.language!r} needs allow_non_english=True"
            )


class ASR(ABC):
    @abstractmethod
    def transcribe(self, wav_path: str) -> str: ...


class TTS(ABC):
    @abstractmethod
    def say(self, text: str) -> None: ...
