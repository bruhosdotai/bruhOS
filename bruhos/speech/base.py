"""Speech config and interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

DEFAULT_LANGUAGE = "en"

# Config names and the codes zerollama already accepts. Tony picks one.
LANGUAGE_CODES = {
    "en": "en",
    "english": "en",
    "zh": "zh",
    "chinese": "zh",
    "ms": "ms",
    "malay": "ms",
    "th": "th",
    "thai": "th",
    "vi": "vi",
    "vietnam": "vi",
    "vietnamese": "vi",
}
LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese",
    "ms": "Malay",
    "th": "Thai",
    "vi": "Vietnamese",
}


def normalize_language(value: str) -> str | None:
    """Map a config word to the code sent on the duplex socket. None if unknown."""
    return LANGUAGE_CODES.get(str(value or "").strip().lower())


def language_name(code: str) -> str:
    """Spoken name for the reply instructions. Unknown codes pass through."""
    return LANGUAGE_NAMES.get(code, code)


# Qwen3-ASR is trained to answer ``language English<asr_text>`` and then the words.
# llama.cpp does not prefill that tag, so it comes back inside the transcript.
# The official toolkit splits on the tag and keeps the words. ``language None``
# with nothing after the tag is silence. No tag means the text is already the
# words, which is what Whisper returns.
_ASR_TEXT_TAG = "<asr_text>"


def clean_transcript(raw: str) -> str:
    """Return the spoken words from an ASR string."""
    s = str(raw or "").strip()
    if _ASR_TEXT_TAG not in s:
        return s
    meta, text = s.split(_ASR_TEXT_TAG, 1)
    text = text.strip()
    if "language none" in meta.lower() and not text:
        return ""
    return text


@dataclass
class SpeechConfig:
    language: str = DEFAULT_LANGUAGE
    asr_model: str = "base.en"          # local whisper checkpoint, or remote model tag
    tts_voice: str = "en-us"            # espeak voice, or remote voice id
    allow_non_english: bool = False
    # ``local`` = faster-whisper / espeak; ``openai`` = OpenAI-compatible remote API
    asr: str = "local"
    tts: str = "local"                  # local | piper | openai | print
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
        code = normalize_language(self.language)
        if code is None:
            if not self.allow_non_english:
                raise ValueError(
                    f"language={self.language!r} is not one of "
                    "english, chinese, malay, thai, vietnam"
                )
            return
        self.language = code


class ASR(ABC):
    @abstractmethod
    def transcribe(self, wav_path: str) -> str: ...


class TTS(ABC):
    @abstractmethod
    def say(self, text: str) -> None: ...
