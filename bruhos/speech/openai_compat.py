"""OpenAI-compatible remote ASR/TTS (OpenAI, Ollama, zerollama, ...).

Uses stdlib only. Endpoints:

- ``POST {base_url}/audio/transcriptions``  (multipart)
- ``POST {base_url}/audio/speech``          (JSON → audio bytes)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .base import ASR, SpeechConfig, TTS, clean_transcript

Transport = Callable[[str, dict[str, str], bytes, float], bytes]


def http_post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
    req = Request(url, data=body, headers=headers, method="POST")
    with urlopen(req, timeout=timeout) as r:
        return r.read()


def http_post_multipart(url: str, headers: dict[str, str], fields: dict[str, str],
                        file_field: str, filename: str, file_bytes: bytes,
                        content_type: str, timeout: float) -> bytes:
    boundary = f"----bruhos{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
        )
    parts.append(
        (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"{file_field}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode()
        + file_bytes
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    hdrs = {**headers, "Content-Type": f"multipart/form-data; boundary={boundary}"}
    return http_post(url, hdrs, body, timeout)


def _auth_headers(api_key: str) -> dict[str, str]:
    h = {"User-Agent": "bruhos"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


class OpenAICompatASR(ASR):
    """Remote speech-to-text via ``/v1/audio/transcriptions``."""

    def __init__(self, cfg: SpeechConfig, transport: Transport | None = None,
                 multipart: Callable[..., bytes] | None = None):
        self.cfg = cfg
        self.base_url = (cfg.base_url or os.environ.get("BRUHOS_LLM_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.model = cfg.asr_model or "whisper-1"
        self.api_key = cfg.api_key if cfg.api_key is not None else os.environ.get(cfg.api_key_env, "")
        self.timeout = cfg.timeout
        self.transport = transport
        self.multipart = multipart or http_post_multipart

    def transcribe(self, wav_path: str) -> str:
        data = Path(wav_path).read_bytes()
        fields = {"model": self.model, "response_format": "json"}
        if self.cfg.language:
            fields["language"] = self.cfg.language
        raw = self.multipart(
            f"{self.base_url}/audio/transcriptions",
            _auth_headers(self.api_key),
            fields,
            "file",
            Path(wav_path).name or "audio.wav",
            data,
            "audio/wav",
            self.timeout,
        )
        return clean_transcript(self._text(raw))

    def translate(self, wav_path: str) -> str:
        """``POST /v1/audio/translations`` — English text from a speech clip."""
        data = Path(wav_path).read_bytes()
        fields = {"model": self.model, "response_format": "json"}
        raw = self.multipart(
            f"{self.base_url}/audio/translations",
            _auth_headers(self.api_key),
            fields,
            "file",
            Path(wav_path).name or "audio.wav",
            data,
            "audio/wav",
            self.timeout,
        )
        return clean_transcript(self._text(raw))

    @staticmethod
    def _text(raw: bytes) -> str:
        try:
            return str(json.loads(raw).get("text", "")).strip()
        except json.JSONDecodeError:
            return raw.decode("utf-8", errors="replace").strip()


def list_voices(cfg: SpeechConfig, model: str | None = None) -> dict:
    """``GET /v1/audio/voices``."""
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    base = (cfg.base_url or "").rstrip("/")
    url = f"{base}/audio/voices"
    if model:
        url = f"{url}?{urlencode({'model': model})}"
    req = Request(url, headers=_auth_headers(
        cfg.api_key if cfg.api_key is not None else os.environ.get(cfg.api_key_env, "")
    ))
    with urlopen(req, timeout=cfg.timeout) as r:
        return json.loads(r.read().decode())


class OpenAICompatTTS(TTS):
    """Remote text-to-speech via ``/v1/audio/speech``.

    ``playback``:
      - ``aplay`` — play locally with aplay (WonderEcho / headset)
      - ``http``  — write wav into ``http_dir`` (serve for VLC)
      - ``both``  — write ``http_dir/tts.wav`` and aplay
      - ``none``  — write a temp file and print its path
    """

    def __init__(self, cfg: SpeechConfig, transport: Transport | None = None):
        self.cfg = cfg
        self.base_url = (cfg.base_url or os.environ.get("BRUHOS_LLM_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.model = cfg.tts_model or "tts-1"
        self.voice = cfg.tts_voice or "alloy"
        self.api_key = cfg.api_key if cfg.api_key is not None else os.environ.get(cfg.api_key_env, "")
        self.timeout = cfg.timeout
        self.transport = transport or http_post
        self.playback = cfg.playback
        self.aplay_device = cfg.aplay_device
        self.http_dir = Path(cfg.http_dir)

    def synthesize(self, text: str) -> bytes:
        payload = {
            "model": self.model,
            "input": text,
            "voice": self.voice,
            "response_format": "wav",
        }
        if self.cfg.language:
            payload["lang"] = self.cfg.language
        if self.cfg.tts_emotion:
            payload["emotion"] = self.cfg.tts_emotion
        headers = {**_auth_headers(self.api_key), "Content-Type": "application/json"}
        try:
            return self.transport(
                f"{self.base_url}/audio/speech", headers, json.dumps(payload).encode(), self.timeout
            )
        except HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"TTS {e.code} from {self.base_url}: {detail}") from e

    def _aplay(self, path: str) -> None:
        player = shutil.which("aplay") or shutil.which("ffplay")
        if not player:
            print(f"[bruh] tts saved (no aplay/ffplay): {path}")
            return
        if "aplay" in player:
            cmd = [player]
            if self.aplay_device:
                cmd.extend(["-D", self.aplay_device])
            cmd.append(path)
        else:
            cmd = [player, "-nodisp", "-autoexit", path]
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def say(self, text: str) -> None:
        audio = self.synthesize(text)
        mode = self.playback
        http_path: Path | None = None
        if mode in ("http", "both"):
            self.http_dir.mkdir(parents=True, exist_ok=True)
            http_path = self.http_dir / "tts.wav"
            http_path.write_bytes(audio)
            print(f"[bruh] tts -> {http_path}")
            if mode == "http":
                return
        if mode == "none":
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio)
                print(f"[bruh] tts -> {f.name}")
            return
        # aplay or both
        if http_path is not None:
            self._aplay(str(http_path))
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio)
            path = f.name
        try:
            self._aplay(path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
