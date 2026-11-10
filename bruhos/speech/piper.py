"""Local Piper TTS.

The voice is an ONNX file. llama-server's speech route is for a GGUF
voice, which this Pi does not have, so the ear shells out to piper.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .base import SpeechConfig
from .openai_compat import OpenAICompatTTS

DEFAULT_DATA_DIR = "/var/lib/bruhos/voices"
DEFAULT_VOICE = "en_US-lessac-medium"


def piper_bin() -> str:
    found = shutil.which("piper")
    if found:
        return found
    candidate = "/opt/piper/bin/piper"
    if os.path.isfile(candidate):
        return candidate
    raise RuntimeError("piper not found (expected /usr/local/bin/piper or /opt/piper/bin/piper)")


def piper_model_path(tts_model: str | None, data_dir: str = DEFAULT_DATA_DIR) -> str:
    name = (tts_model or DEFAULT_VOICE).strip()
    path = Path(name)
    if path.suffix == ".onnx" or path.is_absolute():
        return str(path)
    return str(Path(data_dir) / f"{name}.onnx")


class PiperTTS(OpenAICompatTTS):
    """Synthesize with the piper CLI, then play with the usual aplay/http modes."""

    def __init__(self, cfg: SpeechConfig, runner=None):
        super().__init__(cfg)
        self.runner = runner or subprocess.run
        self.model_path = piper_model_path(cfg.tts_model)

    def synthesize(self, text: str) -> bytes:
        if not os.path.isfile(self.model_path):
            raise RuntimeError(f"piper voice not found: {self.model_path}")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            out_path = handle.name
        cmd = [piper_bin(), "-m", self.model_path, "-f", out_path]
        try:
            proc = self.runner(
                cmd,
                input=text.encode(),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if proc.returncode != 0 or not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
                err = (proc.stderr or b"").decode("utf-8", errors="replace")[:300]
                raise RuntimeError(f"piper failed: {err or proc.returncode}")
            return Path(out_path).read_bytes()
        finally:
            try:
                os.unlink(out_path)
            except OSError:
                pass
