"""Normalize browser / phone audio to 16 kHz mono WAV for STT (system ffmpeg)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class NormalizeError(RuntimeError):
    pass


def require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise NormalizeError("ffmpeg not found (apt install ffmpeg)")
    return path


def is_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def normalize_to_wav(data: bytes, suffix: str = ".webm", rate: int = 16000) -> Path:
    """Write ``data`` through ffmpeg to a temp 16-bit mono WAV. Caller may delete the path."""
    if is_wav(data) and suffix.lower() in (".wav", ""):
        out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out.write(data)
        out.close()
        return Path(out.name)

    ffmpeg = require_ffmpeg()
    if not suffix.startswith("."):
        suffix = "." + suffix
    src = tempfile.NamedTemporaryFile(suffix=suffix or ".bin", delete=False)
    src.write(data)
    src.close()
    dst = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    dst.close()
    # Prefer plain decode; Safari MediaRecorder often emits fragmented MP4 that needs -fflags.
    attempts = [
        [ffmpeg, "-y", "-i", src.name, "-ac", "1", "-ar", str(rate), "-f", "wav", dst.name],
        [ffmpeg, "-y", "-fflags", "+genpts+igndts", "-err_detect", "ignore_err",
         "-i", src.name, "-ac", "1", "-ar", str(rate), "-f", "wav", dst.name],
    ]
    errors: list[str] = []
    try:
        for cmd in attempts:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and Path(dst.name).stat().st_size > 44:
                return Path(dst.name)
            errors.append((r.stderr or r.stdout or "")[-200:])
        raise NormalizeError("ffmpeg failed: " + " | ".join(errors)[-400:])
    finally:
        Path(src.name).unlink(missing_ok=True)


def normalize_file(path: str | Path, rate: int = 16000) -> Path:
    p = Path(path)
    data = p.read_bytes()
    if is_wav(data) and p.suffix.lower() == ".wav":
        return p
    return normalize_to_wav(data, suffix=p.suffix or ".bin", rate=rate)
