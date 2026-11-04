"""ALSA / wav diagnostics for the WonderEcho ear path."""

from __future__ import annotations

import array
import shutil
import struct
import subprocess
import wave
from pathlib import Path


def log(msg: str) -> None:
    print(f"[ear:audio] {msg}", flush=True)


def alsa_snapshot(card_hint: str = "Device") -> None:
    """Print cards + mixer state (best-effort; never raises)."""
    try:
        cards = Path("/proc/asound/cards").read_text(errors="replace").strip()
        log(f"cards:\n{cards}")
    except OSError as e:
        log(f"cards unreadable: {e}")

    for tool, args in (
        ("aplay", ["-l"]),
        ("arecord", ["-l"]),
    ):
        bin_ = shutil.which(tool)
        if not bin_:
            log(f"{tool} missing")
            continue
        r = subprocess.run([bin_, *args], capture_output=True, text=True)
        out = (r.stdout or r.stderr or "").strip()
        log(f"{tool} -l:\n{out}")

    amixer = shutil.which("amixer")
    if not amixer:
        return
    for card in (card_hint, "2", "0"):
        r = subprocess.run(
            [amixer, "-c", card, "sget", "Mic"],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0:
            log(f"amixer -c {card} Mic:\n{(r.stdout or '').strip()}")
            r2 = subprocess.run(
                [amixer, "-c", card, "sget", "Speaker"],
                capture_output=True,
                text=True,
            )
            if r2.returncode == 0:
                log(f"amixer -c {card} Speaker:\n{(r2.stdout or '').strip()}")
            return
    log(f"amixer: no Mic control for hint={card_hint!r}")


def wav_stats(path: Path) -> dict[str, float | int | str]:
    """Peak / RMS for a PCM wav (S16)."""
    info: dict[str, float | int | str] = {
        "path": str(path),
        "bytes": path.stat().st_size if path.exists() else 0,
    }
    if not path.exists() or path.stat().st_size < 44:
        info["error"] = "missing_or_tiny"
        return info
    try:
        with wave.open(str(path), "rb") as w:
            nch, sw, rate, nframes, _, _ = w.getparams()
            raw = w.readframes(nframes)
            info.update(channels=nch, sampwidth=sw, rate=rate, frames=nframes)
    except wave.Error as e:
        info["error"] = f"wave:{e}"
        return info

    if sw != 2 or not raw:
        info["error"] = f"unsupported sampwidth={sw} or empty"
        return info

    samples = array.array("h")
    samples.frombytes(raw)
    if samples.itemsize != struct.calcsize("h"):
        samples.byteswap()
    n = len(samples)
    if not n:
        info["error"] = "no_samples"
        return info
    peak = max(abs(s) for s in samples)
    # mean square without numpy
    acc = 0.0
    for s in samples:
        acc += float(s) * float(s)
    rms = (acc / n) ** 0.5
    info["peak"] = peak
    info["rms"] = round(rms, 1)
    info["peak_dbfs"] = round(20.0 * __import__("math").log10(peak / 32767.0), 1) if peak else -999.0
    info["rms_dbfs"] = round(20.0 * __import__("math").log10(rms / 32767.0), 1) if rms else -999.0
    info["silent"] = peak < 200
    return info


def log_wav(path: Path, label: str) -> None:
    st = wav_stats(path)
    log(f"{label} {st}")
