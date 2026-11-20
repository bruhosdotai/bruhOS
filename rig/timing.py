"""Whole-clip playback scale so LX-824HV moves keep source pose ratios.

Frames are already servo pulses (1–16). Bone degrees are not servo degrees:
shared bones must be split before this function sees them. Never opens the
servo port and never writes a .d6a.
"""

from __future__ import annotations

from dataclasses import dataclass

# Spec from docs/body.md: ~0.20 s per 60° at 11.1 V.
DEG_PER_SEC = 300.0
COUNTS_PER_DEG = 1000.0 / 240.0  # 0–1000 pulse ↔ 0–240°
MARGIN = 1.15


@dataclass(frozen=True)
class ScaleResult:
    """One overall scale for the take, and which step forced it."""

    scale: float
    limiting_step: int  # 0 = stand→first pose; i = frames[i-1]→frames[i]
    step_ms: tuple[int, ...]  # duration to command for each step, whole ms


def _pulse_delta_deg(a: int, b: int) -> float:
    return abs(b - a) / COUNTS_PER_DEG


def scale_take(
    stand: dict[int, int],
    frames: list[dict[int, int]],
    source_dt_s: list[float],
    *,
    deg_per_sec: float = DEG_PER_SEC,
    margin: float = MARGIN,
) -> ScaleResult:
    """Slow the whole take so the busiest servo can finish every shared move.

    ``stand`` is the pose before the first key (usually stand.d6a).
    ``frames[i]`` maps servo id → pulse. ``source_dt_s[i]`` is the source
    interval ending at ``frames[i]`` (index 0 = stand → frames[0]).

    Returns a single scale ≥ 1, the step that set it, and per-step times in ms.
    """
    if not frames:
        raise ValueError("frames must not be empty")
    if len(source_dt_s) != len(frames):
        raise ValueError("source_dt_s must match frames length")
    if any(dt <= 0 for dt in source_dt_s):
        raise ValueError("source_dt_s must be positive")

    body_ids = sorted({sid for sid in stand if 1 <= sid <= 16} | {sid for f in frames for sid in f if 1 <= sid <= 16})

    prev = {sid: stand.get(sid, 500) for sid in body_ids}
    stretches: list[float] = []
    for i, (frame, dt) in enumerate(zip(frames, source_dt_s)):
        peak_deg = 0.0
        for sid in body_ids:
            nxt = frame.get(sid, prev[sid])
            peak_deg = max(peak_deg, _pulse_delta_deg(prev[sid], nxt))
            prev[sid] = nxt
        needed = peak_deg / deg_per_sec if deg_per_sec > 0 else 0.0
        stretches.append(needed / dt)

    worst = max(stretches) if stretches else 0.0
    limiting = max(range(len(stretches)), key=lambda i: stretches[i]) if stretches else 0
    scale = max(1.0, worst * margin)

    step_ms = tuple(max(1, int(round(dt * scale * 1000.0))) for dt in source_dt_s)
    return ScaleResult(scale=scale, limiting_step=limiting, step_ms=step_ms)
