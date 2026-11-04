"""Vertical head + monocular depth: feet for blockers, ahead for range."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from bruhos.roam.depth import MetricDepth, grab_bgr


@dataclass(frozen=True)
class LevelLook:
    """One pitch: left / center / right median metres (lower = closer)."""

    name: str
    pitch_us: int
    left_m: float
    center_m: float
    right_m: float
    elapsed_s: float

    @property
    def clearest(self) -> str:
        m = {"left": self.left_m, "center": self.center_m, "right": self.right_m}
        return max(m, key=m.get)


@dataclass(frozen=True)
class VerticalLook:
    feet: LevelLook | None
    ahead: LevelLook | None
    feet_block_m: float
    ahead_block_m: float

    @property
    def feet_blocked(self) -> bool:
        return self.feet is not None and self.feet.center_m < self.feet_block_m

    @property
    def ahead_blocked(self) -> bool:
        return self.ahead is not None and self.ahead.center_m < self.ahead_block_m

    @property
    def blocked(self) -> bool:
        return self.feet_blocked or self.ahead_blocked

    @property
    def clear(self) -> bool:
        if self.feet is None and self.ahead is None:
            return False
        return not self.blocked

    def prefer_turn_left(self) -> bool:
        """True → turn left; False → turn right. Prefer the clearer side ahead (else feet)."""
        band = self.ahead or self.feet
        if band is None:
            return True
        return band.left_m >= band.right_m


class VerticalScanner:
    """Nod: look down (table pitch) then mid/ahead; depth each pose."""

    def __init__(
        self,
        head: Any,
        cap: Any,
        depth: MetricDepth,
        *,
        settle_s: float = 0.35,
        feet_block_m: float = 0.28,
        ahead_block_m: float = 0.45,
        ahead_pitch_frac: float = 0.45,
    ) -> None:
        self.head = head
        self.cap = cap
        self.depth = depth
        self.settle_s = settle_s
        self.feet_block_m = feet_block_m
        self.ahead_block_m = ahead_block_m
        # Between pitch_min (down) and pitch_max (up); ~0.45 ≈ straight-ish for TonyPi.
        self.ahead_pitch_frac = ahead_pitch_frac

    def _pitch_feet(self) -> int:
        return int(self.head.pitch_min_us)

    def _pitch_ahead(self) -> int:
        lo, hi = self.head.pitch_min_us, self.head.pitch_max_us
        return int(lo + self.ahead_pitch_frac * (hi - lo))

    def _one(self, name: str, pitch_us: int) -> LevelLook | None:
        yaw = int(self.head.yaw_center)
        self.head.go_pulse(yaw, pitch_us)
        time.sleep(self.settle_s)
        bgr = grab_bgr(self.cap)
        if bgr is None:
            return None
        # Feet: lower FOV; ahead: mid band.
        if name == "feet":
            row_lo, row_hi = 0.45, 0.95
        else:
            row_lo, row_hi = 0.30, 0.75
        band = self.depth.bands(bgr, mid_frac=0.35, row_lo=row_lo, row_hi=row_hi)
        return LevelLook(
            name=name,
            pitch_us=pitch_us,
            left_m=band.left_m,
            center_m=band.center_m,
            right_m=band.right_m,
            elapsed_s=band.elapsed_s,
        )

    def look(self) -> VerticalLook:
        feet = self._one("feet", self._pitch_feet())
        ahead = self._one("ahead", self._pitch_ahead())
        # Leave looking ahead for the next walk.
        try:
            self.head.go_pulse(self.head.yaw_center, self._pitch_ahead())
        except Exception:  # noqa: BLE001
            pass
        return VerticalLook(
            feet=feet,
            ahead=ahead,
            feet_block_m=self.feet_block_m,
            ahead_block_m=self.ahead_block_m,
        )


def format_look(v: VerticalLook) -> str:
    parts = []
    if v.feet is not None:
        f = v.feet
        parts.append(f"feet L/C/R={f.left_m:.2f}/{f.center_m:.2f}/{f.right_m:.2f}m")
    if v.ahead is not None:
        a = v.ahead
        parts.append(f"ahead L/C/R={a.left_m:.2f}/{a.center_m:.2f}/{a.right_m:.2f}m")
    flag = "BLOCK" if v.blocked else "clear"
    return f"{flag} " + " | ".join(parts)
