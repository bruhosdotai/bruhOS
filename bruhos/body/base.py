"""Body interface. Skills are written once against these primitives and run on sim or hardware."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from bruhos.schemas import Observation

PRIMITIVES = (
    "stand",
    "forward",
    "turn_left",
    "turn_right",
    "grab",
    "release",
    "kick_left",
    "kick_right",
    "wave",
    "stand_up_front",
    "stand_up_back",
)


def parse_fall_deg(value: Any) -> float:
    """Validate a configured ``fall_deg``: the tilt that counts as a fall.

    Fall detection compares ``abs(tilt) > fall_deg``, so the threshold must be
    a finite angle in (0, 180). NaN or inf never fires the comparison and
    silently disables fall detection (a 70-degree roll reads ``fallen=False``);
    0 or a negative value fires on any micro-tilt, so an upright robot reads
    ``fallen=True``. 180 can never fire either, because no tilt exceeds it.
    """
    try:
        deg = float(value) if not isinstance(value, bool) else math.nan
    except (TypeError, ValueError):
        deg = math.nan
    if not (math.isfinite(deg) and 0.0 < deg < 180.0):
        raise ValueError(
            f"fall_deg must be a finite angle in (0, 180) degrees, got {value!r}"
        )
    return deg


class Body(ABC):
    name: str = "body"
    turn_step_deg: float = 15.0

    @abstractmethod
    def observe(self) -> Observation:
        """Compressed observation of the current state. Never raw video."""

    @abstractmethod
    def act(self, primitive: str) -> bool:
        """Run one primitive to completion. Returns False if the body could not run it."""

    @abstractmethod
    def halt(self) -> None:
        """Stop motion immediately. Must be safe to call from any thread."""

    def close(self) -> None:
        pass
