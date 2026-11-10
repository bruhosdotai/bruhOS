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


def parse_turn_step_deg(value: Any) -> float:
    """Validate a configured ``turn_step_deg``: one turn_* primitive's sweep.

    The scan counts its turns as ``int(360 / turn_step_deg)``, so the step
    must be a finite angle in (0, 180]: 0 divides by zero and crashes the
    run; a negative, NaN, or infinite step sweeps zero turns and reports
    "not found after full scan" without turning; above 180 a single turn
    passes the opposite heading, so a "full revolution" sweep cannot cover
    the circle. The 180 ceiling follows #218's stated load-time bound.
    """
    try:
        deg = float(value) if not isinstance(value, bool) else math.nan
    except (TypeError, ValueError):
        deg = math.nan
    if not (math.isfinite(deg) and 0.0 < deg <= 180.0):
        raise ValueError(
            f"turn_step_deg must be a finite angle in (0, 180] degrees, got {value!r}"
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
