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


def parse_turn_step(value: Any) -> float:
    """Validate a configured ``turn_step_deg``: the degrees one ``turn_left`` / ``turn_right`` rotates.

    Skills divide a revolution by it and steer one step at a time, so it must be a finite angle in
    (0, 180]. Zero or NaN crashes ``_scan``; a negative step or one above 180 turns the other way and
    a step of 360 or more makes ``detect`` give up without turning.
    """
    try:
        step = float(value) if not isinstance(value, bool) else math.nan
    except (TypeError, ValueError):
        step = math.nan
    if not (math.isfinite(step) and 0.0 < step <= 180.0):
        raise ValueError(f"turn_step_deg must be a finite angle in (0, 180] degrees, got {value!r}")
    return step


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
