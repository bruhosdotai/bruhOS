"""Body interface. Skills are written once against these primitives and run on sim or hardware."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from bruhos.schemas import Observation


def parse_forward_sign(value: Any) -> int:
    """Return the IMU mount-direction flip: 1 (normal) or -1 (board reversed).

    Anything else silently breaks fall detection: 0 zeroes the pitch, so a
    robot lying face-down reads ``fallen=False``; any other magnitude
    rescales the pitch, so a small tilt reads as a fall (or vice versa).
    Reject it at load, before the bus opens or the bridge connects.
    """
    if isinstance(value, bool):
        raise ValueError(f"forward_sign must be 1 or -1, got {value!r}")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"forward_sign must be 1 or -1, got {value!r}") from None
    if v == 1.0:
        return 1
    if v == -1.0:
        return -1
    raise ValueError(f"forward_sign must be 1 or -1, got {value!r}")

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
