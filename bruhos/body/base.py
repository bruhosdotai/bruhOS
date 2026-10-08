"""Body interface. Skills are written once against these primitives and run on sim or hardware."""

from __future__ import annotations

from abc import ABC, abstractmethod

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
