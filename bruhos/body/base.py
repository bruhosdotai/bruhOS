"""Body interface. Skills are written once against these primitives and run on sim or hardware."""

from __future__ import annotations

from abc import ABC, abstractmethod

from bruhos.schemas import Detection, Observation

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

# Labels the hand cannot grasp (same set the sim excludes in _p_grab).
UNGRASPABLE_LABELS = frozenset({"box", "ball"})

# Hardware bodies name the grabbed object from the nearest seen object. This
# bound is generous against the sim's 0.12 m reach: hardware distance comes
# from blob size / bridge telemetry, so it is noisy, but anything beyond it is
# clearly not what the hand closed on.
GRAB_NAME_REACH_M = 0.30


def grabbed_name(detections: list[Detection]) -> str | None:
    """Name of the object the hand most plausibly closed on.

    The nearest seen graspable object within reach, else None (the caller
    keeps the generic ``"object"``). Hardware bodies have no grasp sensor, so
    they record the name this way instead of losing it: the skill-level
    held-target checks (#202/#204) parse the held name, and ``"object"``
    never matches a named target.
    """
    best: tuple[float, Detection] | None = None
    for d in detections:
        if d.label in UNGRASPABLE_LABELS:
            continue
        dist = d.distance_m if d.distance_m is not None else 1e9
        if dist <= GRAB_NAME_REACH_M and (best is None or dist < best[0]):
            best = (dist, d)
    return best[1].name if best else None


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
