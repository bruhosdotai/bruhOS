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


def action_basename(name: object) -> bool:
    """True when ``name`` is one action-group file stem.

    Hiwonder ``runAction`` opens ``path + name + ".d6a"`` and writes each row
    with ``set_bus_servo_pulse``. A separator or ``..`` leaves that directory
    and plays some other servo script. ``None`` and ``""`` mean "unmapped".
    """
    if not isinstance(name, str) or not name or name in {".", ".."}:
        return False
    return "/" not in name and "\\" not in name and "\x00" not in name


def require_action_basenames(actions: dict) -> None:
    """Reject action-map values that are not a single basename."""
    bad = sorted(
        (
            (str(key), value)
            for key, value in actions.items()
            if value not in (None, "") and not action_basename(value)
        ),
        key=lambda item: item[0],
    )
    if bad:
        listed = ", ".join(f"{key}={value!r}" for key, value in bad)
        raise ValueError(f"unsafe action group name(s): {listed}; a name must be a single basename, not a path")


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
