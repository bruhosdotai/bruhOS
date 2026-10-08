"""Isaac GR00T System-2 adapter — stub in v0.1.

Contract: GR00T's reasoning head decomposes a task into sub-steps, which map onto bruhOS skills:

    GR00T_S2(image, "Put the red one in the box.")
      -> ["detect", "walk_toward(red_block)", "grab", "walk_toward(box)", "release"]

The DiT action head (continuous whole-body joint chunks) is never routed to the TonyPi servos;
``parse_plan`` raises ``IntegrationError`` if such output shows up.

v0.1 ships the contract and the guard. Pass ``reasoner`` (any callable returning sub-step strings,
e.g. a wrapper around a GR00T inference server) to use it; the real wrapper lands in v0.2.
"""

from __future__ import annotations

from typing import Any, Callable

from bruhos.brain.base import Planner, parse_plan
from bruhos.schemas import Observation, SkillCall

Reasoner = Callable[[bytes | None, str, dict[str, Any]], Any]


class GR00TS2Planner(Planner):
    name = "groot_s2"
    version = "groot_s2-stub-0.1"

    def __init__(self, reasoner: Reasoner | None = None, frame_provider: Callable[[], bytes | None] | None = None):
        self.reasoner = reasoner
        self.frame_provider = frame_provider

    def plan(self, instruction: str, obs: Observation, catalog: list[dict[str, Any]]) -> list[SkillCall]:
        if self.reasoner is None:
            raise NotImplementedError("GR00T S2 adapter is a stub in v0.1: pass reasoner=... (see module docstring)")
        frame = self.frame_provider() if self.frame_provider else None
        return parse_plan(self.reasoner(frame, instruction, obs.to_dict()))
