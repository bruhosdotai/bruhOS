"""Planner interface and the shared plan parser / integration guard."""

from __future__ import annotations

import json
import numbers
import re
from abc import ABC, abstractmethod
from typing import Any

from bruhos.schemas import Observation, SkillCall, parse_call
from bruhos.skills.registry import positional_args


class IntegrationError(RuntimeError):
    """A brain produced something that must never reach the body (e.g. raw joint actions)."""


class Planner(ABC):
    name = "planner"
    version = "0"

    @abstractmethod
    def plan(self, instruction: str, obs: Observation, catalog: list[dict[str, Any]]) -> list[SkillCall]:
        ...

    def replan(self, instruction: str, obs: Observation, catalog: list[dict[str, Any]],
               history: list[dict[str, Any]]) -> list[SkillCall]:
        """Called after a failed skill. Default: plan again from the current observation."""
        return self.plan(instruction, obs, catalog)


def _is_number_chunk(x: Any) -> bool:
    if isinstance(x, numbers.Number) and not isinstance(x, bool):
        return True
    return isinstance(x, (list, tuple)) and len(x) > 0 and all(_is_number_chunk(v) for v in x)


def guard_no_joint_actions(raw: Any) -> None:
    """Reject continuous action chunks (e.g. GR00T / VLA joint targets). Skills only."""
    items = raw.get("plan", raw.get("actions", raw)) if isinstance(raw, dict) else raw
    if _is_number_chunk(items) or (isinstance(items, list) and any(_is_number_chunk(i) for i in items)):
        raise IntegrationError("brain returned a continuous action chunk; bruhOS only executes whitelisted skills")


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_plan(raw: Any) -> list[SkillCall]:
    """Accept ``{"plan": [...]}``, a list of ``{"skill", "args"}`` objects, or shorthand strings."""
    if isinstance(raw, (bytes, str)):
        text = _FENCE.sub("", raw.decode() if isinstance(raw, bytes) else raw).strip()
        raw = json.loads(text)
    guard_no_joint_actions(raw)
    items = raw.get("plan") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError(f"plan must be a list, got {type(items).__name__}")
    pos = positional_args()
    return [parse_call(i, pos) if isinstance(i, str) else SkillCall.from_dict(i) for i in items]
