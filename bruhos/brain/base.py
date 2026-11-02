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

# The prose-recovery scan in _balanced_object is O(n^2) in the response length
# (every '{' restarts an inner scan). The transport allows 4 MiB, so an
# unbounded recovery would let one adversarial or broken model response wedge
# the planner thread for hours. A real plan is a few hundred bytes; recovery is
# only attempted within this bound, and above it the garbage fails immediately
# with the original JSONDecodeError. The cap changes no recovery outcome: the
# scan returns None for any unbalanced tail of this size, so capping it only
# removes the quadratic spin.
_RECOVERY_MAX_BYTES = 64 * 1024


def _balanced_object(text: str) -> str | None:
    """Return the first balanced top-level JSON object in ``text``, or None.

    The first ``{`` that opens a string is not a JSON start (``{"plan": ...}`` begins with
    a quoted key), so scan every ``{`` and validate the balanced span by actually parsing
    it rather than trusting the last ``}`` at depth 0 alone.
    """
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    span = text[start:i + 1]
                    try:
                        json.loads(span)
                    except json.JSONDecodeError:
                        break
                    return span
        start = text.find("{", start + 1)
    return None


def parse_plan(raw: Any) -> list[SkillCall]:
    """Accept ``{"plan": [...]}``, a list of ``{"skill", "args"}`` objects, or shorthand strings.

    Models frequently wrap the plan in prose ("Here is the plan: {...}"). The fenced
    JSON is parsed as-is; when that fails, the first balanced JSON object in the text is
    retried (capped at _RECOVERY_MAX_BYTES so a pathological response cannot wedge the
    planner thread). Garbage is still rejected — only a genuinely parseable object is
    recovered.
    """
    if isinstance(raw, (bytes, str)):
        text = _FENCE.sub("", raw.decode() if isinstance(raw, bytes) else raw).strip()
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            if len(text) > _RECOVERY_MAX_BYTES:
                raise
            candidate = _balanced_object(text)
            if candidate is None:
                raise
            raw = json.loads(candidate)
    guard_no_joint_actions(raw)
    items = raw.get("plan") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ValueError(f"plan must be a list, got {type(items).__name__}")
    pos = positional_args()
    return [parse_call(i, pos) if isinstance(i, str) else SkillCall.from_dict(i) for i in items]
