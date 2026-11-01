"""Offline English planner. No model, no network — the baseline every other brain is compared to."""

from __future__ import annotations

import re
from typing import Any

from bruhos.brain.base import Planner
from bruhos.schemas import Observation, SkillCall

COLORS = {"red", "blue", "green", "yellow", "black", "white", "orange", "purple"}
NOUNS = {"block": "block", "cube": "block", "brick": "block", "box": "box", "bin": "box",
         "basket": "box", "ball": "ball", "line": "line"}


def _target(text: str) -> str | None:
    words = re.findall(r"[a-z]+", text.lower())
    color = next((w for w in words if w in COLORS), None)
    nouns = [NOUNS[w] for w in words if w in NOUNS]
    noun = next((n for n in nouns if n != "box"), "box" if "box" in nouns else None)
    if color and noun:
        return f"{color}_{noun}"
    return color or noun


def _c(skill: str, **args: Any) -> SkillCall:
    return SkillCall(skill, args)


def _holding_matches(holding: str | None, target: str | None) -> bool:
    """True when the held object satisfies a (possibly partial) target name.

    Same token rule as ``Detection.matches``: every token of the target must
    appear in the held name, so "red" and "red_block" both match a held
    "red_block", while "blue" does not.
    """
    if not holding or not target:
        return False
    tokens = {t for t in target.lower().replace("-", "_").split("_") if t and t not in {"the", "one"}}
    have = set(holding.lower().replace("-", "_").split("_"))
    return bool(tokens) and tokens <= have


class RulesPlanner(Planner):
    name = "rules"
    version = "rules-0.1"

    def plan(self, instruction: str, obs: Observation, catalog: list[dict[str, Any]]) -> list[SkillCall]:
        s = " " + re.sub(r"[^a-z ]+", " ", instruction.lower()) + " "
        has = lambda *ws: any(f" {w} " in s or (" " in w and w in s) for w in ws)

        if has("stop", "halt", "freeze"):
            return [_c("stop")]
        if has("stand up", "get up", "get back up"):
            return [_c("stand_up")]

        into_box = re.search(r"\b(in|into|inside|to)\s+(the\s+)?(box|bin|basket)\b", s)
        if has("put", "place", "drop", "bring", "move", "carry", "throw") and into_box:
            obj = _target(s[: into_box.start()])
            if _holding_matches(obs.holding, obj):
                # Already holding the named object: delivering it needs no
                # detect/walk/grab, and detect could never find a held object
                # (held objects are excluded from observations).
                return [_c("walk_toward", target="box"), _c("release")]
            if obj in (None, "line") or has(" it "):
                if obs.holding:
                    return [_c("walk_toward", target="box"), _c("release")]
                obj = obj if obj not in (None, "line") else "block"
            return [_c("detect", target=obj), _c("walk_toward", target=obj), _c("grab"),
                    _c("walk_toward", target="box"), _c("release")]

        if has("kick"):
            foot = "left" if has("left") else "right"
            if has("ball"):
                return [_c("walk_toward", target=_target(s) or "ball", stop_distance_m=0.12), _c("kick", foot=foot)]
            return [_c("kick", foot=foot)]
        if has("follow"):
            return [_c("follow_line")]
        if has("wave", "hello", "hi", "greet", "say hi"):
            return [_c("wave")]
        if has("drop", "release", "let go", "open your hand"):
            return [_c("release")]

        obj = _target(s)
        if has("pick up", "grab", "take", "get", "hold") and obj:
            return [_c("walk_toward", target=obj), _c("grab")]
        if has("walk", "go", "come", "approach", "move") and obj:
            return [_c("walk_toward", target=obj)]
        if has("look", "face", "turn") and obj:
            return [_c("look_at", target=obj)]
        if has("find", "where", "locate", "search", "detect", "see") and obj:
            return [_c("detect", target=obj)]
        if has("what do you see", "look around", "what s around", "scan"):
            return [_c("detect")]
        return []
