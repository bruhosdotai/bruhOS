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


# A "to <object>" destination names somewhere the object must end up. The
# box is the only destination the skill set can place an object at (release
# into the box is a modeled containment); any other object destination is
# silently dropped by the branches below, which then certify success for
# work that never happened. "next to" is a spatial relation, not a
# destination, so it does not count.
_TO_OBJECT = re.compile(r"\b(?<!next\s)to\s+(the\s+)?([a-z]+\s+)?(block|ball|line)\b")
_OBJECT_WORDS = COLORS | set(NOUNS)


def _names_object_destination(text: str) -> bool:
    """True when an object is named before a "to <object>" destination.

    Pure navigation ("Walk to the box.", "Go to the green block.") has no
    object before the "to", so the "to" phrase is the target itself, not a
    destination for something else.
    """
    m = _TO_OBJECT.search(text)
    if not m:
        return False
    head = text[: m.start()]
    return bool(set(re.findall(r"[a-z]+", head)) & _OBJECT_WORDS)


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
            if obj in (None, "line") or has(" it "):
                if obs.holding:
                    return [_c("walk_toward", target="box"), _c("release")]
                obj = obj if obj not in (None, "line") else "block"
            return [_c("detect", target=obj), _c("walk_toward", target=obj), _c("grab"),
                    _c("walk_toward", target="box"), _c("release")]

        if has("kick"):
            if _names_object_destination(s):
                return []
            foot = "left" if has("left") else "right"
            if has("ball"):
                return [_c("walk_toward", target=_target(s) or "ball", stop_distance_m=0.12), _c("kick", foot=foot)]
            return [_c("kick", foot=foot)]
        if has("follow"):
            return [_c("follow_line")]
        if has("wave", "hello", "hi", "greet", "say hi"):
            return [_c("wave")]
        if has("drop", "release", "let go", "open your hand"):
            if _names_object_destination(s):
                return []
            return [_c("release")]

        obj = _target(s)
        if has("pick up", "grab", "take", "get", "hold") and obj:
            if _names_object_destination(s):
                return []
            return [_c("walk_toward", target=obj), _c("grab")]
        if has("walk", "go", "come", "approach", "move") and obj:
            if _names_object_destination(s):
                return []
            return [_c("walk_toward", target=obj)]
        if has("look", "face", "turn") and obj:
            return [_c("look_at", target=obj)]
        if has("find", "where", "locate", "search", "detect", "see") and obj:
            return [_c("detect", target=obj)]
        if has("what do you see", "look around", "what s around", "scan"):
            return [_c("detect")]
        return []
