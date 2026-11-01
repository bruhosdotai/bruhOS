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


class RulesPlanner(Planner):
    name = "rules"
    version = "rules-0.1"

    def plan(self, instruction: str, obs: Observation, catalog: list[dict[str, Any]]) -> list[SkillCall]:
        s = " " + re.sub(r"[^a-z ]+", " ", instruction.lower()) + " "
        has = lambda *ws: any(f" {w} " in s or (" " in w and w in s) for w in ws)

        # A negated instruction forbids the action it names. This grammar is
        # single imperative commands and cannot represent "do X but not Y",
        # so the only faithful answer is no plan — the same answer the
        # planner gives for a request it cannot perform. ("don't" reaches
        # here as the tokens "don t": the apostrophe is normalised away.)
        # One exception: a stop commanded alongside the negated action
        # ("Don't move, stop.") still stops — halting is the safe action,
        # not part of what the negation forbids. A stop word directly after
        # the marker is the negated target itself ("Don't stop.") and
        # stays refused.
        if has("not", "never", "don t"):
            if has("stop", "halt", "freeze") and not re.search(
                r"\b(not|never|don t)\s+(stop|halt|freeze)\b", s
            ):
                return [_c("stop")]
            return []

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
