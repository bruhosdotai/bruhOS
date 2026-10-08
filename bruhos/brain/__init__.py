"""Pluggable Brain. Every adapter emits the same skill JSON; the Runtime never changes."""

from __future__ import annotations

from typing import Any

from .base import IntegrationError, Planner, parse_plan


def make_planner(name: str, **cfg: Any) -> Planner:
    if name == "rules":
        from .adapters.rules import RulesPlanner
        return RulesPlanner()
    if name == "openai":
        from .adapters.openai import OpenAICompatPlanner
        return OpenAICompatPlanner(**cfg)
    if name == "qwen_vl":
        from .adapters.qwen_vl import QwenVLPlanner
        return QwenVLPlanner(**cfg)
    if name == "groot_s2":
        from .adapters.groot_s2 import GR00TS2Planner
        return GR00TS2Planner(**cfg)
    if name == "lerobot_policy":
        from .adapters.lerobot_policy import LeRobotPolicyPlanner
        return LeRobotPolicyPlanner(**cfg)
    raise ValueError(f"unknown brain {name!r} (rules | openai | qwen_vl | groot_s2 | lerobot_policy)")


__all__ = ["IntegrationError", "Planner", "make_planner", "parse_plan"]
