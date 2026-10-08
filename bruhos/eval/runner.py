"""Run the English prompt set against the kinematic sim and score it.

Completion rate and fall-recovery rate are reported separately: a run that falls, gets up and
finishes counts as completed *and* as a recovery.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from bruhos.brain.base import Planner
from bruhos.runtime import Runtime, RuntimeConfig
from bruhos.schemas import Episode
from bruhos.sim.kinematic import SimBody, World

PROMPTS = Path(__file__).with_name("prompts_en.toml")
NEAR_M = 0.15
FACING_DEG = 8.0


def load_prompts(path: str | Path = PROMPTS) -> list[dict[str, Any]]:
    with open(path, "rb") as f:
        return tomllib.load(f)["prompt"]


def check(expect: str, ep: Episode, world: World) -> bool:
    kind, _, arg = expect.partition(":")
    if kind == "event":
        return arg in world.events
    if kind == "holding":
        return world.holding is not None and world.holding.name == arg
    if kind in {"near", "facing"}:
        o = world.get(arg)
        if o is None:
            return False
        dist, bearing = world.relative(o)
        return ep.success and (dist <= NEAR_M if kind == "near" else abs(bearing) <= FACING_DEG)
    if kind == "success":
        return ep.success
    if kind == "stopped":
        return ep.end_reason == "stopped" and ep.success
    if kind == "no_plan":
        return not ep.plan and ep.end_reason == "no plan for instruction"
    raise ValueError(f"unknown expectation {expect!r}")


@dataclass
class PromptResult:
    text: str
    passed: bool
    end_reason: str
    plan: list[str]
    falls: int
    recovered: int
    failed_checks: list[str] = field(default_factory=list)


@dataclass
class EvalReport:
    brain: str
    fall_prob: float
    seed: int
    results: list[PromptResult]

    @property
    def completion_rate(self) -> float:
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def falls(self) -> int:
        return sum(r.falls for r in self.results)

    @property
    def recovered(self) -> int:
        return sum(r.recovered for r in self.results)

    @property
    def recovery_rate(self) -> float | None:
        return self.recovered / self.falls if self.falls else None

    def summary(self) -> dict[str, Any]:
        return {
            "brain": self.brain, "fall_prob": self.fall_prob, "seed": self.seed,
            "prompts": len(self.results), "passed": sum(r.passed for r in self.results),
            "completion_rate": round(self.completion_rate, 3),
            "falls": self.falls, "recovered": self.recovered,
            "recovery_rate": None if self.recovery_rate is None else round(self.recovery_rate, 3),
        }


def run_eval(make_brain: Callable[[], Planner], fall_prob: float = 0.0, seed: int = 0,
             prompts: list[dict[str, Any]] | None = None) -> EvalReport:
    results = []
    for i, p in enumerate(prompts or load_prompts()):
        overrides = dict(p.get("world", {}))
        world = World.tabletop(line_m=overrides.pop("line_m", 0.0))
        for k, v in overrides.items():
            setattr(world, k, v)
        body = SimBody(world, fall_prob=fall_prob, seed=seed * 1000 + i)
        brain = make_brain()
        ep = Runtime(body, brain, config=RuntimeConfig()).run(p["text"])
        failed = [e for e in p["expect"] if not check(e, ep, world)]
        recovered = sum(1 for s in ep.steps if s.kind == "recovery" and s.outcome["status"] == "ok")
        initial_fall = 1 if p.get("world", {}).get("fallen") else 0
        results.append(PromptResult(p["text"], not failed, ep.end_reason,
                                    [f"{c['skill']}" for c in ep.plan], ep.falls + initial_fall, recovered, failed))
    return EvalReport(brain=make_brain().name, fall_prob=fall_prob, seed=seed, results=results)
