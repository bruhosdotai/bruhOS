"""Trace records: one Step per executed (or rejected) skill, grouped into an Episode."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Step:
    index: int
    t: float
    call: dict[str, Any]
    outcome: dict[str, Any]
    obs_before: dict[str, Any]
    obs_after: dict[str, Any] | None = None
    kind: str = "skill"            # "skill" | "recovery" | "rejected" | "stop"
    hash: str = ""

    def body(self) -> dict[str, Any]:
        """Everything that is hashed (the hash field itself excluded)."""
        return {
            "index": self.index, "t": self.t, "kind": self.kind,
            "call": self.call, "outcome": self.outcome,
            "obs_before": self.obs_before, "obs_after": self.obs_after,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.body(), "hash": self.hash}


@dataclass
class Episode:
    episode_id: str
    instruction: str
    brain: str
    body: str
    started_at: float
    plan: list[dict[str, Any]] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    success: bool = False
    recoveries: int = 0
    falls: int = 0
    ended_at: float | None = None
    end_reason: str = ""
    skillpack_hash: str = ""
    brain_version: str = ""
    trace_root: str = ""

    def header(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id, "instruction": self.instruction,
            "brain": self.brain, "brain_version": self.brain_version, "body": self.body,
            "started_at": self.started_at, "ended_at": self.ended_at,
            "plan": self.plan, "success": self.success, "end_reason": self.end_reason,
            "falls": self.falls, "recoveries": self.recoveries,
            "skillpack_hash": self.skillpack_hash, "trace_root": self.trace_root,
            "n_steps": len(self.steps),
        }
