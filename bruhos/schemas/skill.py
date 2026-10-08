"""Skill call and outcome. The Brain emits SkillCalls; the Runtime returns Outcomes."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    OK = "ok"
    FAILED = "failed"
    FALLEN = "fallen"
    STOPPED = "stopped"
    REJECTED = "rejected"


@dataclass
class SkillCall:
    skill: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SkillCall":
        if not isinstance(d, dict) or not isinstance(d.get("skill"), str):
            raise ValueError(f"not a skill call: {d!r}")
        args = d.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError(f"args must be an object: {d!r}")
        return cls(skill=d["skill"], args=args)

    def __str__(self) -> str:
        inner = ", ".join(f"{k}={v}" for k, v in self.args.items())
        return f"{self.skill}({inner})"


@dataclass
class Outcome:
    status: Status
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is Status.OK

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "detail": self.detail, "data": self.data}


_CALL = re.compile(r"^\s*([a-z_][a-z0-9_]*)\s*(?:\((.*)\))?\s*$")


def parse_call(text: str, positional: dict[str, str] | None = None) -> SkillCall:
    """Parse planner shorthand such as ``walk_toward(red_block)`` or ``kick(foot=left)``.

    ``positional`` maps a skill name to the argument name used for a bare value.
    """
    m = _CALL.match(text)
    if not m:
        raise ValueError(f"cannot parse skill call: {text!r}")
    name, inner = m.group(1), (m.group(2) or "").strip()
    args: dict[str, Any] = {}
    if inner:
        for part in (p.strip() for p in inner.split(",")):
            if "=" in part:
                k, v = (s.strip() for s in part.split("=", 1))
            else:
                k = (positional or {}).get(name)
                if k is None:
                    raise ValueError(f"{name} takes no positional argument: {text!r}")
                v = part
            args[k] = _coerce(v.strip("'\""))
    return SkillCall(name, args)


def _coerce(v: str) -> Any:
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v
