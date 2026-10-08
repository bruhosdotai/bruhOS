"""The skill whitelist. Anything not registered here is rejected before it reaches the body."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from bruhos.body.base import Body
from bruhos.schemas import Outcome, SkillCall, Status


class SkillError(ValueError):
    pass


@dataclass(frozen=True)
class Arg:
    type: type
    required: bool = False
    choices: tuple[Any, ...] | None = None
    default: Any = None
    doc: str = ""


@dataclass(frozen=True)
class SkillSpec:
    name: str
    doc: str
    args: dict[str, Arg] = field(default_factory=dict)
    positional: str | None = None       # arg name used by shorthand like walk_toward(red_block)

    def describe(self) -> dict[str, Any]:
        return {
            "skill": self.name,
            "doc": self.doc,
            "args": {
                k: {"type": a.type.__name__, "required": a.required,
                    **({"choices": list(a.choices)} if a.choices else {}),
                    **({"default": a.default} if a.default is not None else {}),
                    **({"doc": a.doc} if a.doc else {})}
                for k, a in self.args.items()
            },
        }


@dataclass
class SkillContext:
    body: Body
    should_stop: Callable[[], bool] = lambda: False
    max_steps: int = 120


SkillFn = Callable[[SkillContext, dict[str, Any]], Outcome]

SKILLS: dict[str, SkillSpec] = {}
_IMPL: dict[str, SkillFn] = {}


def skill(spec: SkillSpec):
    def deco(fn: SkillFn) -> SkillFn:
        SKILLS[spec.name] = spec
        _IMPL[spec.name] = fn
        return fn
    return deco


def validate(call: SkillCall) -> SkillCall:
    """Return a normalized call (defaults filled, types coerced) or raise SkillError."""
    spec = SKILLS.get(call.skill)
    if spec is None:
        raise SkillError(f"unregistered skill: {call.skill!r}")
    unknown = set(call.args) - set(spec.args)
    if unknown:
        raise SkillError(f"{call.skill}: unknown args {sorted(unknown)}")
    out: dict[str, Any] = {}
    for k, a in spec.args.items():
        if k not in call.args:
            if a.required:
                raise SkillError(f"{call.skill}: missing required arg {k!r}")
            if a.default is not None:
                out[k] = a.default
            continue
        v = call.args[k]
        try:
            v = a.type(v)
        except (TypeError, ValueError) as e:
            raise SkillError(f"{call.skill}.{k}: expected {a.type.__name__}, got {v!r}") from e
        if a.choices and v not in a.choices:
            raise SkillError(f"{call.skill}.{k}: {v!r} not in {list(a.choices)}")
        out[k] = v
    return SkillCall(call.skill, out)


def run_skill(ctx: SkillContext, call: SkillCall) -> Outcome:
    call = validate(call)
    if ctx.should_stop() and call.skill != "stop":
        return Outcome(Status.STOPPED, "stop requested")
    return _IMPL[call.skill](ctx, call.args)


def catalog() -> list[dict[str, Any]]:
    return [SKILLS[k].describe() for k in sorted(SKILLS)]


def catalog_hash() -> str:
    blob = json.dumps(catalog(), sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def positional_args() -> dict[str, str]:
    return {k: s.positional for k, s in SKILLS.items() if s.positional}


from . import builtin  # noqa: E402,F401  (registers the v0.1 whitelist)

CATALOG = catalog()
