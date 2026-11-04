"""Export the skill catalog as OpenAI tools and eliza-shaped action descriptors."""

from __future__ import annotations

from typing import Any

from bruhos.skills.registry import SKILLS, SkillSpec


def _json_schema_for(spec: SkillSpec) -> dict[str, Any]:
    props: dict[str, Any] = {}
    required: list[str] = []
    for name, arg in spec.args.items():
        t = "number" if arg.type in (int, float) else "string"
        if arg.type is int:
            t = "integer"
        prop: dict[str, Any] = {"type": t}
        if arg.doc:
            prop["description"] = arg.doc
        if arg.choices:
            prop["enum"] = list(arg.choices)
        if arg.default is not None:
            prop["default"] = arg.default
        props[name] = prop
        if arg.required:
            required.append(name)
    schema: dict[str, Any] = {"type": "object", "properties": props, "additionalProperties": False}
    if required:
        schema["required"] = required
    return schema


def skills_as_openai_tools() -> list[dict[str, Any]]:
    tools = []
    for name in sorted(SKILLS):
        spec = SKILLS[name]
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": spec.doc,
                "parameters": _json_schema_for(spec),
            },
        })
    return tools


def skills_as_eliza_actions() -> list[dict[str, Any]]:
    """Intersection shape both odiza and eliza plugins can load."""
    out = []
    for name in sorted(SKILLS):
        spec = SKILLS[name]
        similes = [name.replace("_", " ")]
        if name == "go":
            similes.extend(["walk forward", "turn left", "turn right", "move"])
        out.append({
            "name": name.upper(),
            "similes": similes,
            "description": spec.doc,
            "parameters": _json_schema_for(spec),
            "skill": name,
        })
    return out
