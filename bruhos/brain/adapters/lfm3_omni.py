"""Liquid LFM3-Omni planner on the robot itself: LFM3-Omni-2B on a Raspberry Pi 5 with llama.cpp:

    llama-server -hf LiquidAI/LFM3-Omni-2B-GGUF:Q4_0 --jinja --swa-full -ub 64 -cms 64 \\
        -t 3 -tb 4 -c 8192 --temp 0 --port 8080

Same schema and frame handling as the Qwen-VL adapter. What a ~2B model needs on top:

- Thinking is off (``chat_template_kwargs.enable_thinking=false``). LFM3 thinks by default, which
  turns a plan of a few seconds into ~50 s on a Pi 5.
- The reply is constrained by a JSON schema built from the skill catalog, so skill names, arg names
  and arg choices outside the whitelist cannot be generated at all. The catalog is not changed.
- A short plain-text prompt (object names, distances, holding) instead of the full observation JSON,
  and a few worked examples as chat turns. They use objects the sim and the eval never contain.

``--swa-full`` keeps the system prompt (the skill catalog) cached across requests; without it the
sliding-window layers drop the cache and every plan re-reads the whole prompt.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from bruhos.schemas import Observation

from .qwen_vl import QwenVLPlanner

_JSON_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}


def _ask(objects: list[tuple[str, float]], instruction: str, holding: str | None = None) -> str:
    seen = [{"name": n, "distance_m": d} for n, d in objects]
    return f"Observation: {json.dumps({'objects': seen, 'holding': holding})}\nInstruction: {instruction}"


def _plan(*calls: tuple[str, dict[str, Any]]) -> str:
    return json.dumps({"plan": [{"skill": s, "args": a} for s, a in calls]})


EXAMPLES = [
    (_ask([("orange_block", 0.7), ("box", 1.0)], "Put the orange cube in the bin."),
     _plan(("detect", {"target": "orange_block"}), ("walk_toward", {"target": "orange_block"}), ("grab", {}),
           ("walk_toward", {"target": "box"}), ("release", {}))),
    (_ask([("box", 0.9), ("purple_ball", 0.5)], "Bring the purple ball to the box."),
     _plan(("detect", {"target": "purple_ball"}), ("walk_toward", {"target": "purple_ball"}), ("grab", {}),
           ("walk_toward", {"target": "box"}), ("release", {}))),
    (_ask([("box", 0.9)], "Kick the purple ball with your left foot."),
     _plan(("detect", {"target": "purple_ball"}), ("walk_toward", {"target": "purple_ball", "stop_distance_m": 0.12}),
           ("kick", {"foot": "left"}))),
    (_ask([("box", 0.9)], "Pick up the white brick."),
     _plan(("detect", {"target": "white_block"}), ("walk_toward", {"target": "white_block"}), ("grab", {}))),
    (_ask([("box", 0.9)], "Sing a song."), _plan()),
]


def plan_schema(catalog: list[dict[str, Any]]) -> dict[str, Any]:
    """JSON schema for ``{"plan": [...]}`` that only admits calls described by ``catalog``."""
    calls = []
    for spec in catalog:
        props: dict[str, Any] = {}
        for name, a in spec["args"].items():
            p: dict[str, Any] = {"type": _JSON_TYPES.get(a["type"], "string")}
            if "choices" in a:
                p["enum"] = a["choices"]
            props[name] = p
        calls.append({
            "type": "object",
            "properties": {
                "skill": {"const": spec["skill"]},
                "args": {"type": "object", "properties": props, "additionalProperties": False,
                         "required": [k for k, a in spec["args"].items() if a["required"]]},
            },
            "required": ["skill", "args"],
            "additionalProperties": False,
        })
    return {"type": "object", "properties": {"plan": {"type": "array", "items": {"anyOf": calls}}},
            "required": ["plan"], "additionalProperties": False}


class LFM3OmniPlanner(QwenVLPlanner):
    name = "lfm3_omni"

    def __init__(self, base_url: str = "http://127.0.0.1:8080/v1", model: str = "LiquidAI/LFM3-Omni-2B-GGUF",
                 constrain: bool = True, timeout: float = 60.0, **kw: Any):
        super().__init__(base_url=base_url, model=model, timeout=timeout, **kw)
        self.constrain = constrain

    def user_content(self, instruction: str, obs: Observation) -> Any:
        text = _ask([(o.name, o.distance_m) for o in obs.objects], instruction, obs.holding)
        jpeg = self.frame_provider() if self.frame_provider else None
        if not jpeg:
            return text
        url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        return [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": text}]

    def messages(self, instruction: str, obs: Observation, catalog: list[dict[str, Any]],
                 history: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        msgs = super().messages(instruction, obs, catalog, history)
        shots = [m for ask, plan in EXAMPLES
                 for m in ({"role": "user", "content": ask}, {"role": "assistant", "content": plan})]
        return msgs[:1] + shots + msgs[1:]

    def extra_payload(self, catalog: list[dict[str, Any]]) -> dict[str, Any]:
        extra: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}
        if self.constrain and catalog:
            extra["response_format"] = {"type": "json_schema",
                                        "json_schema": {"name": "plan", "schema": plan_schema(catalog)}}
        return extra
