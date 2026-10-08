"""OpenAI-compatible chat-completions planner (OpenAI, vLLM, llama.cpp server, Ollama, ...)."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Callable

from bruhos.brain.base import Planner, parse_plan
from bruhos.schemas import Observation, SkillCall

SYSTEM_PROMPT = """You are the planner of bruhOS, a skill operating system on a small humanoid robot.
You never control joints or servos. You only choose skills from the catalog below.
Reply with JSON only: {"plan": [{"skill": "<name>", "args": {...}}, ...]}.
Rules:
- Use only skill names and args that exist in the catalog.
- Targets are object names from the observation, e.g. "red_block", "box", "yellow_ball".
- Keep plans short. If the instruction cannot be done with these skills, return {"plan": []}.
- If the instruction is "stop" or similar, return exactly [{"skill": "stop", "args": {}}].
Skill catalog:
"""

Transport = Callable[[str, dict[str, str], bytes, float], bytes]


def http_post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


class OpenAICompatPlanner(Planner):
    name = "openai"

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", timeout: float = 20.0, json_mode: bool = True,
                 transport: Transport | None = None):
        self.base_url = (base_url or os.environ.get("BRUHOS_LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("BRUHOS_LLM_MODEL") or "gpt-4o-mini"
        self.api_key = api_key if api_key is not None else os.environ.get(api_key_env, "")
        self.timeout = timeout
        self.json_mode = json_mode
        self.transport = transport or http_post
        self.version = f"{self.name}:{self.model}"

    def user_content(self, instruction: str, obs: Observation) -> Any:
        return json.dumps({"instruction": instruction, "observation": obs.to_dict()})

    def messages(self, instruction: str, obs: Observation, catalog: list[dict[str, Any]],
                 history: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT + json.dumps(catalog, indent=1)}]
        if history:
            msgs.append({"role": "user", "content": "Previous attempt failed. History: " + json.dumps(history)})
        msgs.append({"role": "user", "content": self.user_content(instruction, obs)})
        return msgs

    def _complete(self, msgs: list[dict[str, Any]]) -> list[SkillCall]:
        payload: dict[str, Any] = {"model": self.model, "messages": msgs, "temperature": 0}
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        raw = self.transport(f"{self.base_url}/chat/completions", headers, json.dumps(payload).encode(), self.timeout)
        content = json.loads(raw)["choices"][0]["message"]["content"]
        return parse_plan(content)

    def plan(self, instruction: str, obs: Observation, catalog: list[dict[str, Any]]) -> list[SkillCall]:
        return self._complete(self.messages(instruction, obs, catalog))

    def replan(self, instruction: str, obs: Observation, catalog: list[dict[str, Any]],
               history: list[dict[str, Any]]) -> list[SkillCall]:
        return self._complete(self.messages(instruction, obs, catalog, history))
