"""OpenAI-compatible chat-completions planner (OpenAI, vLLM, llama.cpp server, Ollama, ...)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable

from bruhos._net import require_secure
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


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse every redirect. urllib would replay the Authorization header to any origin in Location,
    and a redirected POST becomes a GET without the body anyway. Point base_url at the final endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, f"redirect to {newurl} refused (brain calls never follow "
                                     "redirects; set base_url to the final endpoint)", headers, fp)


_opener = urllib.request.build_opener(_NoRedirect)


MAX_RESPONSE_BYTES = 4 << 20
PLACEHOLDER_KEYS = ("", "EMPTY")


def http_post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with _opener.open(req, timeout=timeout) as r:
        data = r.read(MAX_RESPONSE_BYTES + 1)
    if len(data) > MAX_RESPONSE_BYTES:
        raise ValueError(f"brain response larger than {MAX_RESPONSE_BYTES} bytes")
    return data


class OpenAICompatPlanner(Planner):
    name = "openai"

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", timeout: float = 20.0, json_mode: bool = True,
                 transport: Transport | None = None, allow_insecure: bool = False):
        self.base_url = (base_url or os.environ.get("BRUHOS_LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("BRUHOS_LLM_MODEL") or "gpt-4o-mini"
        self.api_key = api_key if api_key is not None else os.environ.get(api_key_env, "")
        if self.api_key not in PLACEHOLDER_KEYS:
            require_secure(self.base_url, "the API key", allow_insecure)
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

    def extra_payload(self, catalog: list[dict[str, Any]]) -> dict[str, Any]:
        """Server-specific request fields merged into the payload (subclasses override)."""
        return {}

    def _complete(self, msgs: list[dict[str, Any]], catalog: list[dict[str, Any]]) -> list[SkillCall]:
        payload: dict[str, Any] = {"model": self.model, "messages": msgs, "temperature": 0}
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.extra_payload(catalog))
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        raw = self.transport(f"{self.base_url}/chat/completions", headers, json.dumps(payload).encode(), self.timeout)
        try:
            content = json.loads(raw)["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            # A response without a chat choice is not a plan: raise ValueError so the
            # runtime records a rejected step instead of a bare "brain error" with no
            # step at all (an error payload, empty choices, or a choice without
            # content all land here; invalid JSON already raises ValueError).
            raise ValueError("brain response has no chat choice content") from e
        return parse_plan(content)

    def plan(self, instruction: str, obs: Observation, catalog: list[dict[str, Any]]) -> list[SkillCall]:
        return self._complete(self.messages(instruction, obs, catalog), catalog)

    def replan(self, instruction: str, obs: Observation, catalog: list[dict[str, Any]],
               history: list[dict[str, Any]]) -> list[SkillCall]:
        return self._complete(self.messages(instruction, obs, catalog, history), catalog)
