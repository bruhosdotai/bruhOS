"""OpenAI-compatible chat-completions planner (OpenAI, vLLM, llama.cpp server, Ollama, ...)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable

from bruhos._net import require_secure
from bruhos.brain.base import IntegrationError, Planner, parse_plan
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


def completion_content(raw: bytes, model: str) -> str:
    """Pull the assistant text out of an OpenAI-compatible reply, or say why not.

    A 200 does not imply a completion: gateways and vLLM answer with
    ``{"detail": ...}`` on an upstream error, a tool-call-only or length-capped
    reply carries ``content: null`` or ``""``, and a proxy can return an empty
    ``choices`` list. Reading the envelope as ``["choices"][0]["message"]
    ["content"]`` turned each of those into ``KeyError('choices')`` (or
    ``ValueError: plan must be a list, got NoneType``), which ``Runtime`` reports
    as ``brain error: 'choices'`` or, worse, as a rejected plan -- a policy
    violation the brain never committed. Name the failure here instead.
    """
    where = f"brain {model!r}"
    try:
        body = json.loads(raw)
    except ValueError as e:
        raise IntegrationError(f"{where} returned a body that is not JSON ({e}): {raw[:200]!r}") from e
    if not isinstance(body, dict):
        raise IntegrationError(f"{where} returned {type(body).__name__}, not a completion object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        detail = body.get("detail") or body.get("error") or body
        raise IntegrationError(f"{where} answered without any choices (body: {str(detail)[:200]})")
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise IntegrationError(
            f"{where} returned no content (finish_reason={first.get('finish_reason')!r})")
    return content


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
        content = completion_content(raw, self.model)
        return parse_plan(content)

    def plan(self, instruction: str, obs: Observation, catalog: list[dict[str, Any]]) -> list[SkillCall]:
        return self._complete(self.messages(instruction, obs, catalog), catalog)

    def replan(self, instruction: str, obs: Observation, catalog: list[dict[str, Any]],
               history: list[dict[str, Any]]) -> list[SkillCall]:
        return self._complete(self.messages(instruction, obs, catalog, history), catalog)
