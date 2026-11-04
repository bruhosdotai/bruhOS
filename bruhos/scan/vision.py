"""OpenAI-compatible multi-image object listing (zerollama / vLLM / Ollama)."""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Callable

Transport = Callable[[str, dict[str, str], bytes, float], bytes]

SYSTEM = (
    "You inventory objects visible across several camera views of the same scene. "
    "Reply with JSON only: {\"objects\": [\"...\"]}. "
    "Use short lowercase names (optional one adjective, e.g. \"red cup\"). "
    "Merge the same object seen in multiple views into one entry. "
    "Pose labels include quality hints (score/lap); trust sharper, better-lit views more "
    "when views disagree. Ignore blurry, black, or empty frames. "
    "Ignore wall, floor, ceiling, sky, ground, and the robot's own body. "
    "If nothing notable is visible, return {\"objects\": []}."
)

DENYLIST = frozenset({
    "wall", "walls", "floor", "ceiling", "sky", "ground", "background",
    "robot", "robot body", "arm", "hand", "servo", "camera", "table surface",
})

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def http_post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def normalize_objects(items: list[Any]) -> list[str]:
    """Dedupe case-insensitively; drop denylisted / empty labels; keep first casing."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        if isinstance(raw, dict):
            raw = raw.get("name") or raw.get("label") or raw.get("object") or ""
        if not isinstance(raw, str):
            continue
        label = " ".join(raw.strip().split())
        if not label:
            continue
        key = label.casefold()
        if key in DENYLIST or key in seen:
            continue
        seen.add(key)
        out.append(label.casefold())
    return sorted(out)


def parse_objects_payload(content: str) -> list[str]:
    text = content.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return []
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []
    if isinstance(data, list):
        return normalize_objects(data)
    if isinstance(data, dict):
        objs = data.get("objects")
        if objs is None:
            objs = data.get("items") or data.get("objects_of_interest") or []
        if isinstance(objs, list):
            return normalize_objects(objs)
    return []


class VisionLister:
    def __init__(
        self,
        base_url: str | None = None,
        model: str = "ministral-3:3b",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 180.0,
        json_mode: bool = True,
        transport: Transport | None = None,
        merge_model: str | None = "qwen2.5:7b",
    ):
        self.base_url = (
            base_url or os.environ.get("BRUHOS_LLM_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model
        self.api_key = api_key if api_key is not None else os.environ.get(api_key_env, "")
        self.timeout = timeout
        self.json_mode = json_mode
        self.transport = transport or http_post
        self.merge_model = merge_model

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, payload: dict[str, Any]) -> bytes:
        return self.transport(
            f"{self.base_url}/chat/completions",
            self._headers(),
            json.dumps(payload).encode(),
            self.timeout,
        )

    def _complete(self, messages: list[dict[str, Any]], model: str | None = None) -> str:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": 0,
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            raw = self._post(payload)
        except urllib.error.HTTPError as e:
            if self.json_mode and e.code in (400, 404, 422):
                payload.pop("response_format", None)
                raw = self._post(payload)
            else:
                raise
        return json.loads(raw)["choices"][0]["message"]["content"]

    def list_objects(self, images: list[tuple[str, bytes]]) -> list[str]:
        if not images:
            return []
        try:
            return self._list_multi(images)
        except urllib.error.HTTPError as e:
            if e.code not in (400, 404, 413, 422):
                raise
            return self._list_sequential_and_merge(images)

    def _user_content(self, images: list[tuple[str, bytes]]) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                f"These are {len(images)} quality-filtered views from a small humanoid's "
                "head camera (labels include score/lap). Prefer clearer views. "
                "List distinct notable objects in the scene."
            ),
        }]
        for name, jpeg in images:
            b64 = base64.b64encode(jpeg).decode()
            parts.append({"type": "text", "text": f"pose={name}"})
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        return parts

    def _list_multi(self, images: list[tuple[str, bytes]]) -> list[str]:
        content = self._complete([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": self._user_content(images)},
        ])
        return parse_objects_payload(content)

    def _list_one(self, name: str, jpeg: bytes) -> list[str]:
        b64 = base64.b64encode(jpeg).decode()
        user = [
            {"type": "text", "text": f"pose={name}. List notable objects in this single view."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]
        content = self._complete([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ])
        return parse_objects_payload(content)

    def _list_sequential_and_merge(self, images: list[tuple[str, bytes]]) -> list[str]:
        per_view: dict[str, list[str]] = {}
        for name, jpeg in images:
            per_view[name] = self._list_one(name, jpeg)
        flat = normalize_objects([o for objs in per_view.values() for o in objs])
        if not self.merge_model or len(flat) <= 1:
            return flat
        prompt = (
            "Merge these per-view object lists from the same scene into one deduplicated "
            "JSON object {\"objects\": [...]}. Synonyms for the same thing become one entry.\n"
            + json.dumps(per_view)
        )
        try:
            content = self._complete(
                [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                model=self.merge_model,
            )
            merged = parse_objects_payload(content)
            return merged or flat
        except Exception:
            return flat
