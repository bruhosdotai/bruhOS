"""Vision: look through zerollama while hearing owns qwen2.5:14b.

POST ``/v1/chat/completions`` with ``ministral-3:3b`` and ``keep_alive: 30m``.
Never load the reply model, never open ``/v1/realtime``, never send
``keep_alive: 0`` for the ear's model. If ministral is missing, try
``lfm3-omni:latest``.
"""

from __future__ import annotations

import json
import re
import urllib.error
from typing import Any, Callable

from bruhos.scan.vision import http_post, normalize_objects

Transport = Callable[[str, dict[str, str], bytes, float], bytes]

EYE_MODEL = "ministral-3:3b"
EYE_FALLBACK = "lfm3-omni:latest"
KEEP_ALIVE = "30m"
SEE_TEXT = "What is in front of the camera? One sentence."
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def parse_understand(content: str) -> tuple[str, list[str]]:
    text = str(content or "").strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    data: Any = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                data = None
    if isinstance(data, dict):
        caption = " ".join(str(data.get("caption") or data.get("text") or "").split())[:240]
        objs = data.get("objects")
        if not isinstance(objs, list):
            objs = []
        if caption:
            return caption, normalize_objects(objs)
    caption = " ".join(text.split())[:240]
    return caption, []


def _payload(jpeg: bytes, model: str) -> dict[str, Any]:
    import base64

    return {
        "model": model,
        "keep_alive": KEEP_ALIVE,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SEE_TEXT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
                        },
                    },
                ],
            },
        ],
    }


def _model_missing(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in (404, 422):
            return True
        try:
            body = exc.read().decode("utf-8", "replace").lower()
        except Exception:  # noqa: BLE001
            body = str(exc).lower()
        return "not found" in body or "unknown model" in body or "model" in body and "404" in body
    text = str(exc).lower()
    return "not found" in text or "unknown model" in text


def understand_frame(
    jpeg: bytes,
    *,
    base_url: str,
    model: str = EYE_MODEL,
    timeout: float = 20.0,
    transport: Transport | None = None,
    fallback: str = EYE_FALLBACK,
) -> tuple[str, list[str]]:
    post = transport or http_post
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    used = model or EYE_MODEL
    try:
        raw = post(url, headers, json.dumps(_payload(jpeg, used)).encode(), timeout)
    except Exception as e:  # noqa: BLE001
        if fallback and used != fallback and _model_missing(e):
            print(f"[eye] model {used} missing; trying {fallback}", flush=True)
            used = fallback
            raw = post(url, headers, json.dumps(_payload(jpeg, used)).encode(), timeout)
        else:
            raise
    content = json.loads(raw)["choices"][0]["message"]["content"]
    return parse_understand(str(content))
