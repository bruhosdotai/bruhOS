"""OpenAI-compatible chat for spoken Q&A (not the skill planner)."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable
from urllib.request import Request, urlopen

Transport = Callable[[str, dict[str, str], bytes, float], bytes]

DEFAULT_SYSTEM = """You are Tony, a humanoid robot running bruhOS, talking out loud with one person.

Voice rules:
- Speak as Tony only. One voice. Never write "Assistant:", "User:", "Prosecutor:", or stage directions for other people unless you are briefly quoting them in one short line.
- Plain spoken English. No markdown, no bullet lists, no JSON, no emoji.
- Keep turns short enough to say aloud (usually 1–4 sentences). When running a game, you may use a bit more, but still speakable.

Role / game rules (critical):
- If the user assigns you a role (Game Master, judge, announcer, narrator, etc.), accept that ONE role and keep it for the rest of the conversation until they change it.
- Do not assign the user a conflicting role. Do not say both "you'll be the judge" and then speak as the judge yourself.
- As Game Master / announcer: you facilitate. You narrate the scene, call on the next speaker, keep score of where the trial is, and ask the user what they say or do next. You are not every character at once.
- If roles are unclear, ask one clarifying question before continuing the scene.
- Remember facts the user already stated (the case, who is who) from earlier turns.

Default (no game): be helpful and brief. Do not invent robot sensor readings."""


_LEAK_PREFIX = re.compile(
    r"^\s*(Assistant|User|System|Prosecutor|Judge|Defendant|Narrator|Game Master)\s*:\s*",
    re.I,
)
_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)


def clean_reply(text: str) -> str:
    """Strip leaked role labels / emoji that break TTS and confuse roles."""
    t = text.replace("\r\n", "\n").strip()
    # drop repeated "Assistant:" style lines; keep content after the label
    lines = []
    for line in t.split("\n"):
        line = _LEAK_PREFIX.sub("", line).strip()
        if line:
            lines.append(line)
    t = " ".join(lines) if len(lines) > 1 and all(len(x) < 120 for x in lines) else "\n".join(lines)
    t = _EMOJI.sub("", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip(" \n-")


def http_post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
    req = Request(url, data=body, headers=headers, method="POST")
    with urlopen(req, timeout=timeout) as r:
        return r.read()


class ChatReply:
    """``POST {base_url}/chat/completions`` → assistant text."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 120.0,
        system: str = DEFAULT_SYSTEM,
        transport: Transport | None = None,
        temperature: float = 0.35,
    ):
        self.base_url = (base_url or os.environ.get("BRUHOS_LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("BRUHOS_LLM_MODEL") or "gpt-4o-mini"
        self.api_key = api_key if api_key is not None else os.environ.get(api_key_env, "")
        self.timeout = timeout
        self.system = system
        self.transport = transport or http_post
        self.temperature = temperature

    def reply(self, user_text: str, history: list[dict[str, str]] | None = None) -> str:
        msgs: list[dict[str, Any]] = [{"role": "system", "content": self.system}]
        if history:
            msgs.extend(history)
        msgs.append({"role": "user", "content": user_text})
        payload = {
            "model": self.model,
            "messages": msgs,
            "temperature": self.temperature,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        raw = self.transport(
            f"{self.base_url}/chat/completions",
            headers,
            json.dumps(payload).encode(),
            self.timeout,
        )
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        return clean_reply(str(content))
