"""Native (in-process tool loop) and plugin (remote eliza/odiza) harnesses."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bruhos.listen.chat import DEFAULT_SYSTEM, ChatReply, GenRate, clean_reply, gen_rate_from  # not via listen.__init__
from bruhos.schemas import SkillCall, parse_call
from bruhos.skills.dispatch import DispatchResult, SkillHub
from bruhos.skills.registry import SkillError, positional_args

from .tools import skills_as_openai_tools

Transport = Callable[[str, dict[str, str], bytes, float], bytes]

TOOL_SYSTEM = DEFAULT_SYSTEM + """

You control a small humanoid robot through tools (skills). Rules for tools:
- Call a tool when the user asks the robot to move or act (walk, turn, wave, kick, stop, grab, …).
- For open-space walking or turning with no object to approach, use go (direction forward|left|right, steps).
- Do not invent a walk_toward target when there is no object named.
- After tools run you will be asked for one short spoken sentence about what happened — do not narrate motion before tools run.
- If the user is only chatting or playing a game with no robot motion, reply in plain speech and call no tools.
"""

SPEAK_SYSTEM = (
    "You are Tony speaking out loud. Given the user request and skill outcomes, "
    "reply with ONE short spoken sentence (1–2 sentences). No markdown, no JSON, no role labels."
)

# response_format builds a grammar that starts with the assistant tag and then
# feeds that tag back into the sampler. TinyLlama's first token is a space glued
# to "<", so the sampler refuses it. A grammar on the request is not prefilled,
# so it constrains only the tokens the model still has to write.
_JSON_OBJECT_GRAMMAR = r"""
root ::= object
value ::= object | array | string | number | boolean | null
object ::= "{" space ( string ":" space value ("," space string ":" space value)* )? space "}"
array ::= "[" space ( value ("," space value)* )? space "]"
string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}))* "\""
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
boolean ::= "true" | "false"
null ::= "null"
space ::= " "*
"""

_SKILL_JSON = re.compile(r"\b([a-z_][a-z0-9_]*)\s*(\{[\s\S]*?\})", re.I)
_SKILL_CALL = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(([^)]*)\)", re.I)


@dataclass
class Turn:
    say: str
    results: list[DispatchResult] = field(default_factory=list)
    gen: GenRate | None = None

    @property
    def acted(self) -> bool:
        return bool(self.results)


@dataclass
class AgentConfig:
    harness: str = "native"          # native | plugin
    url: str | None = None           # plugin: http://host/bruhos/message
    body: str = "tonypi"             # ear body; not zerollama [body]=sim
    body_config: str | None = None   # path to tonypi.toml when merging
    max_steps: int = 6
    timeout: float = 120.0


class Harness(ABC):
    @abstractmethod
    def turn(self, text: str, history: list[dict[str, str]] | None = None) -> Turn: ...


def http_post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
    req = Request(url, data=body, headers=headers, method="POST")
    with urlopen(req, timeout=timeout) as r:
        return r.read()


def _http_error_body(e: HTTPError) -> str:
    fp = getattr(e, "fp", None)
    if fp is None:
        return ""
    try:
        raw = fp.read()
    except Exception:  # noqa: BLE001
        return ""
    if isinstance(raw, str):
        return raw
    return (raw or b"").decode("utf-8", errors="replace")


def _phase(msg: str) -> None:
    print(f"[ear] {msg}", flush=True)


def _calls_from_text(text: str) -> list[SkillCall]:
    """Recover skill calls when the model dumps them into prose instead of tools/JSON."""
    from bruhos.skills.registry import SKILLS

    known = set(SKILLS)
    found: list[SkillCall] = []
    seen: set[str] = set()

    def add(call: SkillCall) -> None:
        key = f"{call.skill}:{json.dumps(call.args, sort_keys=True)}"
        if key in seen:
            return
        seen.add(key)
        found.append(call)

    for m in re.finditer(r"\{[^{}]*\"plan\"\s*:\s*\[[\s\S]*?\][^{}]*\}", text):
        try:
            data = json.loads(m.group(0))
            for item in data.get("plan") or []:
                if isinstance(item, dict) and item.get("skill"):
                    add(SkillCall.from_dict(item))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    for m in _SKILL_JSON.finditer(text):
        name = m.group(1).lower()
        if name not in known:
            continue
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            continue
        if isinstance(args, dict):
            add(SkillCall(name, args))

    # { "wave": {} } and { "go": { "direction": "forward", "steps": 3 } }
    # The skill name is the object key, so the prefix scan above never sees it.
    decoder = json.JSONDecoder()
    cursor = 0
    while True:
        start = text.find("{", cursor)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        cursor = end
        if not isinstance(obj, dict) or not obj:
            continue
        keys = [str(k).lower() for k in obj]
        if any(k not in known for k in keys):
            continue
        for key, raw in zip(keys, obj.values()):
            if raw is None:
                args: dict = {}
            elif isinstance(raw, dict):
                args = raw
            else:
                continue
            add(SkillCall(key, args))

    pos = positional_args()
    for m in _SKILL_CALL.finditer(text):
        name = m.group(1).lower()
        if name not in known:
            continue
        try:
            add(parse_call(m.group(0), pos))
        except ValueError:
            pass

    return found


class NativeHarness(Harness):
    """Tool loop in-process: decide → run_skill → speak from outcomes."""

    def __init__(
        self,
        hub: SkillHub,
        chat: ChatReply,
        *,
        transport: Transport | None = None,
        temperature: float = 0.2,
    ):
        self.hub = hub
        self.chat = chat
        self.transport = transport or http_post
        self.temperature = temperature
        self.base_url = chat.base_url
        self.model = chat.model
        self.api_key = chat.api_key
        self.timeout = chat.timeout

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _complete(self, payload: dict[str, Any]) -> tuple[dict[str, Any], GenRate | None]:
        raw = self.transport(
            f"{self.base_url}/chat/completions",
            self._headers(),
            json.dumps(payload).encode(),
            self.timeout,
        )
        data = json.loads(raw)
        return data["choices"][0]["message"], gen_rate_from(data)

    def _decide_json_plan(
        self, text: str, history: list[dict[str, str]] | None
    ) -> tuple[str, list[SkillCall], GenRate | None]:
        system = (
            TOOL_SYSTEM
            + "\nReply with JSON only (no other text): "
            '{"say":"","plan":[{"skill":"go","args":{"direction":"forward","steps":3}}]}. '
            'For "go forward" / "walk forward" use skill go with direction forward. '
            "Empty plan means chat only; put the spoken reply in say. "
            "Never write skill calls as prose."
        )
        msgs: list[dict[str, Any]] = [{"role": "system", "content": system}]
        if history:
            msgs.extend(history)
        msgs.append({"role": "user", "content": text})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "temperature": self.temperature,
            "stream": False,
            "max_tokens": 160,
            "response_format": {"type": "json_object"},
        }
        try:
            msg, gen = self._complete(payload)
        except HTTPError as e:
            if e.code in (400, 404, 422) and "response_format" in payload:
                detail = _http_error_body(e)
                payload.pop("response_format", None)
                # The tag is already in the prompt. Constraining the new tokens
                # still gets a JSON plan. A server that rejected the format for
                # some other reason never sees this grammar.
                if "sampler" in detail:
                    payload["grammar"] = _JSON_OBJECT_GRAMMAR
                    try:
                        msg, gen = self._complete(payload)
                    except HTTPError:
                        payload.pop("grammar", None)
                        msg, gen = self._complete(payload)
                else:
                    msg, gen = self._complete(payload)
            else:
                raise
        content = msg.get("content") or "{}"
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        text_blob = str(content).strip()
        text_blob = re.sub(r"^```(?:json)?\s*|\s*```$", "", text_blob, flags=re.MULTILINE).strip()
        try:
            data = json.loads(text_blob)
        except json.JSONDecodeError:
            scraped = _calls_from_text(text_blob)
            if scraped:
                return "", scraped, None
            return clean_reply(text_blob), [], gen
        say = clean_reply(str(data.get("say") or ""))
        calls: list[SkillCall] = []
        for item in data.get("plan") or []:
            if isinstance(item, dict) and item.get("skill"):
                calls.append(SkillCall.from_dict(item))
            elif isinstance(item, str):
                from bruhos.brain.base import parse_plan
                try:
                    calls.extend(parse_plan([item]))
                except (ValueError, SkillError):
                    pass
        if not calls:
            calls = _calls_from_text(text_blob)
        return say, calls, gen if say and not calls else None

    def _decide(self, text: str, history: list[dict[str, str]] | None) -> tuple[str, list[SkillCall], GenRate | None]:
        # Prefer JSON plan on zerollama/qwen — tools are often ignored and the
        # model dumps "go{...}" into content (which used to become chat-only yap).
        say, calls, plan_gen = self._decide_json_plan(text, history)
        if calls:
            return say, calls, None
        scraped = _calls_from_text(say)
        if scraped:
            _phase(f"DECIDE scraped {len(scraped)} call(s) from text")
            return "", scraped, None

        msgs: list[dict[str, Any]] = [{"role": "system", "content": TOOL_SYSTEM}]
        if history:
            msgs.extend(history)
        msgs.append({"role": "user", "content": text})
        tools = skills_as_openai_tools()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "temperature": self.temperature,
            "stream": False,
            "tools": tools,
            "tool_choice": "auto",
        }
        try:
            msg, tool_gen = self._complete(payload)
        except HTTPError as e:
            if e.code not in (400, 404, 422):
                raise
            return say, [], plan_gen if say else None

        content = msg.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if name:
                calls.append(SkillCall(name, args if isinstance(args, dict) else {}))
        if not calls:
            calls = _calls_from_text(str(content))
            if calls:
                return "", calls, None
        out_say = clean_reply(str(content)) if not calls else say
        if out_say:
            return out_say, calls, tool_gen
        return say, calls, plan_gen if say else None

    def _speak_from_outcomes(self, text: str, results: list[DispatchResult]) -> str:
        summary = [
            {
                "skill": r.call.skill,
                "args": r.call.args,
                "status": r.outcome.status.value,
                "detail": r.outcome.detail,
            }
            for r in results
        ]
        msgs = [
            {"role": "system", "content": SPEAK_SYSTEM},
            {
                "role": "user",
                "content": json.dumps({"request": text, "outcomes": summary}),
            },
        ]
        payload = {
            "model": self.model,
            "messages": msgs,
            "temperature": 0.3,
            "stream": False,
        }
        msg, gen = self._complete(payload)
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in content
            )
        spoken = clean_reply(str(content))
        if spoken:
            self._reply_gen = gen
            return spoken
        self._reply_gen = None
        bits = [f"{r.call.skill}: {r.outcome.detail}" for r in results]
        return clean_reply("; ".join(bits) or "Done.")

    def turn(self, text: str, history: list[dict[str, str]] | None = None) -> Turn:
        _phase(f"DECIDE {text!r}")
        try:
            say, calls, gen = self._decide(text, history)
        except Exception as e:  # noqa: BLE001
            _phase(f"DECIDE_FAIL {e}")
            try:
                return Turn(say=self.chat.reply(text, history=history), gen=self.chat.last_gen)
            except Exception as e2:  # noqa: BLE001
                return Turn(say=f"I could not think: {e2}")

        if not calls:
            if not say:
                say = self.chat.reply(text, history=history)
                gen = self.chat.last_gen
            _phase("DECIDE chat-only")
            return Turn(say=say, gen=gen)

        results: list[DispatchResult] = []
        for call in calls:
            _phase(f"ACT {call}")
            result = self.hub.dispatch(call, brain="native")
            results.append(result)
            _phase(
                f"ACT_DONE {result.call.skill} status={result.outcome.status.value} "
                f"{result.outcome.detail!r}"
            )

        self._reply_gen = None
        try:
            spoken = self._speak_from_outcomes(text, results)
        except Exception as e:  # noqa: BLE001
            _phase(f"SAY_FAIL {e}")
            spoken = "; ".join(r.outcome.detail for r in results) or "Done."
            self._reply_gen = None
        _phase(f"SAY {spoken!r}")
        return Turn(say=spoken, results=results, gen=self._reply_gen)


class PluginHarness(Harness):
    """POST to odiza/eliza plugin route; plugin actions call Pi /v1/skills/run."""

    def __init__(self, url: str, *, timeout: float = 120.0, transport: Transport | None = None):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.transport = transport or http_post

    def turn(self, text: str, history: list[dict[str, str]] | None = None) -> Turn:
        _phase(f"DECIDE plugin {self.url}")
        payload = {"text": text, "roomId": "bruhos-ear", "history": history or []}
        try:
            raw = self.transport(
                self.url if self.url.endswith("/bruhos/message") else f"{self.url}/bruhos/message",
                {"Content-Type": "application/json"},
                json.dumps(payload).encode(),
                self.timeout,
            )
        except (HTTPError, URLError, TimeoutError, OSError) as e:
            _phase(f"DECIDE_FAIL plugin {e}")
            return Turn(say="I could not reach the agent.")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return Turn(say="The agent returned a bad reply.")
        say = clean_reply(str(data.get("text") or data.get("say") or ""))
        if not say:
            say = "Okay."
        _phase(f"SAY {say!r}")
        return Turn(say=say)


def make_harness(
    cfg: AgentConfig,
    hub: SkillHub | None,
    chat: ChatReply,
) -> Harness:
    name = (cfg.harness or "native").lower()
    if name == "plugin":
        if not cfg.url:
            raise ValueError("agent.harness=plugin needs agent.url")
        return PluginHarness(cfg.url, timeout=cfg.timeout)
    if hub is None:
        raise ValueError("native harness needs a SkillHub / body")
    return NativeHarness(hub, chat)
