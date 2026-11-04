"""Native harness tool loop + go skill + SkillHub dispatch."""

from __future__ import annotations

import json

from bruhos.agent.harness import NativeHarness, PluginHarness, Turn
from bruhos.agent.tools import skills_as_eliza_actions, skills_as_openai_tools
from bruhos.listen.chat import ChatReply
from bruhos.schemas import SkillCall
from bruhos.skills import SkillHub, validate
from bruhos.skills.builtin import GO_MAX_STEPS
from bruhos.sim.kinematic import SimBody


def test_go_skill_cap_and_forward():
    body = SimBody(fall_prob=0.0, seed=0)
    hub = SkillHub(body)
    # steps requested above cap are clamped inside the skill after validate... 
    # validate does not clamp; skill does. Pass steps=20 via validate which accepts int.
    call = validate(SkillCall("go", {"direction": "forward", "steps": 20}))
    # Arg has no max in validate — skill clamps
    assert call.args["steps"] == 20
    result = hub.dispatch(SkillCall("go", {"direction": "forward", "steps": 20}))
    assert result.outcome.ok
    assert result.outcome.data["steps"] == GO_MAX_STEPS
    assert result.outcome.data["direction"] == "forward"


def test_go_rejects_back():
    from bruhos.skills import SkillError
    import pytest

    with pytest.raises(SkillError):
        validate(SkillCall("go", {"direction": "back"}))


def test_openai_tools_include_go():
    names = {t["function"]["name"] for t in skills_as_openai_tools()}
    assert "go" in names
    assert "walk_toward" in names


def test_eliza_actions_shape():
    acts = skills_as_eliza_actions()
    go = next(a for a in acts if a["skill"] == "go")
    assert go["name"] == "GO"
    assert "parameters" in go and "similes" in go


def test_native_harness_tool_then_speak():
    body = SimBody(fall_prob=0.0, seed=1)
    hub = SkillHub(body)
    chat = ChatReply(base_url="http://unused/v1", model="test", transport=lambda *a: b"{}")

    n = {"i": 0}

    def transport(url, headers, body_b, timeout):
        n["i"] += 1
        if n["i"] == 1:
            # JSON plan decide (preferred path)
            return json.dumps({
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "say": "",
                            "plan": [{"skill": "go", "args": {"direction": "forward", "steps": 2}}],
                        }),
                    }
                }]
            }).encode()
        # speak-from-outcomes
        return json.dumps({
            "choices": [{"message": {"content": "Walking forward."}}]
        }).encode()

    harness = NativeHarness(hub, chat, transport=transport)
    turn = harness.turn("walk forward please")
    assert isinstance(turn, Turn)
    assert turn.acted
    assert turn.results[0].call.skill == "go"
    assert turn.results[0].outcome.ok
    assert turn.say == "Walking forward."


def test_native_harness_chat_only():
    body = SimBody(fall_prob=0.0, seed=2)
    hub = SkillHub(body)
    chat = ChatReply(base_url="http://unused/v1", model="test", transport=lambda *a: b"{}")

    def transport(url, headers, body_b, timeout):
        return json.dumps({
            "choices": [{"message": {"content": "Sure, let's play.", "tool_calls": []}}]
        }).encode()

    harness = NativeHarness(hub, chat, transport=transport)
    turn = harness.turn("tell me a joke as game master")
    assert not turn.acted
    assert "play" in turn.say.lower() or "Sure" in turn.say


def test_plugin_harness_unreachable():
    h = PluginHarness("http://127.0.0.1:9", timeout=0.5)

    def boom(url, headers, body, timeout):
        raise TimeoutError("nope")

    h.transport = boom
    turn = h.turn("walk forward")
    assert "could not reach" in turn.say.lower()


def test_scrape_go_blob_from_yap():
    from bruhos.agent.harness import _calls_from_text

    # exact shape qwen dumped into TTS last time
    blob = 'go{ "direction": "forward", "steps": 3 } I move forward a few steps. Now what?'
    calls = _calls_from_text(blob)
    assert len(calls) == 1
    assert calls[0].skill == "go"
    assert calls[0].args["direction"] == "forward"
    assert calls[0].args["steps"] == 3


def test_native_scrapes_when_json_plan_is_prose():
    body = SimBody(fall_prob=0.0, seed=3)
    hub = SkillHub(body)
    chat = ChatReply(base_url="http://unused/v1", model="test", transport=lambda *a: b"{}")

    def transport(url, headers, body_b, timeout):
        # first call is JSON plan — model yaps skill into content instead
        return json.dumps({
            "choices": [{
                "message": {
                    "content": 'go{ "direction": "forward", "steps": 2 } okay',
                }
            }]
        }).encode()

    harness = NativeHarness(hub, chat, transport=transport)
    # speak-from-outcomes will hit transport again — provide a second response via counter
    n = {"i": 0}
    raw = transport

    def transport2(url, headers, body_b, timeout):
        n["i"] += 1
        if n["i"] == 1:
            return raw(url, headers, body_b, timeout)
        return json.dumps({
            "choices": [{"message": {"content": "Stepping forward."}}]
        }).encode()

    harness.transport = transport2
    turn = harness.turn("Go forward.")
    assert turn.acted
    assert turn.results[0].call.skill == "go"
    assert turn.say == "Stepping forward."
