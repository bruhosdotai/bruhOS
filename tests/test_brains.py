import json

import pytest

from bruhos.brain import IntegrationError, make_planner, parse_plan
from bruhos.brain.adapters.groot_s2 import GR00TS2Planner
from bruhos.brain.adapters.lerobot_policy import LeRobotPolicyPlanner
from bruhos.brain.adapters.openai import OpenAICompatPlanner
from bruhos.brain.adapters.qwen_vl import QwenVLPlanner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody

PLAN = ["detect", "walk_toward(red_block)", "grab", "walk_toward(box)", "release"]
PLAN_JSON = {"plan": [{"skill": "detect", "args": {}},
                      {"skill": "walk_toward", "args": {"target": "red_block"}},
                      {"skill": "grab", "args": {}},
                      {"skill": "walk_toward", "args": {"target": "box"}},
                      {"skill": "release", "args": {}}]}


def fake_chat(captured):
    def transport(url, headers, body, timeout):
        captured.append((url, headers, json.loads(body)))
        return json.dumps({"choices": [{"message": {"content": json.dumps(PLAN_JSON)}}]}).encode()
    return transport


def _run(brain):
    body = SimBody()
    ep = Runtime(body, brain).run("Put the red one in the box.")
    return ep, body


@pytest.mark.parametrize("brain", [
    lambda: make_planner("rules"),
    lambda: OpenAICompatPlanner(base_url="http://x/v1", api_key="k", transport=fake_chat([])),
    lambda: QwenVLPlanner(transport=fake_chat([])),
    lambda: GR00TS2Planner(reasoner=lambda frame, text, obs: PLAN),
    lambda: LeRobotPolicyPlanner(transport=lambda *a: json.dumps(PLAN_JSON).encode()),
])
def test_swapping_brain_keeps_skill_names(brain):
    ep, body = _run(brain())
    assert ep.success, ep.end_reason
    assert [c["skill"] for c in ep.plan] == ["detect", "walk_toward", "grab", "walk_toward", "release"]
    assert "in_box:red_block" in body.world.events


def test_openai_request_shape():
    cap = []
    OpenAICompatPlanner(base_url="http://h/v1", model="m", api_key="k", transport=fake_chat(cap)).plan(
        "Wave.", SimBody().observe(), [])
    url, headers, payload = cap[0]
    assert url == "http://h/v1/chat/completions"
    assert headers["Authorization"] == "Bearer k"
    assert payload["model"] == "m" and payload["temperature"] == 0
    assert payload["response_format"] == {"type": "json_object"}


def test_qwen_vl_attaches_frame():
    cap = []
    QwenVLPlanner(transport=fake_chat(cap), frame_provider=lambda: b"\xff\xd8jpeg").plan("Wave.", SimBody().observe(), [])
    content = cap[0][2]["messages"][-1]["content"]
    assert content[0]["type"] == "image_url" and content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_groot_joint_chunk_is_integration_error():
    chunk = [[0.01 * j for j in range(16)] for _ in range(8)]       # 8 x 16-servo action chunk
    with pytest.raises(IntegrationError):
        GR00TS2Planner(reasoner=lambda *a: chunk).plan("Grab it.", SimBody().observe(), [])
    with pytest.raises(IntegrationError):
        parse_plan({"actions": chunk})


def test_groot_stub_without_reasoner():
    with pytest.raises(NotImplementedError):
        GR00TS2Planner().plan("Wave.", SimBody().observe(), [])


def test_joint_chunk_rejected_by_runtime_not_executed():
    body = SimBody()
    ep = Runtime(body, GR00TS2Planner(reasoner=lambda *a: [[0.1] * 16])).run("Wave.")
    assert not ep.success and ep.steps[0].kind == "rejected"
    assert body.world.events == []


def test_fenced_json_parses():
    assert len(parse_plan("```json\n" + json.dumps(PLAN_JSON) + "\n```")) == 5
