import json

import pytest

from bruhos.brain import IntegrationError, make_planner, parse_plan
from bruhos.brain.adapters.groot_s2 import GR00TS2Planner
from bruhos.brain.adapters.lfm3_omni import LFM3OmniPlanner, plan_schema
from bruhos.brain.adapters.lerobot_policy import LeRobotPolicyPlanner
from bruhos.brain.adapters.openai import OpenAICompatPlanner
from bruhos.brain.adapters.qwen_vl import QwenVLPlanner
from bruhos.brain.adapters.rules import RulesPlanner
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
    lambda: OpenAICompatPlanner(base_url="https://x/v1", api_key="k", transport=fake_chat([])),
    lambda: QwenVLPlanner(transport=fake_chat([])),
    lambda: LFM3OmniPlanner(transport=fake_chat([])),
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
    OpenAICompatPlanner(base_url="https://h/v1", model="m", api_key="k", transport=fake_chat(cap)).plan(
        "Wave.", SimBody().observe(), [])
    url, headers, payload = cap[0]
    assert url == "https://h/v1/chat/completions"
    assert headers["Authorization"] == "Bearer k"
    assert payload["model"] == "m" and payload["temperature"] == 0
    assert payload["response_format"] == {"type": "json_object"}


def test_qwen_vl_attaches_frame():
    cap = []
    QwenVLPlanner(transport=fake_chat(cap), frame_provider=lambda: b"\xff\xd8jpeg").plan("Wave.", SimBody().observe(), [])
    content = cap[0][2]["messages"][-1]["content"]
    assert content[0]["type"] == "image_url" and content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_lfm3_omni_request_shape():
    from bruhos.skills import CATALOG

    cap = []
    LFM3OmniPlanner(transport=fake_chat(cap), frame_provider=lambda: b"\xff\xd8jpeg").plan("Wave.", SimBody().observe(), CATALOG)
    url, _, payload = cap[0]
    assert url == "http://127.0.0.1:8080/v1/chat/completions"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["response_format"]["type"] == "json_schema"
    image, text = payload["messages"][-1]["content"]
    assert image["type"] == "image_url" and text["text"].endswith("Instruction: Wave.")
    assert '"name": "red_block"' in text["text"] and "bearing_deg" not in text["text"]
    assert payload["messages"][1]["role"] == "user" and payload["messages"][2]["role"] == "assistant"


def test_lfm3_omni_schema_only_admits_catalog_skills():
    from bruhos.skills import CATALOG

    calls = plan_schema(CATALOG)["properties"]["plan"]["items"]["anyOf"]
    assert sorted(c["properties"]["skill"]["const"] for c in calls) == sorted(s["skill"] for s in CATALOG)
    kick = next(c for c in calls if c["properties"]["skill"]["const"] == "kick")["properties"]["args"]
    assert kick["properties"]["foot"]["enum"] == ["left", "right"] and kick["additionalProperties"] is False


def test_lfm3_omni_unconstrained_falls_back_to_json_object():
    cap = []
    LFM3OmniPlanner(constrain=False, transport=fake_chat(cap)).plan("Wave.", SimBody().observe(), [])
    assert cap[0][2]["response_format"] == {"type": "json_object"}


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


def _rules_run(instruction: str):
    body = SimBody()
    return Runtime(body, RulesPlanner()).run(instruction), body


def _bearing(body: SimBody, name: str) -> float:
    return abs(body.world.relative(body.world.get(name))[1])


def test_rules_landmark_color_does_not_rename_the_ball():
    """'Kick the ball next to the red block' used to walk toward red_ball, which is not in the scene."""
    ep, body = _rules_run("Kick the ball next to the red block.")
    assert ep.success, ep.end_reason
    assert ep.plan[0]["args"]["target"] == "ball"
    assert "kicked:yellow_ball" in body.world.events


def test_rules_landmark_does_not_replace_the_box():
    """'Face the box next to the ball' used to look at the ball and report success."""
    ep, body = _rules_run("Face the box next to the ball.")
    assert ep.success, ep.end_reason
    assert [c["skill"] for c in ep.plan] == ["look_at"]
    assert ep.plan[0]["args"]["target"] == "box"
    assert _bearing(body, "box") <= 8


def test_rules_landmark_color_does_not_rename_a_look_target():
    ep, body = _rules_run("Look at the ball beside the green block.")
    assert ep.success, ep.end_reason
    assert ep.plan[0]["args"]["target"] == "ball"
    assert _bearing(body, "yellow_ball") <= 8


def test_rules_single_color_still_binds_to_its_noun():
    ep, body = _rules_run("Look at the block that is green.")
    assert ep.success, ep.end_reason
    assert ep.plan[0]["args"]["target"] == "green_block"
    assert _bearing(body, "green_block") <= 8


def test_rules_keeps_the_objects_own_color_before_a_landmark():
    ep, body = _rules_run("Kick the yellow ball next to the blue block.")
    assert ep.success, ep.end_reason
    assert ep.plan[0]["args"]["target"] == "yellow_ball"
    assert "kicked:yellow_ball" in body.world.events


def test_rules_landmark_preposition_without_an_object_is_the_destination():
    ep, body = _rules_run("Walk near the box.")
    assert ep.success, ep.end_reason
    assert ep.plan[0]["args"]["target"] == "box"
    assert body.world.relative(body.world.get("box"))[0] <= 0.15
