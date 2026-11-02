import json

from bruhos.brain.adapters.lfm3_omni import LFM3OmniPlanner
from bruhos.brain.adapters.openai import OpenAICompatPlanner
from bruhos.brain.adapters.qwen_vl import QwenVLPlanner
from bruhos.sim.kinematic import SimBody


def test_openai_replan_messages_no_consecutive_user_roles():
    planner = OpenAICompatPlanner(base_url="https://x/v1", api_key="k")
    history = [{"call": {"skill": "grab", "args": {}}, "outcome": {"status": "failed", "detail": "nothing in reach"}}]
    msgs = planner.messages("Put red in box", SimBody().observe(), [], history)
    roles = [m["role"] for m in msgs]
    for i in range(len(roles) - 1):
        assert (roles[i], roles[i + 1]) != ("user", "user"), f"Consecutive user roles found: {roles}"
    assert roles == ["system", "user"]
    content = json.loads(msgs[-1]["content"])
    assert "history" in content and content["history"] == history


def test_lfm3_omni_replan_messages_strictly_alternates_roles():
    planner = LFM3OmniPlanner()
    history = [{"call": {"skill": "grab", "args": {}}, "outcome": {"status": "failed", "detail": "nothing in reach"}}]
    msgs = planner.messages("Put red in box", SimBody().observe(), [], history)
    roles = [m["role"] for m in msgs]
    for i in range(len(roles) - 1):
        assert roles[i] != roles[i + 1], f"Consecutive same roles found at {i}: {roles[i]} and {roles[i + 1]}"
    assert "Previous attempt failed" in msgs[-1]["content"]


def test_qwen_vl_replan_messages_no_consecutive_user_roles():
    planner = QwenVLPlanner(frame_provider=lambda: b"\xff\xd8jpeg")
    history = [{"call": {"skill": "grab", "args": {}}, "outcome": {"status": "failed", "detail": "nothing in reach"}}]
    msgs = planner.messages("Put red in box", SimBody().observe(), [], history)
    roles = [m["role"] for m in msgs]
    for i in range(len(roles) - 1):
        assert (roles[i], roles[i + 1]) != ("user", "user"), f"Consecutive user roles found: {roles}"
    assert roles == ["system", "user"]
    text_parts = [p["text"] for p in msgs[-1]["content"] if p.get("type") == "text"]
    assert any("history" in p for p in text_parts)
