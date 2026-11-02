import json
import pytest

from bruhos.brain.adapters.lerobot_policy import LeRobotPolicyPlanner
from bruhos.schemas import Imu, Observation, SkillCall, Status
from bruhos.skills import CATALOG


def test_lerobot_policy_replan_includes_history():
    captured = []

    def fake_transport(url, headers, body, timeout):
        captured.append((url, headers, json.loads(body.decode())))
        return json.dumps({
            "plan": [
                {"skill": "detect", "args": {"target": "red_block"}},
                {"skill": "walk_toward", "args": {"target": "red_block"}},
                {"skill": "grab", "args": {}},
            ]
        }).encode()

    planner = LeRobotPolicyPlanner(url="http://127.0.0.1:8100/plan", transport=fake_transport)
    obs = Observation(t=1.0, frame=10, imu=Imu())
    history = [
        {"call": {"skill": "grab", "args": {}}, "outcome": {"status": "failed", "detail": "nothing in reach"}}
    ]

    plan = planner.replan("Put the red block in the box.", obs, CATALOG, history)

    assert len(captured) == 1
    url, headers, payload = captured[0]
    assert url == "http://127.0.0.1:8100/plan"
    assert headers["Content-Type"] == "application/json"
    assert payload["instruction"] == "Put the red block in the box."
    assert payload["observation"] == obs.to_dict()
    assert payload["skills"] == CATALOG
    assert "history" in payload, "LeRobotPolicyPlanner.replan must forward failure history to the policy server"
    assert payload["history"] == history
    assert len(plan) == 3
    assert plan[0].skill == "detect"


def test_lerobot_policy_plan_does_not_include_history():
    captured = []

    def fake_transport(url, headers, body, timeout):
        captured.append(json.loads(body.decode()))
        return json.dumps({"plan": [{"skill": "wave", "args": {}}]}).encode()

    planner = LeRobotPolicyPlanner(transport=fake_transport)
    obs = Observation(t=0.0, frame=0, imu=Imu())
    planner.plan("Wave.", obs, CATALOG)

    assert len(captured) == 1
    assert "history" not in captured[0]
    assert captured[0]["instruction"] == "Wave."


def test_runtime_invokes_lerobot_policy_replan_with_history():
    from bruhos.body.base import Body
    from bruhos.runtime import Runtime, RuntimeConfig

    class MockBody(Body):
        def __init__(self):
            self._halted = False

        def observe(self):
            return Observation(t=0.0, frame=0, imu=Imu())

        def act(self, primitive: str) -> bool:
            return False

        def halt(self):
            self._halted = True

    captured = []
    responses = [
        {"plan": [{"skill": "grab", "args": {}}]},
        {"plan": []},
    ]

    def fake_transport(url, headers, body, timeout):
        captured.append(json.loads(body.decode()))
        resp = responses.pop(0) if responses else {"plan": []}
        return json.dumps(resp).encode()

    planner = LeRobotPolicyPlanner(transport=fake_transport)
    rt = Runtime(MockBody(), planner, config=RuntimeConfig(max_replans=1))
    ep = rt.run("Wave.")

    assert len(captured) == 2
    # First call: initial plan
    assert "history" not in captured[0]
    # Second call: replan after failure
    assert "history" in captured[1]
    assert len(captured[1]["history"]) == 1
    assert captured[1]["history"][0]["call"]["skill"] == "grab"

