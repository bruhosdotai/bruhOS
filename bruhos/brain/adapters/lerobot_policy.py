"""LeRobot policy-service adapter — interface in v0.1.

Talks to a policy server over HTTP: ``POST {url}`` with ``{"instruction", "observation"}`` and
expects skill JSON back (``{"plan": [...]}``). Policies that emit continuous actions are for arms
(v0.3, SO-ARM101 / NexArm) and are rejected here by the shared guard.
"""

from __future__ import annotations

import json
from typing import Any

from bruhos.brain.base import Planner, parse_plan
from bruhos.schemas import Observation, SkillCall

from .openai import Transport, http_post


class LeRobotPolicyPlanner(Planner):
    name = "lerobot_policy"

    def __init__(self, url: str = "http://127.0.0.1:8100/plan", timeout: float = 10.0,
                 transport: Transport | None = None):
        self.url = url
        self.timeout = timeout
        self.transport = transport or http_post
        self.version = f"lerobot_policy:{url}"

    def plan(self, instruction: str, obs: Observation, catalog: list[dict[str, Any]]) -> list[SkillCall]:
        body = json.dumps({"instruction": instruction, "observation": obs.to_dict(), "skills": catalog}).encode()
        raw = self.transport(self.url, {"Content-Type": "application/json"}, body, self.timeout)
        return parse_plan(raw)
