"""A brain response must not be allowed to run an unbounded number of skills.

``max_skill_steps`` limits primitives inside one skill. Nothing limited how many
skill calls one ``plan`` or ``replan`` could hand the body, so a response of
thousands of whitelisted calls was executed in full and sealed as success.
"""

from bruhos.brain.base import Planner
from bruhos.runtime import Runtime, RuntimeConfig
from bruhos.schemas import SkillCall
from bruhos.sim.kinematic import SimBody, World
from bruhos.trace import verify_episode


class FixedPlanner(Planner):
    name = "fixed"

    def __init__(self, plan, replan=None):
        self._plan = plan
        self._replan = replan

    def plan(self, instruction, obs, catalog):
        # Lists are copied; other iterables are returned as-is so the runtime,
        # not the planner, decides how much of a long response to pull.
        return list(self._plan) if isinstance(self._plan, list) else self._plan

    def replan(self, instruction, obs, catalog, history):
        plan = self._plan if self._replan is None else self._replan
        return list(plan) if isinstance(plan, list) else plan


def _waves(n):
    return [SkillCall("wave") for _ in range(n)]


def test_plan_over_the_default_budget_never_reaches_the_body():
    body = SimBody(World(objects=[]))
    ep = Runtime(body, FixedPlanner(_waves(5000))).run("Wave forever.")
    assert not ep.success
    assert ep.end_reason == "plan rejected: plan exceeds 24 skill calls"
    assert ep.plan == []
    assert [s.kind for s in ep.steps] == ["rejected"]
    assert body.world.events == []
    assert verify_episode(ep)


def test_plan_at_the_budget_still_runs():
    body = SimBody(World(objects=[]))
    ep = Runtime(body, FixedPlanner(_waves(24))).run("Wave twenty-four times.")
    assert ep.success, ep.end_reason
    assert body.world.events.count("wave") == 24
    assert [s.call["skill"] for s in ep.steps] == ["wave"] * 24


def test_plan_budget_is_configurable_and_stops_reading_the_response():
    pulled = {"n": 0}

    def calls():
        for _ in range(5000):
            pulled["n"] += 1
            yield SkillCall("wave")

    body = SimBody(World(objects=[]))
    ep = Runtime(body, FixedPlanner(calls()), config=RuntimeConfig(max_plan_steps=4)).run("Wave.")
    assert not ep.success
    assert ep.end_reason == "plan rejected: plan exceeds 4 skill calls"
    assert pulled["n"] == 5
    assert body.world.events == []


def test_replan_over_the_budget_is_rejected_after_the_failed_skill():
    body = SimBody()
    ep = Runtime(body, FixedPlanner([SkillCall("grab")], replan=_waves(100))).run("Grab it.")
    assert not ep.success
    assert ep.end_reason == "plan rejected: plan exceeds 24 skill calls"
    assert [s.call["skill"] for s in ep.steps] == ["grab", "plan"]
    assert ep.steps[0].outcome["status"] == "failed"
    assert ep.steps[1].kind == "rejected"
    assert "wave" not in body.world.events
    assert verify_episode(ep)
