"""A plan is bounded like every other budget in the runtime.

``max_skill_steps`` bounds one skill, ``max_replans`` bounds replans and ``max_recoveries`` bounds
stand-ups; the plan itself had no bound. A brain -- or a model server answering for it -- could
queue thousands of skill calls and the runtime executed every one of them: on ``main` a 5000-call
plan of ``wave`` ran 5000 action groups and sealed as ``success: true``.
"""

from bruhos.body.base import Body
from bruhos.brain.base import Planner
from bruhos.runtime import Runtime, RuntimeConfig
from bruhos.schemas import Imu, Observation, SkillCall


class CountingBody(Body):
    name = "counting"

    def __init__(self):
        self.acts = 0

    def observe(self):
        return Observation(t=0, frame=0, imu=Imu())

    def act(self, primitive):
        self.acts += 1
        return True

    def halt(self):
        pass


class FloodPlanner(Planner):
    name = "flood"

    def __init__(self, n):
        self.n = n

    def plan(self, instruction, obs, catalog):
        return [SkillCall("wave") for _ in range(self.n)]


def test_a_plan_longer_than_the_budget_never_reaches_the_body():
    body = CountingBody()
    ep = Runtime(body, FloodPlanner(5000)).run("Wave hello.")
    assert body.acts == 0
    assert not ep.success and ep.plan == []
    assert ep.end_reason.startswith("plan rejected:") and "5000" in ep.end_reason


def test_a_plan_at_the_budget_still_runs():
    body = CountingBody()
    ep = Runtime(body, FloodPlanner(24)).run("Wave hello.")
    assert body.acts == 24 and ep.success


def test_the_budget_is_configurable():
    body = CountingBody()
    ep = Runtime(body, FloodPlanner(4), config=RuntimeConfig(max_plan_steps=3)).run("Wave hello.")
    assert body.acts == 0 and not ep.success

