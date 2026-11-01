"""Negative numeric skill args must be rejected at validation, not executed.

Failing-test-first for a reproduced defect: on base, ``follow_line(steps=-3)``
returned ``Status.OK / "followed 0 steps"`` — a skill reporting success without
moving — and ``walk_toward(stop_distance_m=-0.5)`` could never satisfy its
reach test, so it burned the whole step budget and timed out. A distance and a
step count cannot be negative, so the whitelist must reject them before the
body ever sees them.
"""

import pytest

from bruhos.brain.base import Planner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation, SkillCall
from bruhos.sim.kinematic import SimBody, World
from bruhos.skills import SkillContext, SkillError, run_skill, validate


class _ScriptedBrain(Planner):
    """Returns one fixed plan, whatever the instruction says."""

    name = "scripted"
    version = "scripted-0.1"

    def __init__(self, plan):
        self._plan = plan

    def plan(self, instruction, obs, catalog):
        return self._plan


def test_validate_rejects_negative_follow_line_steps():
    with pytest.raises(SkillError):
        validate(SkillCall("follow_line", {"steps": -3}))


def test_validate_rejects_negative_walk_stop_distance():
    with pytest.raises(SkillError):
        validate(SkillCall("walk_toward", {"target": "red_block", "stop_distance_m": -0.5}))


def test_run_skill_rejects_negative_steps_instead_of_false_success():
    body = SimBody(World.tabletop(line_m=5.0))
    with pytest.raises(SkillError):
        run_skill(SkillContext(body), SkillCall("follow_line", {"steps": -3}))


def test_runtime_rejects_negative_steps_plan_instead_of_succeeding():
    body = SimBody(World.tabletop(line_m=5.0))
    brain = _ScriptedBrain([SkillCall("follow_line", {"steps": -3})])
    ep = Runtime(body, brain).run("Follow the line.")
    assert not ep.success
    assert ep.end_reason.startswith("plan rejected")


def test_zero_and_positive_bounds_still_validate():
    # The fix rejects negatives only; it must not change valid plans.
    assert validate(SkillCall("follow_line", {"steps": 0})).args["steps"] == 0
    assert validate(SkillCall("follow_line", {"steps": 20})).args["steps"] == 20
    assert validate(SkillCall("walk_toward", {"target": "red_block",
                                              "stop_distance_m": 0.0})).args["stop_distance_m"] == 0.0
