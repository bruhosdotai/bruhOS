"""Numeric skill args that only make sense positive must be rejected at the whitelist.

``follow_line(steps=0)`` used to be certified as ``ok / followed 0 steps`` with zero
locomotion, and ``walk_toward(stop_distance_m=0)`` used to be accepted, drive the robot
into the object for the whole primitive budget and end in ``walk timeout``.
"""

import pytest

from bruhos.brain.base import Planner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation, SkillCall
from bruhos.sim.kinematic import SimBody, World
from bruhos.skills.registry import SkillError, catalog, validate


class _FixedPlan(Planner):
    name = "fixed"

    def __init__(self, plan: list[SkillCall]):
        self._plan = plan

    def plan(self, instruction: str, obs: Observation, catalog) -> list[SkillCall]:
        return list(self._plan)


def test_follow_line_rejects_non_positive_steps():
    for steps in (0, -3):
        with pytest.raises(SkillError, match=r"follow_line\.steps: must be >= 1"):
            validate(SkillCall("follow_line", {"steps": steps}))
    assert validate(SkillCall("follow_line", {"steps": 1})).args["steps"] == 1
    assert validate(SkillCall("follow_line", {})).args["steps"] == 20


def test_walk_toward_rejects_non_positive_stop_distance():
    for dist in (0, 0.0, -0.5):
        with pytest.raises(SkillError, match=r"walk_toward\.stop_distance_m: must be > 0"):
            validate(SkillCall("walk_toward", {"target": "red_block", "stop_distance_m": dist}))
    ok = validate(SkillCall("walk_toward", {"target": "red_block", "stop_distance_m": 0.05}))
    assert ok.args["stop_distance_m"] == 0.05
    assert validate(SkillCall("walk_toward", {"target": "red_block"})).args["stop_distance_m"] == 0.1


def test_catalog_advertises_the_bounds():
    spec = {s["skill"]: s for s in catalog()}
    assert spec["follow_line"]["args"]["steps"]["ge"] == 1
    assert spec["walk_toward"]["args"]["stop_distance_m"]["gt"] == 0.0


def test_a_zero_step_line_follow_is_rejected_before_the_body_moves():
    body = SimBody(world=World.tabletop(line_m=2.0))
    ep = Runtime(body, _FixedPlan([SkillCall("follow_line", {"steps": 0})])).run("follow the line")

    assert ep.success is False
    assert ep.end_reason.startswith("plan rejected")
    assert body.n_locomotion == 0
    assert [s.kind for s in ep.steps] == ["rejected"]
