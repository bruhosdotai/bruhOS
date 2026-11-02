"""Explicit JSON null must not be coerced into a phantom value by validation.

Failing-test-first for a reproduced defect: on base, ``validate()`` turned an
explicit ``None`` for a ``str`` arg into the literal string ``"None"`` via
``str(None)``. A brain emitting JSON ``null`` for a target — the standard way
to express "no value" — therefore validated as a search for an object literally
named ``None``. ``walk_toward(target=None)`` burned a full 24-turn scan and
failed with "None not found after full scan" instead of being rejected before
any motion, and ``detect(target=None)`` scanned for ``"None"`` instead of
listing visible objects. ``None`` for a required arg must be rejected, and
``None`` for an optional arg must mean the same as omitting it.
"""

import pytest

from bruhos.brain.base import Planner
from bruhos.runtime import Runtime
from bruhos.schemas import SkillCall
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


def test_validate_rejects_null_required_target():
    with pytest.raises(SkillError):
        validate(SkillCall("walk_toward", {"target": None}))
    with pytest.raises(SkillError):
        validate(SkillCall("look_at", {"target": None}))


def test_validate_null_optional_means_absent():
    # detect.target is optional: null must behave like omitting it.
    assert "target" not in validate(SkillCall("detect", {"target": None})).args
    # Optional args with defaults: null must fall back to the default.
    assert validate(SkillCall("follow_line", {"steps": None})).args["steps"] == 20
    assert validate(SkillCall("walk_toward", {"target": "red_block",
                                              "stop_distance_m": None})).args["stop_distance_m"] == 0.1


def test_run_skill_null_target_rejected_before_motion():
    body = SimBody(World.tabletop())
    with pytest.raises(SkillError):
        run_skill(SkillContext(body), SkillCall("walk_toward", {"target": None}))
    # No wasted scan: the rejection happens before the body ever moves.
    assert body.n_locomotion == 0


def test_runtime_null_target_plan_rejected():
    body = SimBody(World.tabletop())
    brain = _ScriptedBrain([SkillCall("walk_toward", {"target": None})])
    ep = Runtime(body, brain).run("Walk to nothing.")
    assert not ep.success
    assert ep.end_reason.startswith("plan rejected")
    assert body.n_locomotion == 0


def test_detect_null_lists_objects_like_no_target():
    body = SimBody(World.tabletop())
    out = run_skill(SkillContext(body), SkillCall("detect", {"target": None}))
    assert out.status.value == "ok"
    assert "red_block" in out.data["objects"]
