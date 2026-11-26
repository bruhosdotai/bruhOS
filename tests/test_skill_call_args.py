"""Malformed args must be rejected, never silently replaced by skill defaults."""

import pytest

from bruhos.brain.base import parse_plan
from bruhos.runtime import Runtime
from bruhos.schemas import SkillCall
from bruhos.sim.kinematic import SimBody


@pytest.mark.parametrize("bad", [[], "", 0, False, "left", 42])
def test_non_object_args_are_rejected_whatever_their_truthiness(bad):
    with pytest.raises(ValueError, match="args must be an object"):
        SkillCall.from_dict({"skill": "kick", "args": bad})


@pytest.mark.parametrize("payload", [{"skill": "kick"}, {"skill": "kick", "args": None}])
def test_absent_or_null_args_still_default_to_empty(payload):
    assert SkillCall.from_dict(payload) == SkillCall("kick", {})


@pytest.mark.parametrize("bad", [[], "", 0, False])
def test_plan_with_falsy_non_object_args_is_rejected_before_the_body_runs(bad):
    body = SimBody()
    ep = Runtime(body, _FixedPlan([{"skill": "kick", "args": bad}])).run("kick the ball")
    assert ep.success is False
    assert ep.end_reason.startswith("plan rejected:")
    assert body.world.events == []
    assert [s.kind for s in ep.steps] == ["rejected"]


def test_valid_args_still_execute():
    body = SimBody()
    ep = Runtime(body, _FixedPlan([{"skill": "kick", "args": {"foot": "left"}}])).run("kick")
    assert ep.success is True
    assert "kick:left" in body.world.events


class _FixedPlan:
    name = "fixed"
    version = "fixed-0"

    def __init__(self, calls):
        self.calls = calls

    def plan(self, instruction, obs, catalog):
        return parse_plan(self.calls)
