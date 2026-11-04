import pytest

from bruhos.brain.base import Planner
from bruhos.runtime import Runtime
from bruhos.schemas import SkillCall
from bruhos.sim.kinematic import SimBody
from bruhos.skills import SkillError, validate


class SingleCallPlanner(Planner):
    name = "single"

    def __init__(self, call):
        self._call = call

    def plan(self, instruction, obs, catalog):
        return [self._call]


def test_validate_non_skillcall_rejected():
    with pytest.raises(SkillError, match="expected SkillCall, got str"):
        validate("detect")
    with pytest.raises(SkillError, match="expected SkillCall, got dict"):
        validate({"skill": "detect"})


@pytest.mark.parametrize("invalid_args,expected_type", [
    (["target"], "list"),
    (None, "NoneType"),
    (42, "int"),
    ("red_block", "str"),
])
def test_validate_non_dict_args_rejected(invalid_args, expected_type):
    with pytest.raises(SkillError, match=f"args must be a dict, got {expected_type}"):
        validate(SkillCall("detect", invalid_args))


def test_validate_heterogeneous_unknown_keys_sorted_without_error():
    with pytest.raises(SkillError, match=r"detect: unknown args"):
        validate(SkillCall("detect", {1: "box", "extra": "foo"}))


def test_runtime_records_rejected_step_for_non_dict_args():
    body = SimBody()
    ep = Runtime(body, SingleCallPlanner(SkillCall("detect", ["target"]))).run("Find.")
    assert not ep.success
    assert len(ep.steps) == 1
    assert ep.steps[0].kind == "rejected"
    assert ep.steps[0].outcome["status"] == "rejected"
    assert ep.end_reason.startswith("plan rejected:")


def test_runtime_records_rejected_step_for_heterogeneous_unknown_keys():
    body = SimBody()
    ep = Runtime(body, SingleCallPlanner(SkillCall("detect", {1: "box", "extra": "foo"}))).run("Find.")
    assert not ep.success
    assert len(ep.steps) == 1
    assert ep.steps[0].kind == "rejected"
    assert ep.steps[0].outcome["status"] == "rejected"
    assert ep.end_reason.startswith("plan rejected:")
