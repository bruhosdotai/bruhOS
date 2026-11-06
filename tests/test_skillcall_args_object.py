"""A present-but-malformed `args` value must be rejected, not silently dropped.

`SkillCall.from_dict` checked `isinstance(args, dict)` only after
`d.get("args") or {}`, so every falsy non-dict (`[]`, `""`, `0`, `False`)
was replaced by `{}` and the skill executed on its defaults, while a truthy
non-dict (`"left"`) was rejected as "args must be an object". Same malformed
class, verdict flipped on truthiness.
"""

import pytest

from bruhos.brain.base import Planner, parse_plan
from bruhos.runtime import Runtime
from bruhos.schemas import SkillCall
from bruhos.sim.kinematic import SimBody


@pytest.mark.parametrize("bad_args", [[], "", 0, False, [1], "left", 1.5])
def test_from_dict_rejects_non_object_args(bad_args):
    with pytest.raises(ValueError, match="args must be an object"):
        SkillCall.from_dict({"skill": "kick", "args": bad_args})


@pytest.mark.parametrize("bad_args", [[], "", 0, False])
def test_parse_plan_rejects_falsy_non_object_args(bad_args):
    with pytest.raises(ValueError, match="args must be an object"):
        parse_plan([{"skill": "kick", "args": bad_args}])


def test_from_dict_keeps_absent_null_and_object_args():
    assert SkillCall.from_dict({"skill": "kick"}).args == {}
    assert SkillCall.from_dict({"skill": "kick", "args": None}).args == {}
    assert SkillCall.from_dict({"skill": "kick", "args": {}}).args == {}
    assert SkillCall.from_dict({"skill": "kick", "args": {"foot": "left"}}).args == {"foot": "left"}


class RawPayloadPlanner(Planner):
    """Mimics the dict-emitting brain adapters: plan() is parse_plan(raw)."""

    name = "raw-payload"

    def __init__(self, raw):
        self._raw = raw

    def plan(self, instruction, obs, catalog):
        return parse_plan(self._raw)


def test_falsy_args_payload_is_a_rejected_plan_not_a_default_kick():
    body = SimBody()
    ep = Runtime(body, RawPayloadPlanner([{"skill": "kick", "args": []}])).run("Kick.")
    assert not ep.success
    assert ep.steps and ep.steps[0].kind == "rejected"
    assert not any(event.startswith("kick") for event in body.world.events)
