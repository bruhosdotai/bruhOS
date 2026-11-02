import pytest

from bruhos.schemas import SkillCall, parse_call
from bruhos.skills import SKILLS, SkillError, catalog_hash, validate
from bruhos.skills.registry import positional_args

V01 = {"look_at", "detect", "walk_toward", "follow_line", "grab", "release", "kick", "wave", "stand_up", "stop"}


def test_whitelist_is_exactly_v01():
    assert set(SKILLS) == V01


def test_unregistered_skill_rejected():
    with pytest.raises(SkillError, match="unregistered"):
        validate(SkillCall("backflip"))


def test_bad_args_rejected():
    with pytest.raises(SkillError):
        validate(SkillCall("kick", {"foot": "middle"}))
    with pytest.raises(SkillError):
        validate(SkillCall("walk_toward", {}))
    with pytest.raises(SkillError):
        validate(SkillCall("wave", {"servo_7": 1500}))


def test_defaults_filled():
    assert validate(SkillCall("kick")).args == {"foot": "right"}
    assert validate(SkillCall("walk_toward", {"target": "box"})).args["stop_distance_m"] == 0.1


def test_shorthand_parse():
    pos = positional_args()
    assert parse_call("walk_toward(red_block)", pos) == SkillCall("walk_toward", {"target": "red_block"})
    assert parse_call("kick(foot=left)", pos) == SkillCall("kick", {"foot": "left"})
    assert parse_call("detect", pos) == SkillCall("detect", {})


def test_shorthand_rejects_extra_positional():
    """A second bare value used to silently overwrite the first: walk_toward(red_block, 0.2)
    parsed to target='0.2', so the robot searched for an object literally named '0.2' and the
    plan failed after a full scan. Extra values must be named explicitly or the plan is rejected.
    """
    pos = positional_args()
    assert parse_call("walk_toward(red_block, stop_distance_m=0.2)", pos) == \
        SkillCall("walk_toward", {"target": "red_block", "stop_distance_m": 0.2})
    with pytest.raises(ValueError, match="one positional"):
        parse_call("walk_toward(red_block, 0.2)", pos)
    with pytest.raises(ValueError, match="one positional"):
        parse_call("follow_line(10, 20)", pos)


def test_shorthand_rejects_duplicate_arg():
    pos = positional_args()
    with pytest.raises(ValueError, match="twice"):
        parse_call("walk_toward(red_block, target=box)", pos)
    with pytest.raises(ValueError, match="twice"):
        parse_call("kick(foot=left, foot=right)", pos)


def test_catalog_hash_stable():
    assert catalog_hash() == catalog_hash()
    assert catalog_hash().startswith("sha256:")
