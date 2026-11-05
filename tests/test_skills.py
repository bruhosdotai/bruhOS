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


def test_catalog_hash_stable():
    assert catalog_hash() == catalog_hash()
    assert catalog_hash().startswith("sha256:")


def test_string_args_validation():
    with pytest.raises(SkillError, match=r"walk_toward\.target: expected str, got None"):
        validate(SkillCall("walk_toward", {"target": None}))
    with pytest.raises(SkillError, match=r"walk_toward\.target: expected str, got bool"):
        validate(SkillCall("walk_toward", {"target": True}))
    with pytest.raises(SkillError, match=r"walk_toward\.target: expected str, got list"):
        validate(SkillCall("walk_toward", {"target": ["red_block"]}))
    with pytest.raises(SkillError, match=r"walk_toward\.target: expected str, got dict"):
        validate(SkillCall("walk_toward", {"target": {"name": "red_block"}}))

    assert validate(SkillCall("detect", {"target": None})).args.get("target") is None
