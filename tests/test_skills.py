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


def test_bool_args_rejected_for_string_params():
    # A boolean is not a target name: str(True) == "True" can never match a
    # detection, so the robot would burn a full scan revolution before
    # failing. Numbers are rejected for string params by the same rule;
    # bools are ints in Python, so they must be rejected explicitly here.
    with pytest.raises(SkillError):
        validate(SkillCall("walk_toward", {"target": True}))
    with pytest.raises(SkillError):
        validate(SkillCall("detect", {"target": False}))
    assert validate(SkillCall("walk_toward", {"target": "red_block"})).args["target"] == "red_block"
