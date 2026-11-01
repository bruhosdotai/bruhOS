import pytest

from bruhos.schemas import SkillCall, parse_call
from bruhos.skills import SKILLS, SkillError, catalog_hash, validate
from bruhos.skills.registry import SkillSpec, positional_args, skill, _IMPL

V01 = {"look_at", "detect", "walk_toward", "follow_line", "grab", "release", "kick", "wave", "stand_up", "stop"}


def test_whitelist_is_exactly_v01():
    assert set(SKILLS) == V01


def test_hijacking_a_registered_skill_is_rejected():
    """A late import cannot silently replace a whitelisted skill's spec or implementation."""
    spec_before, impl_before = SKILLS["grab"], _IMPL["grab"]

    def trojan(ctx, args):
        raise AssertionError("hijacked implementation must never run")

    with pytest.raises(SkillError, match="already registered"):
        skill(SkillSpec("grab", "hijacked"))(trojan)
    assert SKILLS["grab"] is spec_before
    assert _IMPL["grab"] is impl_before


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
