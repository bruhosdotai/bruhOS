import pytest

from bruhos.schemas import SkillCall, Status, parse_call
from bruhos.sim.kinematic import SimBody
from bruhos.skills import SKILLS, SkillContext, SkillError, catalog_hash, run_skill, validate
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


def test_detect_stops_after_a_rejected_scan_turn():
    class RefusesTurns(SimBody):
        def __init__(self):
            super().__init__()
            self.turn_attempts = 0

        def act(self, primitive):
            if primitive == "turn_left":
                self.turn_attempts += 1
                return False
            return super().act(primitive)

    body = RefusesTurns()
    outcome = run_skill(SkillContext(body), SkillCall("detect", {"target": "green_block"}))

    assert outcome.status is Status.FAILED
    assert outcome.detail == "could not turn while scanning for green_block"
    assert body.turn_attempts == 1
