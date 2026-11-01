"""An empty segment in a shorthand call must be a parse error, not an empty value.

Before this fix, ``walk_toward(red_block,)`` parsed to ``target=""``: the empty
segment after the comma was treated as a second positional value and silently
overwrote the real target, so the robot scanned for nothing and failed.
"""

import pytest

from bruhos.brain.base import parse_plan
from bruhos.schemas import SkillCall, parse_call
from bruhos.skills.registry import positional_args

POS = positional_args()


@pytest.mark.parametrize("text", [
    "walk_toward(red_block,)",
    "walk_toward(,)",
    "walk_toward(,red_block)",
    "walk_toward(red_block,,)",
    "detect(,)",
])
def test_empty_segment_is_rejected(text):
    with pytest.raises(ValueError):
        parse_call(text, POS)


def test_empty_segment_is_rejected_through_parse_plan():
    with pytest.raises(ValueError):
        parse_plan(["walk_toward(red_block,)"])


def test_normal_calls_still_parse():
    assert parse_call("walk_toward(red_block)", POS) == SkillCall("walk_toward", {"target": "red_block"})
    assert parse_call("kick(foot=left)", POS) == SkillCall("kick", {"foot": "left"})
    assert parse_call("follow_line(20)", POS) == SkillCall("follow_line", {"steps": 20})
    assert parse_call("wave", POS) == SkillCall("wave", {})
    assert parse_call("wave()", POS) == SkillCall("wave", {})
