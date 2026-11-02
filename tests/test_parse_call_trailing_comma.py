from bruhos.schemas import SkillCall, parse_call
from bruhos.skills.registry import positional_args


def test_parse_call_with_trailing_comma_positional():
    pos = positional_args()
    parsed = parse_call("walk_toward(red_block, )", pos)
    assert parsed == SkillCall("walk_toward", {"target": "red_block"})


def test_parse_call_with_trailing_comma_keyword():
    pos = positional_args()
    parsed = parse_call("kick(foot=left, )", pos)
    assert parsed == SkillCall("kick", {"foot": "left"})


def test_parse_call_with_multiple_args_and_trailing_comma():
    pos = positional_args()
    parsed = parse_call("walk_toward(red_block, stop_distance_m=0.15, )", pos)
    assert parsed == SkillCall("walk_toward", {"target": "red_block", "stop_distance_m": 0.15})


def test_parse_call_no_positional_with_trailing_comma():
    pos = positional_args()
    parsed = parse_call("wave(param=value, )", pos)
    assert parsed == SkillCall("wave", {"param": "value"})
