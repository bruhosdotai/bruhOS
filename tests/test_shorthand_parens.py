"""Shorthand skill calls with unbalanced parentheses must be rejected at parse time.

On pristine code, ``parse_call``'s greedy ``(.*)`` swallows a stray paren into the
argument value, so ``walk_toward(red_block))`` validates cleanly as target
``"red_block)"`` and the robot burns a full scan revolution looking for a target
that can never match, then fails and replans -- instead of rejecting the
malformed call the way keyword garbage already is (``kick(foot=left))`` fails
validation via the choices check).
"""

import pytest

from bruhos.schemas import SkillCall, parse_call
from bruhos.skills.registry import positional_args

POS = positional_args()


@pytest.mark.parametrize("text", [
    "walk_toward(red_block))",
    "walk_toward((red_block)",
    "detect(red_block))",
    "look_at(red_block))",
    "follow_line(20))",
])
def test_unbalanced_parens_rejected(text):
    with pytest.raises(ValueError):
        parse_call(text, POS)


@pytest.mark.parametrize("text, expected", [
    ("walk_toward(red_block)", SkillCall("walk_toward", {"target": "red_block"})),
    ("kick(foot=left)", SkillCall("kick", {"foot": "left"})),
    ("detect", SkillCall("detect", {})),
    # A paren inside quotes is data, not structure, and stays accepted.
    ("walk_toward('a)b')", SkillCall("walk_toward", {"target": "a)b"})),
])
def test_well_formed_calls_still_parse(text, expected):
    assert parse_call(text, POS) == expected
