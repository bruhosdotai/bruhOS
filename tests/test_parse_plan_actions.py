"""``parse_plan`` must accept the ``actions`` container its own guard recognizes.

``guard_no_joint_actions`` reads a plan from ``raw["plan"]`` or ``raw["actions"]``,
but ``parse_plan`` only ever read ``"plan"`` — so a brain that returned the
``actions`` shape passed the safety guard and then died with
``ValueError: plan must be a list, got NoneType``.
"""

import pytest

from bruhos.brain.base import IntegrationError, parse_plan
from bruhos.schemas import SkillCall


def test_actions_container_with_shorthand_strings():
    assert parse_plan({"actions": ["wave", "kick(foot=left)"]}) == [
        SkillCall("wave", {}),
        SkillCall("kick", {"foot": "left"}),
    ]


def test_actions_container_with_dicts():
    assert parse_plan({"actions": [{"skill": "wave", "args": {}}]}) == [SkillCall("wave", {})]


def test_actions_container_still_guards_joint_chunks():
    with pytest.raises(IntegrationError):
        parse_plan({"actions": [[0.1, 0.2, 0.3]]})


def test_plan_container_unchanged():
    assert parse_plan({"plan": ["wave"]}) == [SkillCall("wave", {})]
    assert parse_plan(["wave"]) == [SkillCall("wave", {})]


def test_dict_without_a_container_is_still_rejected():
    with pytest.raises(ValueError):
        parse_plan({"skills": ["wave"]})
