"""[brain] settings for the rules brain must not be silently dropped.

On ``main`` every other brain rejects an unknown ``[brain]`` key at startup
(its planner constructor raises ``TypeError``), but ``make_planner("rules")``
discarded the whole config mapping, so a misspelled or leftover setting in
the ``[brain]`` table was ignored and the robot ran the offline rules
planner exactly as if the setting had never been written.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from bruhos.brain import make_planner
from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.cli import _brain

ROOT = Path(__file__).resolve().parents[1]


def test_rules_brain_without_settings_still_builds():
    assert isinstance(make_planner("rules"), RulesPlanner)
    assert isinstance(_brain(None, {"brain": {"name": "rules"}}), RulesPlanner)
    assert isinstance(_brain(None, {}), RulesPlanner)


def test_rules_brain_rejects_settings_it_cannot_use():
    with pytest.raises(ValueError, match=r"rules.*modle"):
        make_planner("rules", modle="gpt-x")


def test_config_for_rules_brain_rejects_leftover_settings():
    cfg = {"brain": {"name": "rules", "base_url": "http://gpu:8000/v1", "model": "m"}}
    with pytest.raises(ValueError, match=r"rules.*base_url.*model"):
        _brain(None, cfg)


def test_shipped_configs_still_build_their_brains():
    for path in sorted((ROOT / "configs").glob("*.toml")):
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
        assert _brain(None, cfg) is not None, path.name
