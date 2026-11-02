"""Arg coercion in the skill whitelist must be faithful, not merely non-fatal.

Two defects on main:

1. ``bool`` coercion used ``bool(value)``, so any non-empty string became True --
   ``{"confirm": "false"}`` reached the body as True, and ``{"confirm": "0"}``
   too. A model that emits the string "false" gets the opposite of what it asked
   for, on the whitelist path that is supposed to be the safety boundary.
2. Falsy defaults were dropped: ``Arg(int, default=0)`` and ``Arg(bool,
   default=False)`` never filled the key, because the check was
   ``if a.default is not None``. The normalized call then differed from the spec.
"""

from __future__ import annotations

import pytest

from bruhos.schemas import SkillCall
from bruhos.skills.registry import SKILLS, Arg, SkillError, SkillSpec, validate


@pytest.fixture()
def temp_skill():
    """Register a throwaway spec, then restore the whitelist."""
    name = "_test_coercion"

    def _register(args, positional=None):
        SKILLS[name] = SkillSpec(name=name, doc="test only", args=args, positional=positional)
        return name

    yield _register
    SKILLS.pop(name, None)


def test_bool_arg_reads_strings_as_booleans(temp_skill):
    name = temp_skill({"confirm": Arg(bool)})
    for raw, expected in (("true", True), ("True", True), (1, True),
                          ("false", False), ("False", False), ("0", False), (0, False)):
        assert validate(SkillCall(name, {"confirm": raw})).args["confirm"] is expected, raw


def test_bool_arg_rejects_ambiguous_strings(temp_skill):
    name = temp_skill({"confirm": Arg(bool)})
    for raw in ("yes", "no", "maybe", 2, -1, 0.5, "", "2"):
        with pytest.raises(SkillError):
            validate(SkillCall(name, {"confirm": raw}))


def test_falsy_defaults_are_filled(temp_skill):
    name = temp_skill({"speed": Arg(int, default=0), "dry": Arg(bool, default=False),
                       "label": Arg(str, default="")})
    out = validate(SkillCall(name, {})).args
    assert out == {"speed": 0, "dry": False, "label": ""}


def test_int_arg_rejects_float_with_fraction(temp_skill):
    name = temp_skill({"steps": Arg(int)})
    assert validate(SkillCall(name, {"steps": 3})).args["steps"] == 3
    assert validate(SkillCall(name, {"steps": 3.0})).args["steps"] == 3
    with pytest.raises(SkillError):
        validate(SkillCall(name, {"steps": 2.9}))
