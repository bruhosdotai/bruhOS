"""Validation must not silently rewrite a skill argument into a different value.

Failing-test-first for a reproduced defect: on base, ``validate()`` coerced every
argument with ``a.type(v)`` and accepted whatever came out, so a brain emitting
JSON values of the wrong type had its plan executed against different values
than it sent:

- ``walk_toward(target=123)`` validated as a search for an object literally
  named ``"123"`` (and ``target=True`` as ``"True"``, ``target=["red_block"]``
  as ``"['red_block']"``) — phantom targets that burn a full scan and fail.
- ``follow_line(steps=True)`` validated as ``steps=1`` and ``steps=2.9`` as
  ``steps=2``: a boolean is not a count, and truncating a fractional count
  changes the motion the brain asked for.
- ``walk_toward(stop_distance_m=True)`` validated as ``1.0`` metres.

String coercion that preserves the value (``"20"`` -> ``20``) is documented
behaviour and keeps working; what is rejected is a value whose type makes the
coercion change its meaning.
"""

import pytest

from bruhos.schemas import SkillCall
from bruhos.sim.kinematic import SimBody
from bruhos.skills import SkillContext, SkillError, run_skill, validate


@pytest.mark.parametrize("bad", [123, 1.5, True, ["red_block"], {"name": "red_block"}])
def test_non_string_target_is_rejected(bad):
    with pytest.raises(SkillError):
        validate(SkillCall("walk_toward", {"target": bad}))


@pytest.mark.parametrize("skill,args", [
    ("follow_line", {"steps": True}),
    ("follow_line", {"steps": False}),
    ("walk_toward", {"target": "box", "stop_distance_m": True}),
    ("walk_toward", {"target": "box", "stop_distance_m": False}),
])
def test_boolean_is_not_a_number(skill, args):
    with pytest.raises(SkillError):
        validate(SkillCall(skill, args))


def test_fractional_float_is_not_an_int():
    with pytest.raises(SkillError):
        validate(SkillCall("follow_line", {"steps": 2.9}))


def test_value_preserving_coercions_still_work():
    assert validate(SkillCall("follow_line", {"steps": "20"})).args["steps"] == 20
    assert validate(SkillCall("follow_line", {"steps": 20.0})).args["steps"] == 20
    out = validate(SkillCall("walk_toward", {"target": "box", "stop_distance_m": 1}))
    assert out.args["stop_distance_m"] == 1.0
    assert validate(SkillCall("kick", {"foot": "left"})).args["foot"] == "left"


def test_phantom_target_never_reaches_the_body():
    with pytest.raises(SkillError):
        run_skill(SkillContext(SimBody()), SkillCall("walk_toward", {"target": 123}))
