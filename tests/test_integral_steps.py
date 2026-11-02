import pytest

from bruhos.schemas import SkillCall
from bruhos.skills import SkillError, validate


@pytest.mark.parametrize("steps", [1.9, -0.9, True, False])
def test_integer_steps_do_not_silently_change_the_requested_value(steps):
    with pytest.raises(SkillError):
        validate(SkillCall("follow_line", {"steps": steps}))


@pytest.mark.parametrize("steps", [2, 2.0, "2"])
def test_lossless_integer_coercions_remain_supported(steps):
    assert validate(SkillCall("follow_line", {"steps": steps})).args["steps"] == 2
