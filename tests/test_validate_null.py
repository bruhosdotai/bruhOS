"""A JSON null arg must not become the literal string 'None'."""
from bruhos.schemas import SkillCall
from bruhos.skills.registry import SkillError, validate
import pytest

def test_null_string_arg_rejected():
    with pytest.raises(SkillError):
        validate(SkillCall("detect", {"target": None}))

def test_null_int_arg_rejected():
    with pytest.raises(SkillError):
        validate(SkillCall("follow_line", {"steps": None}))
