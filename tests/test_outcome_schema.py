import pytest

from bruhos.schemas import Outcome, Status


def test_outcome_with_status_enum():
    o = Outcome(Status.OK, "success", {"key": "val"})
    assert o.status is Status.OK
    assert o.ok is True
    assert o.to_dict() == {"status": "ok", "detail": "success", "data": {"key": "val"}}


def test_outcome_with_status_string():
    o = Outcome("ok", "success")
    assert o.status is Status.OK
    assert o.ok is True
    assert o.to_dict() == {"status": "ok", "detail": "success", "data": {}}


@pytest.mark.parametrize("status_str,expected_enum", [
    ("ok", Status.OK),
    ("failed", Status.FAILED),
    ("fallen", Status.FALLEN),
    ("stopped", Status.STOPPED),
    ("rejected", Status.REJECTED),
])
def test_outcome_normalizes_all_status_strings(status_str, expected_enum):
    o = Outcome(status_str)
    assert o.status is expected_enum
    assert o.to_dict()["status"] == status_str


@pytest.mark.parametrize("bad_status", ["success", "completed", "error", 123, None, ""])
def test_outcome_invalid_status_raises(bad_status):
    with pytest.raises(ValueError):
        Outcome(bad_status)


def test_outcome_from_dict():
    d = {"status": "failed", "detail": "lost target", "data": {"bearing": 12.5}}
    o = Outcome.from_dict(d)
    assert o.status is Status.FAILED
    assert o.ok is False
    assert o.detail == "lost target"
    assert o.data == {"bearing": 12.5}
    assert o.to_dict() == d


def test_outcome_from_dict_defaults():
    o = Outcome.from_dict({"status": "ok"})
    assert o.status is Status.OK
    assert o.detail == ""
    assert o.data == {}


@pytest.mark.parametrize("bad_dict", ["not_a_dict", 42, None, [], {}, {"detail": "no status"}])
def test_outcome_from_dict_invalid(bad_dict):
    with pytest.raises(ValueError):
        Outcome.from_dict(bad_dict)


def test_outcome_roundtrip():
    original = Outcome(Status.OK, "done", {"items": [1, 2]})
    restored = Outcome.from_dict(original.to_dict())
    assert restored.status == original.status
    assert restored.detail == original.detail
    assert restored.data == original.data
    assert restored.ok is True
