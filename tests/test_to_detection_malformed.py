"""to_detection must survive malformed bridge telemetry.

Perception entities are streamed in from the eliza bridge — external, off-machine
data. ``observe()`` maps ``to_detection`` over every field of every entity, so a
single malformed numeric field must not raise and take the runtime control loop
down. ``to_detection`` is pure, so this test needs no websockets and runs in CI.
"""

import math

import pytest

from bruhos.body.eliza_bridge import to_detection
from bruhos.schemas import Detection


def test_well_formed_entity_is_unchanged():
    d = to_detection({"label": "red block", "x": 1.0, "y": 0.2, "confidence": 0.9})
    assert d.label == "block"
    assert d.color == "red"
    assert d.confidence == 0.9
    assert d.distance_m == pytest.approx(math.hypot(1.0, 0.2), abs=1e-3)
    assert d.bearing_deg == pytest.approx(math.degrees(math.atan2(0.2, 1.0)), abs=0.1)


def test_missing_numeric_fields_fall_back_to_defaults():
    d = to_detection({"label": "ball"})
    assert d.bearing_deg == 0.0
    assert d.distance_m == 0.0
    assert d.confidence == 1.0


@pytest.mark.parametrize("field", ["x", "y", "confidence"])
def test_garbage_string_field_does_not_raise(field):
    d = to_detection({"label": "red block", field: "garbage"})
    assert isinstance(d, Detection)


def test_none_field_does_not_raise():
    d = to_detection({"label": "red block", "x": None, "y": None, "confidence": None})
    assert d.bearing_deg == 0.0
    assert d.distance_m == 0.0
    assert d.confidence == 1.0


def test_non_finite_field_degrades_to_default():
    # A NaN/inf would otherwise propagate into the bearing/distance and corrupt
    # every plan for the frame; treat it as malformed.
    d = to_detection({"label": "red block", "x": float("nan"), "y": 1.0, "confidence": float("inf")})
    assert d.confidence == 1.0
    assert math.isfinite(d.distance_m)


def test_bool_is_not_a_number():
    # bool is an int subclass; a true flag is not a coordinate.
    d = to_detection({"label": "red block", "x": True, "y": False, "confidence": True})
    assert d.bearing_deg == 0.0
    assert d.distance_m == 0.0
    assert d.confidence == 1.0


def test_explicit_distance_is_used_when_well_formed():
    d = to_detection({"label": "block", "x": 1.0, "y": 1.0, "distance": 1.4})
    assert d.distance_m == 1.4


def test_malformed_distance_falls_back_to_hypot():
    d = to_detection({"label": "block", "x": 3.0, "y": 4.0, "distance": "bad"})
    assert d.distance_m == pytest.approx(5.0, abs=1e-3)


def test_non_dict_entity_is_safe():
    d = to_detection("not-a-dict")
    assert d.label == "object"
