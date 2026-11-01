"""A hostile bridge peer must not crash the runtime or poison observations with non-finite values."""

import json

from bruhos.body.eliza_bridge import ElizaBridgeBody, to_detection
from bruhos.schemas import Observation


def _zombie_body(telemetry, entities):
    """ElizaBridgeBody without the websocket: sensor state injected directly."""
    body = ElizaBridgeBody.__new__(ElizaBridgeBody)
    body.frame = 0
    body._telemetry = telemetry
    body._entities = entities
    body.fall_deg = 55.0
    body.forward_sign = 1
    body.holding = None
    return body


def test_entity_with_garbage_coordinates_is_dropped():
    assert to_detection({"label": "red block", "x": "boom", "y": None}) is None


def test_entity_non_finite_coordinate_is_dropped():
    assert to_detection({"label": "b", "x": "NaN", "y": 1.0}) is None


def test_entity_that_is_not_a_dict_is_dropped():
    assert to_detection("red block") is None


def test_unusable_distance_stays_unknown_instead_of_zero():
    d = to_detection({"label": "b", "x": 1.0, "y": 0.0, "distance": "boom"})
    assert d is not None and d.distance_m is None


def test_label_only_entity_with_a_distance_keeps_the_distance():
    d = to_detection({"label": "b", "distance": 2.0})
    assert d is not None and d.distance_m == 2.0 and d.bearing_deg == 0.0


def test_observe_survives_hostile_telemetry_and_serializes():
    body = _zombie_body({"imu_roll": "nan", "imu_pitch": "40"},
                        [{"label": "red block", "x": "NaN"}, "junk", {"label": "box"}])
    obs = body.observe()
    assert isinstance(obs, Observation)
    assert json.dumps(obs.to_dict(), allow_nan=False)  # what every trace consumer does
    assert obs.objects == []  # every entity above has no usable position


def test_perception_event_with_non_dict_data_keeps_reader_alive():
    body = _zombie_body({}, [])
    body._handle_event("telemetry.perception", ["not", "a", "dict"])
    assert body._entities == []
    body._handle_event("telemetry.perception", {"entities": None})
    assert body._entities == []
    body._handle_event("telemetry.perception", {"entities": [{"label": "box", "x": 1.0, "y": 0.0}, "junk"]})
    assert body._entities == [{"label": "box", "x": 1.0, "y": 0.0}]
    assert [d.name for d in body.observe().objects] == ["box"]
