"""Tests for Observation.from_dict serialization and null tolerance.

External JSON telemetry, API payloads, or trace logs often serialize absent
optional fields as explicit nulls (e.g. {"objects": null, "imu": null, "line": null}).
Observation.from_dict must safely degrade to default data structures instead of
raising TypeError.
"""

from bruhos.schemas import Detection, Imu, LineReading, Observation


def test_observation_from_dict_roundtrip():
    obs = Observation(
        t=2.5,
        frame=42,
        objects=[Detection("block", "red", bearing_deg=12.0, distance_m=0.35, confidence=0.95)],
        imu=Imu(roll_deg=1.2, pitch_deg=-0.8, fallen=False),
        line=LineReading(visible=True, offset=0.05),
        holding="red_block",
    )
    d = obs.to_dict()
    restored = Observation.from_dict(d)
    assert restored.t == obs.t
    assert restored.frame == obs.frame
    assert restored.holding == obs.holding
    assert len(restored.objects) == 1
    assert restored.objects[0].label == "block"
    assert restored.objects[0].color == "red"
    assert restored.objects[0].name == "red_block"
    assert restored.imu.roll_deg == obs.imu.roll_deg
    assert restored.line.visible == obs.line.visible


def test_observation_from_dict_tolerates_explicit_nulls():
    payload = {
        "t": 1.0,
        "frame": 5,
        "objects": None,
        "imu": None,
        "line": None,
        "holding": None,
    }
    obs = Observation.from_dict(payload)
    assert obs.t == 1.0
    assert obs.frame == 5
    assert obs.objects == []
    assert isinstance(obs.imu, Imu) and obs.imu.fallen is False
    assert isinstance(obs.line, LineReading) and obs.line.visible is False
    assert obs.holding is None


def test_observation_from_dict_tolerates_partial_nulls():
    payload = {
        "t": 3.0,
        "frame": 10,
        "objects": [{"label": "ball", "color": "yellow", "bearing_deg": 0.0, "distance_m": 1.0, "confidence": 1.0}],
        "imu": None,
        "line": {"visible": True, "offset": -0.1},
        "holding": None,
    }
    obs = Observation.from_dict(payload)
    assert len(obs.objects) == 1
    assert obs.objects[0].name == "yellow_ball"
    assert isinstance(obs.imu, Imu)
    assert obs.line.visible is True and obs.line.offset == -0.1


def test_observation_from_dict_minimal():
    payload = {"t": 0.5, "frame": 1}
    obs = Observation.from_dict(payload)
    assert obs.t == 0.5
    assert obs.frame == 1
    assert obs.objects == []
    assert isinstance(obs.imu, Imu)
    assert isinstance(obs.line, LineReading)
