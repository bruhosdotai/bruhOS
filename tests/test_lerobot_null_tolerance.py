"""Tests ensuring to_lerobot_frames tolerates null and missing fields.

When exporting episode traces to LeRobot dataset frames, external or partial traces
can have explicit nulls (e.g. {"imu": null, "line": null, "objects": null}) or missing
fields in obs_before, call, or outcome. to_lerobot_frames must fall back to safe default
values rather than raising AttributeError, TypeError, or KeyError.
"""

from bruhos.schemas import Episode, Step
from bruhos.trace import LEROBOT_FEATURES, to_lerobot_frames


def test_lerobot_frames_tolerates_explicit_nulls():
    step = Step(
        index=0,
        t=10.0,
        call={"skill": "stop", "args": {}},
        outcome={"status": "ok"},
        obs_before={"imu": None, "line": None, "objects": None, "holding": None},
        kind="skill",
        hash="abc1",
    )
    ep = Episode(
        episode_id="ep-null-fields",
        instruction="Halt.",
        brain="test",
        body="test",
        started_at=10.0,
        steps=[step],
        success=True,
    )
    frames = to_lerobot_frames(ep)
    assert len(frames) == 1
    f = frames[0]
    assert len(f["observation.state"]) == LEROBOT_FEATURES["observation.state"]["shape"][0]
    assert f["observation.state"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert f["action"] == [float(f["action"][0]), 1.0]
    assert f["next.done"] is True
    assert f["next.success"] is True


def test_lerobot_frames_tolerates_nested_null_values():
    step = Step(
        index=0,
        t=5.0,
        call={"skill": "wave", "args": {}},
        outcome={"status": "failed"},
        obs_before={
            "imu": {"roll_deg": None, "pitch_deg": None, "fallen": None},
            "line": {"visible": None, "offset": None},
            "objects": None,
            "holding": None,
        },
        kind="skill",
        hash="abc2",
    )
    ep = Episode(
        episode_id="ep-nested-nulls",
        instruction="Wave.",
        brain="test",
        body="test",
        started_at=5.0,
        steps=[step],
        success=False,
    )
    frames = to_lerobot_frames(ep)
    assert len(frames) == 1
    assert frames[0]["observation.state"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert frames[0]["action"][1] == 0.0  # status not ok -> 0.0


def test_lerobot_frames_tolerates_empty_obs_and_missing_call_outcome_keys():
    step = Step(
        index=0,
        t=1.0,
        call={},
        outcome={},
        obs_before={},
        kind="skill",
        hash="abc3",
    )
    ep = Episode(
        episode_id="ep-empty-dicts",
        instruction="Stop.",
        brain="test",
        body="test",
        started_at=1.0,
        steps=[step],
        success=False,
    )
    frames = to_lerobot_frames(ep)
    assert len(frames) == 1
    assert frames[0]["observation.state"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert frames[0]["action"] == [-1.0, 0.0]
