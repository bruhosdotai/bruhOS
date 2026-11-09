import json
from pathlib import Path

from bruhos.eye.survey import (
    SURVEY_POSES,
    caption_changes,
    run_survey,
    save_memory,
    score_focus,
)


def _jpeg(mean: int = 80) -> bytes:
    import cv2
    import numpy as np

    frame = np.full((48, 64, 3), int(mean), dtype=np.uint8)
    # Soft texture so focus can score something.
    frame[10:20, 10:20] = min(255, mean + 40)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return bytes(buf)


def test_survey_pose_order_and_dark_skips_vision(tmp_path):
    poses = []
    vision_calls = []

    def grab():
        return _jpeg(mean=5), 5.0

    def understand(jpeg):
        vision_calls.append(1)
        return "should not run", []

    def go(pose):
        poses.append(pose)

    report = run_survey(
        head_url="http://unused/api/head",
        grab=grab,
        understand=understand,
        root=tmp_path,
        min_mean=40.0,
        settle_s=0.0,
        memory_path=tmp_path / "memory.json",
        go_pose=go,
        wait_head=lambda: True,
    )
    assert poses == list(SURVEY_POSES)
    assert vision_calls == []
    assert report["dark"] is True
    assert report["ok"] is True
    assert all(s.get("caption") == "too dark to see" for s in report["stops"] if s.get("file"))
    assert (tmp_path / "survey.json").is_file()
    assert (tmp_path / "survey" / "C1.jpg").is_file()


def test_survey_lit_records_soft_focus_without_blocking(tmp_path, monkeypatch):
    def grab():
        return _jpeg(mean=90), 90.0

    def understand(jpeg):
        return "A desk.", []

    monkeypatch.setattr(
        "bruhos.eye.survey.score_focus",
        lambda images: {
            "score": 25.0,
            "verdict": "soft — usable up close; ring may need a tweak",
            "usable_frames": 5,
            "excluded_frames": 0,
        },
    )
    report = run_survey(
        head_url="http://unused/api/head",
        grab=grab,
        understand=understand,
        root=tmp_path,
        min_mean=40.0,
        settle_s=0.0,
        memory_path=tmp_path / "memory.json",
        go_pose=lambda pose: None,
        wait_head=lambda: True,
    )
    assert report["dark"] is False
    assert "soft" in report["focus"]
    assert report["focus_score"] == 25.0


def test_survey_changed_captions_only(tmp_path):
    first = {
        "stops": [
            {"pose": "L2", "caption": "a wall"},
            {"pose": "C1", "caption": "a desk"},
        ],
        "focus": "focus ok",
        "dark": False,
    }
    save_memory(first, tmp_path / "memory.json")

    def grab():
        return _jpeg(mean=90), 90.0

    captions = {"L2": "a wall", "R2": "a lamp", "R0": "floor", "L0": "chair", "C1": "a person"}

    def understand(jpeg):
        # run_survey calls understand per pose; use call order via stops
        return "unused", []

    posed = []

    def go(pose):
        posed.append(pose)

    # Inject captions by wrapping understand with pose list
    idx = {"n": 0}

    def understand_pose(jpeg):
        pose = SURVEY_POSES[idx["n"]]
        idx["n"] += 1
        return captions[pose], []

    report = run_survey(
        head_url="http://unused/api/head",
        grab=grab,
        understand=understand_pose,
        root=tmp_path,
        min_mean=40.0,
        settle_s=0.0,
        memory_path=tmp_path / "memory.json",
        go_pose=go,
        wait_head=lambda: True,
    )
    # Previous only had L2 and C1; L2 same, C1 changed, others new.
    assert "C1" in report["changed"]
    assert "L2" not in report["changed"]


def test_caption_changes_helper():
    prev = {"captions": {"L2": "a", "C1": "b"}}
    survey = {"stops": [{"pose": "L2", "caption": "a"}, {"pose": "C1", "caption": "c"}]}
    assert caption_changes(prev, survey) == ["C1"]
    assert caption_changes(None, survey) == []


def test_score_focus_empty():
    got = score_focus([])
    assert got["usable_frames"] == 0
