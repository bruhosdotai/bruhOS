"""Head pose grid, VL listing, and scan orchestration (no hardware)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bruhos.body.head import Head, clamp_pulse, load_servo_centers, scan_poses
from bruhos.scan.camera_tune import diagnose, retune_pose, score_frame
from bruhos.scan.focus import analyze_images, format_focus_table, select_for_vl
from bruhos.scan.scan import run_scan
from bruhos.body.perception import CameraModel
from bruhos.scan.ground import enrich, parse_grounded_payload, run_vl_boxes, vfov_deg
from bruhos.scan.vision import VisionLister, normalize_objects, parse_objects_payload


def _tiny_jpeg(seed: int = 0) -> bytes:
    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)
    # Mix flat + edges so Laplacian varies with seed
    img = np.full((48, 64, 3), 40 + (seed % 50), dtype=np.uint8)
    if seed % 3 == 0:
        img[:, ::4] = 200
        img[::4, :] = 200
    else:
        img = np.clip(img.astype(np.int16) + rng.integers(-5, 6, img.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    assert ok
    return bytes(buf)


class FakeCtl:
    def __init__(self):
        self.calls: list[tuple[int, int, int]] = []

    def set_pwm_servo_pulse(self, servo_id, pulse, use_time):
        self.calls.append((servo_id, pulse, use_time))


def test_load_servo_centers(tmp_path):
    p = tmp_path / "servo_config.yaml"
    p.write_text("# centers\nservo1: 914\nservo2: 1477\n")
    assert load_servo_centers(p) == (914, 1477)


def test_scan_poses_3x3_snake_looks_up():
    # servo1=914 table-down → 1800 look-up (not KickBall's 1200 floor search)
    poses = scan_poses(914, 1477, yaw_offset_us=400, pitch_max_us=1800)
    assert [p.name for p in poses] == [
        "L0", "C0", "R0", "R1", "C1", "L1", "L2", "C2", "R2",
    ]
    assert poses[0].pitch_us == 914
    assert poses[3].pitch_us == (914 + 1800) // 2
    assert poses[6].pitch_us == 1800 and poses[8].pitch_us == 1800
    assert poses[0].yaw_us == 1477 - 400
    assert poses[2].yaw_us == 1477 + 400


def test_scan_poses_2_rows_when_span_medium():
    poses = scan_poses(914, 1477, yaw_offset_us=400, pitch_min_us=914, pitch_max_us=1200)
    assert [p.name for p in poses] == ["L0", "C0", "R0", "R1", "C1", "L1"]
    assert poses[-1].pitch_us == 1200


def test_scan_poses_collapse_when_pitch_range_tiny():
    poses = scan_poses(1000, 1500, yaw_offset_us=400, pitch_min_us=1000, pitch_max_us=1020)
    assert [p.name for p in poses] == ["L0", "C0", "R0"]
    assert all(p.pitch_us == 1000 for p in poses)


def test_head_go_pulse_clamps(tmp_path):
    ctl = FakeCtl()
    head = Head(ctl, pitch_center=914, yaw_center=1477, yaw_offset_us=400, pitch_max_us=1800)
    head.go_pulse(yaw_us=9999, pitch_us=0)
    pitch_call = next(c for c in ctl.calls if c[0] == 1)
    yaw_call = next(c for c in ctl.calls if c[0] == 2)
    assert pitch_call[1] == 914
    assert yaw_call[1] == 1477 + 400
    ctl.calls.clear()
    head.go_pulse(yaw_us=1477, pitch_us=9999)
    assert next(c for c in ctl.calls if c[0] == 1)[1] == 1800


def test_clamp_pulse():
    assert clamp_pulse(500, 914, 1800) == 914
    assert clamp_pulse(2000, 914, 1800) == 1800
    assert clamp_pulse(1000, 914, 1800) == 1000


def test_normalize_and_parse_objects():
    assert normalize_objects(["Cup", "cup", "Wall", "red mug", ""]) == ["cup", "red mug"]
    assert parse_objects_payload('{"objects": ["box", "sponge"]}') == ["box", "sponge"]
    assert parse_objects_payload('```json\n{"objects": ["Ball"]}\n```') == ["ball"]


def test_vision_lister_multi_image_payload():
    cap = []

    def transport(url, headers, body, timeout):
        cap.append(json.loads(body))
        return json.dumps({
            "choices": [{"message": {"content": json.dumps({"objects": ["mug", "Floor", "box"]})}}]
        }).encode()

    vl = VisionLister(base_url="http://h/v1", model="ministral-3:3b", transport=transport)
    objs = vl.list_objects([("L0", b"\xff\xd8a"), ("C0", b"\xff\xd8b")])
    assert objs == ["box", "mug"]
    content = cap[0]["messages"][-1]["content"]
    imgs = [p for p in content if p.get("type") == "image_url"]
    assert len(imgs) == 2
    assert imgs[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert cap[0]["model"] == "ministral-3:3b"


def test_vision_lister_falls_back_on_400():
    import urllib.error

    calls = []

    def transport(url, headers, body, timeout):
        payload = json.loads(body)
        calls.append(payload)
        content = payload["messages"][-1]["content"]
        # Reject multi-image (list content with >1 image)
        if isinstance(content, list) and sum(1 for p in content if p.get("type") == "image_url") > 1:
            raise urllib.error.HTTPError(url, 400, "bad", hdrs=None, fp=None)
        # Single-image or text merge
        if isinstance(content, list):
            return json.dumps({
                "choices": [{"message": {"content": json.dumps({"objects": ["cup"]})}}]
            }).encode()
        return json.dumps({
            "choices": [{"message": {"content": json.dumps({"objects": ["cup", "mug"]})}}]
        }).encode()

    vl = VisionLister(base_url="http://h/v1", model="m", transport=transport, merge_model="qwen2.5:7b")
    objs = vl.list_objects([("L0", b"a"), ("C0", b"b")])
    assert "cup" in objs
    assert any(
        isinstance(c["messages"][-1]["content"], list)
        and sum(1 for p in c["messages"][-1]["content"] if p.get("type") == "image_url") == 1
        for c in calls
    )


def test_focus_table_ranks_sharp_over_soft():
    soft = _tiny_jpeg(1)          # noisy flat
    sharp = _tiny_jpeg(0)         # edged
    report = analyze_images([("soft", soft), ("sharp", sharp)])
    by = {r["pose"]: r for r in report["frames"]}
    assert by["sharp"]["lap_var"] > by["soft"]["lap_var"]
    assert "score" in report["overall"]
    table = format_focus_table(report)
    assert "OVERALL FOCUS" in table and "sharp" in table and "soft" in table


def test_overall_focus_ignores_dark_frames():
    from bruhos.scan.focus import overall_focus

    frames = [
        {"pose": "A", "lap_var": 80.0, "mean": 90.0, "verdict": "ok"},
        {"pose": "B", "lap_var": 90.0, "mean": 85.0, "verdict": "ok"},
        {"pose": "D", "lap_var": 0.4, "mean": 1.0, "verdict": "DARK+SOFT"},
    ]
    o = overall_focus(frames)
    assert o["usable_frames"] == 2 and o["excluded_frames"] == 1
    assert o["score"] > 0


def test_score_frame_prefers_sharp_lit():
    soft_dark = _tiny_jpeg(2)
    # force a near-black jpeg
    import cv2
    import numpy as np
    black = np.zeros((48, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", black)
    assert ok
    s_black, _ = score_frame(bytes(buf), "b")
    s_soft, _ = score_frame(soft_dark, "s")
    s_sharp, _ = score_frame(_tiny_jpeg(0), "k")
    assert s_black < s_soft <= s_sharp or s_black < s_sharp


def test_diagnose_dark_suggests_brightness():
    issues, actions = diagnose(
        {"mean": 20.0, "p99": 40.0, "lap_var": 5.0},
        {"brightness": 64.0, "gamma": 100.0},
    )
    assert "dark" in issues
    assert any(c == "brightness" and v > 64 for c, v in actions)


def test_retune_guided_stops_when_excellent_and_clean():
    import cv2
    import numpy as np

    class FakeCap:
        def __init__(self):
            self.props = {
                cv2.CAP_PROP_BRIGHTNESS: 128.0,
                cv2.CAP_PROP_SHARPNESS: 200.0,
                cv2.CAP_PROP_CONTRAST: 128.0,
                cv2.CAP_PROP_GAMMA: 100.0,
            }

        def get(self, pid):
            return float(self.props.get(pid, 0.0))

        def set(self, pid, val):
            self.props[pid] = float(val)
            return True

    grabs = {"n": 0}

    def grab():
        grabs["n"] += 1
        # Mid-bright edged frame → diagnose "ok" and high score (not clipped)
        img = np.full((120, 160, 3), 100, dtype=np.uint8)
        img[:, ::4] = 160
        img[::4, :] = 140
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        return bytes(buf)

    result = retune_pose(
        FakeCap(), grab, pose="C0", settle_s=0.0, mode="guided",
        good_enough=50.0, max_trials=6, cv2_mod=cv2,
    )
    assert result.mode == "guided"
    assert grabs["n"] == 1  # baseline only — excellent + clean
    assert len(result.trials) == 1


def test_select_for_vl_drops_dark():
    import cv2
    import numpy as np

    def jpg(val, edged=False):
        img = np.full((48, 64, 3), val, dtype=np.uint8)
        if edged:
            img[:, ::3] = 220
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        return bytes(buf)

    images = [("dark", jpg(5)), ("good", jpg(110, edged=True)), ("ok2", jpg(100, edged=True))]
    kept, dropped = select_for_vl(images, min_lap_var=10, min_mean=40, min_score=10, max_images=6)
    names = " ".join(k for k, _ in kept)
    assert "good" in names and "dark" in dropped


def test_retune_guided_tries_brightness_when_dark():
    import cv2
    import numpy as np

    class FakeCap:
        def __init__(self):
            self.props = {
                cv2.CAP_PROP_BRIGHTNESS: 40.0,
                cv2.CAP_PROP_SHARPNESS: 64.0,
                cv2.CAP_PROP_CONTRAST: 128.0,
                cv2.CAP_PROP_GAMMA: 100.0,
                cv2.CAP_PROP_AUTO_EXPOSURE: 0.25,
                cv2.CAP_PROP_BACKLIGHT: 0.0,
            }

        def get(self, pid):
            return float(self.props.get(pid, 0.0))

        def set(self, pid, val):
            self.props[pid] = float(val)
            return True

    def grab():
        b = int(cap.props[cv2.CAP_PROP_BRIGHTNESS])
        if b < 100:
            img = np.full((48, 64, 3), 8, dtype=np.uint8)
        else:
            img = np.full((48, 64, 3), 110, dtype=np.uint8)
            img[:, ::3] = 200
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        return bytes(buf)

    cap = FakeCap()
    result = retune_pose(
        cap, grab, pose="L0", settle_s=0.0, mode="guided",
        good_enough=55.0, max_trials=6, cv2_mod=cv2,
    )
    assert result.mode == "guided"
    assert "dark" in result.diagnosis or result.settings.get("brightness", 0) >= 100
    assert len(result.trials) <= 7  # baseline + max_trials
    assert any(t.control == "brightness" for t in result.trials)


def test_run_scan_orchestration(tmp_path):
    ctl = FakeCtl()
    head = Head(ctl, pitch_center=914, yaw_center=1477, yaw_offset_us=400,
                pitch_max_us=1800, move_ms=0)
    frames = {"n": 0}

    def grab():
        frames["n"] += 1
        return _tiny_jpeg(frames["n"])

    cap = []

    def transport(url, headers, body, timeout):
        cap.append(json.loads(body))
        return json.dumps({
            "choices": [{"message": {"content": json.dumps({"objects": ["sponge", "box"]})}}]
        }).encode()

    vl = VisionLister(base_url="http://h/v1", model="ministral-3:3b", transport=transport)
    save = tmp_path / "shots"
    result = run_scan(
        {"scan": {
            "retune": False,
            "burst_frames": 1,
            "vl_min_lap_var": 0,
            "vl_min_mean": 0,
            "vl_min_score": 0,
            "vl_max_images": 9,
        }, "camera": {}},
        head=head,
        vision=vl,
        settle_s=0.0,
        save_dir=save,
        grab=grab,
        retune=False,
    )
    assert result["poses"] == ["L0", "C0", "R0", "R1", "C1", "L1", "L2", "C2", "R2"]
    assert result["objects"] == ["box", "sponge"]
    assert result["model"] == "ministral-3:3b"
    assert frames["n"] == 9
    assert len(result["focus"]["frames"]) == 9
    assert "score" in result["focus"]["overall"]
    assert result["focus_score"] == result["focus"]["overall"]["score"]
    assert all("lap_var" in r and "verdict" in r for r in result["focus"]["frames"])
    assert (save / "C0.jpg").read_bytes()[:2] == b"\xff\xd8"
    content = cap[0]["messages"][-1]["content"]
    n_img = sum(1 for p in content if p.get("type") == "image_url")
    assert n_img == len(result["vl_frames"]) == 9
    assert any(c[0] == 2 and c[1] == 1477 for c in ctl.calls)
    assert any(c[0] == 1 and c[1] == 1800 for c in ctl.calls)


def test_head_from_config_with_inline_centers():
    ctl = FakeCtl()
    head = Head.from_config(
        {"head": {"servo1": 900, "servo2": 1500, "yaw_offset_us": 400, "pitch_max_us": 1800}},
        ctl=ctl,
    )
    assert [p.name for p in head.poses()] == [
        "L0", "C0", "R0", "R1", "C1", "L1", "L2", "C2", "R2",
    ]
    assert head.pitch_center == 900


def test_parse_grounded_and_enrich_bearing():
    raw = parse_grounded_payload(
        '{"objects":[{"name":"Shelf","bbox":[0.0,0.25,0.5,0.75],"conf":0.9}]}'
    )
    assert len(raw) == 1 and raw[0].label == "Shelf"
    cam = CameraModel(width=640, height=480, hfov_deg=62.0)
    assert vfov_deg(cam) > 40
    dets = enrich(raw, cam)
    assert dets[0].label == "shelf"
    # box center at x=0.25*640=160 → left of center → positive bearing
    assert dets[0].bearing_deg > 0
    assert dets[0].x1 == 0.0 and dets[0].x2 == 320.0


def test_vl_boxes_backend_mocked():
    jpeg = _tiny_jpeg(1)

    def transport(url, headers, body, timeout):
        return json.dumps({
            "choices": [{"message": {"content": json.dumps({
                "objects": [{"name": "box", "bbox": [0.2, 0.2, 0.8, 0.8], "conf": 0.7}],
            })}}]
        }).encode()

    fr = run_vl_boxes(
        label="C0", jpeg=jpeg, base_url="http://h/v1", model="ministral-3:3b",
        transport=transport,
    )
    assert fr.ok and fr.detections[0].label == "box"
    assert fr.detections[0].x2 > fr.detections[0].x1


def test_yaw_sign_flips_left_right_pulses():
    poses = scan_poses(914, 1477, yaw_offset_us=400, pitch_max_us=1800, yaw_sign=-1)
    by = {p.name: p for p in poses}
    # With mirrored sign, L is +offset and R is -offset
    assert by["L0"].yaw_us == 1477 + 400
    assert by["R0"].yaw_us == 1477 - 400
