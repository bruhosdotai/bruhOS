"""Walk the head pose grid, capture frames, ask VL for a scene inventory."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from bruhos.body.head import Head
from bruhos.scan.camera_tune import burst_best, format_retune_summary, retune_pose, sweeps_from_config
from bruhos.scan.focus import analyze_images, select_for_vl
from bruhos.scan.vision import VisionLister

DEFAULT_SETTLE_S = 1.2
DEFAULT_RETUNE_SETTLE_S = 0.25
WARMUP_FRAMES = 5
DEFAULT_JPEG_QUALITY = 95       # quality-first encode


def _open_camera(cam: dict[str, Any]):
    import cv2

    device = cam.get("device", 0)
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera device {device!r}")
    w, h = int(cam.get("width", 640)), int(cam.get("height", 480))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap


def _grab_jpeg(cap, quality: int = DEFAULT_JPEG_QUALITY) -> bytes:
    import cv2

    for _ in range(WARMUP_FRAMES):
        cap.grab()
    ok, frame = cap.retrieve()
    if not ok or frame is None:
        ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("camera read failed")
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return bytes(buf)


def _wait_still(head: Head, settle_s: float) -> None:
    time.sleep(max(0.0, head.move_ms / 1000.0) + max(0.0, settle_s))


def run_scan(
    cfg: dict[str, Any],
    *,
    head: Head | None = None,
    vision: VisionLister | None = None,
    settle_s: float | None = None,
    save_dir: str | Path | None = None,
    grab: Any = None,
    retune: bool | None = None,
) -> dict[str, Any]:
    """Capture the head grid; quality-gate stills before VL inventory."""
    scan_cfg = cfg.get("scan", {})
    head_cfg = cfg.get("head", {})
    cam = cfg.get("camera", {})
    if settle_s is None:
        if "settle_ms" in scan_cfg:
            settle = float(scan_cfg["settle_ms"]) / 1000.0
        elif "settle_ms" in head_cfg:
            settle = float(head_cfg["settle_ms"]) / 1000.0
        else:
            settle = DEFAULT_SETTLE_S
    else:
        settle = float(settle_s)

    jpeg_quality = int(cam.get("jpeg_quality", scan_cfg.get("jpeg_quality", DEFAULT_JPEG_QUALITY)))
    do_retune = bool(scan_cfg.get("retune", True)) if retune is None else bool(retune)
    retune_settle = float(scan_cfg.get("retune_settle_ms", DEFAULT_RETUNE_SETTLE_S * 1000)) / 1000.0
    retune_mode = str(scan_cfg.get("retune_mode", "guided"))
    retune_good_enough = float(scan_cfg.get("retune_good_enough", 75.0))
    retune_max_trials = int(scan_cfg.get("retune_max_trials", 12))
    burst_n = int(scan_cfg.get("burst_frames", 5))
    vl_min_lap = float(scan_cfg.get("vl_min_lap_var", 25.0))
    vl_min_mean = float(scan_cfg.get("vl_min_mean", 40.0))
    vl_min_score = float(scan_cfg.get("vl_min_score", 30.0))
    vl_max_images = int(scan_cfg.get("vl_max_images", 6))
    order, sweeps = sweeps_from_config(cam)

    own_head = head is None
    if head is None:
        head = Head.from_config(cfg)
    if vision is None:
        vision = VisionLister(
            base_url=scan_cfg.get("base_url") or cfg.get("brain", {}).get("base_url"),
            model=scan_cfg.get("model", "ministral-3:3b"),
            api_key=scan_cfg.get("api_key", cfg.get("brain", {}).get("api_key")),
            api_key_env=scan_cfg.get(
                "api_key_env", cfg.get("brain", {}).get("api_key_env", "OPENAI_API_KEY")
            ),
            timeout=float(scan_cfg.get("timeout", 180.0)),
            json_mode=bool(scan_cfg.get("json_mode", True)),
            merge_model=scan_cfg.get("merge_model", "qwen2.5:7b"),
        )

    poses = head.poses()
    out_dir = Path(save_dir) if save_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    cap = None
    images: list[tuple[str, bytes]] = []
    retune_rows: list[dict[str, Any]] = []
    try:
        head.recenter()
        _wait_still(head, settle)
        if grab is None:
            cap = _open_camera(cam)
            grab_fn = lambda: _grab_jpeg(cap, quality=jpeg_quality)
        else:
            grab_fn = grab

        for pose in poses:
            head.go_pose(pose)
            _wait_still(head, settle)

            if do_retune and cap is not None:
                result = retune_pose(
                    cap,
                    grab_fn,
                    pose=pose.name,
                    settle_s=retune_settle,
                    order=order,
                    sweeps=sweeps,
                    mode=retune_mode,
                    good_enough=retune_good_enough,
                    max_trials=retune_max_trials,
                )
                jpeg = result.jpeg
                score = result.score
                # Burst-pick on the winning settings for a cleaner still.
                if burst_n > 1:
                    jpeg, score, burst_met = burst_best(
                        grab_fn, n=burst_n, pose=pose.name, pause_s=0.08,
                    )
                    result.metrics = burst_met or result.metrics
                retune_rows.append({
                    "pose": pose.name,
                    "score": score,
                    "settings": result.settings,
                    "trials": len(result.trials),
                    "diagnosis": result.diagnosis,
                    "mode": result.mode,
                    "burst": burst_n,
                    "lap_var": result.metrics.get("lap_var"),
                    "mean": result.metrics.get("mean"),
                })
                if out_dir:
                    tune_dir = out_dir / "retune" / pose.name
                    tune_dir.mkdir(parents=True, exist_ok=True)
                    (tune_dir / "best.jpg").write_bytes(jpeg)
                    (tune_dir / "trials.json").write_text(
                        json.dumps(
                            [
                                {
                                    "control": t.control,
                                    "value": t.value,
                                    "score": t.score,
                                    "lap_var": t.lap_var,
                                    "mean": t.mean,
                                    "issue": t.issue,
                                }
                                for t in result.trials
                            ],
                            indent=2,
                        )
                        + "\n"
                    )
            else:
                if burst_n > 1 and cap is not None:
                    jpeg, _, _ = burst_best(grab_fn, n=burst_n, pose=pose.name, pause_s=0.08)
                else:
                    jpeg = grab_fn()

            images.append((pose.name, jpeg))
            if out_dir:
                (out_dir / f"{pose.name}.jpg").write_bytes(jpeg)

        focus = analyze_images(images)
        vl_images, vl_dropped = select_for_vl(
            images,
            min_lap_var=vl_min_lap,
            min_mean=vl_min_mean,
            min_score=vl_min_score,
            max_images=vl_max_images,
        )
        objects = vision.list_objects(vl_images)
        return {
            "poses": [p.name for p in poses],
            "model": vision.model,
            "objects": objects,
            "focus": focus,
            "focus_score": focus["overall"]["score"],
            "retune": retune_rows,
            "vl_frames": [name for name, _ in vl_images],
            "vl_dropped": vl_dropped,
        }
    finally:
        try:
            head.recenter()
            _wait_still(head, min(settle, 0.5))
        except Exception:
            pass
        if cap is not None:
            cap.release()
        if own_head:
            head.close()
