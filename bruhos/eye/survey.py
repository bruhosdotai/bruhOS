"""Boot look: four corners, then center. No footsteps. No focus hunting."""

from __future__ import annotations

import json
import ssl
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from bruhos.eye.state import DEFAULT_DIR, atomic_write_bytes, atomic_write_json, eye_dir
from bruhos.eye.track import hold_head

SURVEY_POSES = ("L2", "R2", "R0", "L0", "C1")
MEMORY_PATH = Path("/var/lib/bruhos/eye-survey.json")
SETTLE_S = 0.9
HEAD_WAIT_S = 20.0
DARK_CAPTION = "too dark to see"


def clear_head_hold(root: str | Path | None = None) -> None:
    hold_head(0.0, root)


def wait_for_head(url: str, *, timeout_s: float = HEAD_WAIT_S, get: Callable[[str], dict[str, Any]] | None = None) -> bool:
    deadline = time.monotonic() + timeout_s
    fetch = get or _get_json
    while time.monotonic() < deadline:
        try:
            st = fetch(url)
            if st.get("ok"):
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    return False


def load_memory(path: Path = MEMORY_PATH) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def save_memory(survey: dict[str, Any], path: Path = MEMORY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    captions = {
        str(stop.get("pose")): str(stop.get("caption") or "")
        for stop in survey.get("stops") or []
        if stop.get("pose")
    }
    atomic_write_json(path, {
        "updated": time.time(),
        "focus": survey.get("focus") or "",
        "focus_score": survey.get("focus_score"),
        "captions": captions,
        "dark": bool(survey.get("dark")),
    })


def caption_changes(previous: dict[str, Any] | None, survey: dict[str, Any]) -> list[str]:
    if not previous:
        return []
    old = previous.get("captions") if isinstance(previous.get("captions"), dict) else {}
    changed: list[str] = []
    for stop in survey.get("stops") or []:
        pose = str(stop.get("pose") or "")
        if not pose:
            continue
        new = str(stop.get("caption") or "")
        if str(old.get(pose) or "") != new:
            changed.append(pose)
    return changed


def score_focus(images: list[tuple[str, bytes]]) -> dict[str, Any]:
    from bruhos.scan.focus import analyze_images

    if not images:
        return {"score": 0.0, "verdict": "no usable frames (too dark / unreadable)", "usable_frames": 0}
    report = analyze_images(images)
    overall = report.get("overall") or {}
    return {
        "score": float(overall.get("score") or 0.0),
        "verdict": str(overall.get("verdict") or ""),
        "usable_frames": int(overall.get("usable_frames") or 0),
        "excluded_frames": int(overall.get("excluded_frames") or 0),
    }


def run_survey(
    *,
    head_url: str,
    grab: Callable[[], tuple[bytes, float] | None],
    understand: Callable[[bytes], tuple[str, list[str]]],
    root: str | Path | None = None,
    min_mean: float = 40.0,
    settle_s: float = SETTLE_S,
    poses: tuple[str, ...] = SURVEY_POSES,
    memory_path: Path = MEMORY_PATH,
    go_pose: Callable[[str], None] | None = None,
    wait_head: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Look at the corners, then the middle. Returns the survey report."""
    root_path = eye_dir(root)
    survey_dir = root_path / "survey"
    survey_dir.mkdir(parents=True, exist_ok=True)
    hold_head(120.0, root_path)
    print("[eye] SURVEY start", flush=True)

    ready = (wait_head or (lambda: wait_for_head(head_url)))()
    if not ready:
        clear_head_hold(root_path)
        report = {
            "ok": False,
            "error": "head unavailable",
            "poses": list(poses),
            "stops": [],
            "changed": [],
            "dark": False,
            "focus": "",
            "focus_score": None,
        }
        atomic_write_json(root_path / "survey.json", report)
        print("[eye] SURVEY head unavailable — skipping", flush=True)
        return report

    move = go_pose or (lambda pose: _post_pose(head_url, pose))
    previous = load_memory(memory_path)

    # Probe brightness from a settled center-ish grab before deciding on VL.
    first = _settle_grab(grab, settle_s)
    dark = first is None or first[1] < min_mean
    if dark:
        print("[eye] SURVEY dark — poses only, no vision", flush=True)

    stops: list[dict[str, Any]] = []
    images: list[tuple[str, bytes]] = []
    for pose in poses:
        try:
            move(pose)
        except Exception as e:  # noqa: BLE001
            print(f"[eye] SURVEY pose {pose} fail {e}", flush=True)
            stops.append({"pose": pose, "caption": "", "file": "", "mean": 0.0, "error": str(e)[:160]})
            continue
        pair = _settle_grab(grab, settle_s)
        if pair is None:
            stops.append({"pose": pose, "caption": DARK_CAPTION if dark else "", "file": "", "mean": 0.0})
            continue
        jpeg, mean = pair
        path = survey_dir / f"{pose}.jpg"
        atomic_write_bytes(path, jpeg)
        caption = DARK_CAPTION
        objects: list[str] = []
        if not dark and mean >= min_mean:
            try:
                caption, objects = understand(jpeg)
                caption = caption or DARK_CAPTION
            except Exception as e:  # noqa: BLE001
                caption = ""
                print(f"[eye] SURVEY see {pose} fail {e}", flush=True)
                stops.append({
                    "pose": pose,
                    "caption": "",
                    "objects": [],
                    "file": str(path.name),
                    "mean": round(mean, 1),
                    "error": str(e)[:160],
                })
                images.append((pose, jpeg))
                continue
        elif dark:
            caption = DARK_CAPTION
        stops.append({
            "pose": pose,
            "caption": caption,
            "objects": objects,
            "file": str(path.name),
            "mean": round(mean, 1),
        })
        images.append((pose, jpeg))
        print(f"[eye] SURVEY {pose} mean={mean:.0f} {caption!r}", flush=True)

    focus = {"score": 0.0, "verdict": "no usable frames (too dark / unreadable)", "usable_frames": 0}
    if not dark:
        focus = score_focus(images)
        if float(focus.get("score") or 0.0) < 40.0 and focus.get("usable_frames"):
            print(f"[eye] SURVEY focus {focus.get('verdict')}", flush=True)
        elif focus.get("usable_frames"):
            print(f"[eye] SURVEY focus ok score={focus.get('score')}", flush=True)

    focus_text = ""
    if dark:
        focus_text = str(focus.get("verdict") or "no usable frames (too dark / unreadable)")
    elif float(focus.get("score") or 0.0) < 40.0:
        focus_text = str(focus.get("verdict") or "soft — usable up close; ring may need a tweak")
    else:
        focus_text = "focus ok"

    report = {
        "ok": True,
        "poses": list(poses),
        "stops": stops,
        "dark": dark,
        "focus": focus_text,
        "focus_score": focus.get("score"),
        "focus_detail": focus,
        "changed": [],
        "updated": time.time(),
    }
    report["changed"] = caption_changes(previous, report)
    if report["changed"]:
        print(f"[eye] SURVEY changed {','.join(report['changed'])}", flush=True)
    atomic_write_json(root_path / "survey.json", report)
    try:
        save_memory(report, memory_path)
    except OSError as e:
        print(f"[eye] SURVEY memory {e}", flush=True)

    # Live view ends on the center shot.
    center = next((s for s in reversed(stops) if s.get("pose") == poses[-1] and s.get("file")), None)
    if center and center.get("file"):
        jpeg_path = survey_dir / str(center["file"])
        try:
            jpeg = jpeg_path.read_bytes()
        except OSError:
            jpeg = b""
        if jpeg:
            from bruhos.eye.state import write_frame

            write_frame(jpeg, root_path)

    clear_head_hold(root_path)
    print("[eye] SURVEY done", flush=True)
    return report


def _settle_grab(grab: Callable[[], tuple[bytes, float] | None], settle_s: float) -> tuple[bytes, float] | None:
    time.sleep(max(0.0, settle_s))
    # Drop a few frames after the head moves.
    last = None
    for _ in range(4):
        last = grab()
    return last


def _get_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=2.0, context=ctx) as resp:
        return json.loads(resp.read().decode())


def _post_pose(url: str, pose: str) -> None:
    body = json.dumps({"pose": pose, "preview": False, "source": "eye"}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=3.0, context=ctx) as resp:
        resp.read()


def head_url_from_config(cfg: dict[str, Any]) -> str:
    eye = cfg.get("eye") if isinstance(cfg.get("eye"), dict) else {}
    web = cfg.get("web") if isinstance(cfg.get("web"), dict) else {}
    port = int(web.get("port", 8090))
    tls = bool(web.get("tls", False))
    scheme = "https" if tls else "http"
    return str(eye.get("head_url") or f"{scheme}://127.0.0.1:{port}/api/head")
