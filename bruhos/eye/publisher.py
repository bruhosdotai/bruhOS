"""Hold the USB camera and publish frame.jpg + state.json."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
from pathlib import Path
from typing import Any, Callable

from bruhos.body.perception import CameraModel
from bruhos.eye.flow import motion_flow
from bruhos.eye.state import (
    DEFAULT_DIR,
    STALE_CAPTION_S,
    EyeState,
    write_frame,
    write_state,
)
from bruhos.eye.understand import understand_frame


def motion_fraction(prev: bytes, curr: bytes, *, jump: int = 24) -> float:
    if not prev or len(prev) != len(curr):
        return 0.0
    changed = sum(1 for a, b in zip(prev, curr) if abs(a - b) > jump)
    return changed / len(prev)

MOTION_FRACTION = 0.04
COOLDOWN_S = 15.0
PUBLISH_HZ = 4.0
JPEG_QUALITY = 90
LIFT_EVERY_S = 60.0


class EyePublisher:
    """Only process that opens /dev/video0 for the always-on view."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str = "ministral-3:3b",
        device: int = 0,
        width: int = 640,
        height: int = 480,
        root: str | Path | None = None,
        min_mean: float = 40.0,
        cooldown_s: float = COOLDOWN_S,
        understand: Callable[[bytes], tuple[str, list[str]]] | None = None,
        open_capture: Callable[[], Any] | None = None,
        follow: Any | None = None,
        survey: bool = True,
        head_url: str | None = None,
        memory_path: Path | None = None,
        detect: Any | None = None,
        camera: CameraModel | None = None,
        jpeg_quality: int = JPEG_QUALITY,
    ):
        self.base_url = base_url
        self.model = model
        self.device = int(device)
        self.width = int(width)
        self.height = int(height)
        self.root = Path(root) if root else DEFAULT_DIR
        self.min_mean = float(min_mean)
        self.cooldown_s = float(cooldown_s)
        self._understand = understand or (
            lambda jpeg: understand_frame(jpeg, base_url=base_url, model=model)
        )
        self._open_capture = open_capture
        self.follow = follow
        self.survey = survey
        self.head_url = head_url
        self.memory_path = memory_path
        self.detect = detect
        self.camera = camera or CameraModel(width=self.width, height=self.height)
        self.jpeg_quality = int(jpeg_quality)
        self.state = EyeState(model=model)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._prev = b""
        self._last_sent = 0.0
        self._saw_lit = False
        self._direction = "still"
        self._survey_done = False
        self._last_lift = 0.0
        self._state_lock = threading.Lock()
        self._vision_thread: threading.Thread | None = None
        self._detect_error = ""

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="bruhos-eye", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self.wait_vision(timeout=1.0)

    def wait_vision(self, timeout: float = 2.0) -> None:
        """Block until an in-flight caption/detect call finishes."""
        thread = self._vision_thread
        if thread is not None:
            thread.join(timeout=timeout)

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _open(self):
        if self._open_capture is not None:
            return self._open_capture()
        import cv2

        cap = cv2.VideoCapture(self.device)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap

    def _encode(self, frame) -> tuple[bytes, bytes, float] | None:
        import cv2

        from bruhos.eye.expose import signal_mean

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (80, 60))
        mean = signal_mean(gray)
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return None
        return small.tobytes(), bytes(buf), mean

    def _publish_frame(self, jpeg: bytes, mean: float, motion: float, flow: dict[str, Any]) -> None:
        write_frame(jpeg, self.root)
        moved: tuple[str, float, float] | None = None
        with self._state_lock:
            self.state.frame_seq += 1
            self.state.mean = mean
            self.state.motion = motion
            self.state.direction = str(flow.get("direction") or "still")
            self.state.flow_x = float(flow.get("flow_x") or 0.0)
            self.state.flow_y = float(flow.get("flow_y") or 0.0)
            self.state.movers = list(flow.get("movers") or [])
            if self.state.direction != self._direction:
                self._direction = self.state.direction
                if self._direction != "still":
                    moved = (self._direction, self.state.flow_x, self.state.flow_y)
            write_state(self.state, self.root)
        if moved is not None:
            print(f"[eye] MOVE {moved[0]} dx={moved[1]} dy={moved[2]}", flush=True)

    def _set_caption(
        self,
        caption: str,
        objects: list[str],
        *,
        error: str = "",
        detections: list[dict[str, Any]] | None = None,
    ) -> None:
        caption = " ".join(caption.split())[:240]
        with self._state_lock:
            self.state.caption = caption
            self.state.objects = list(objects)
            if detections is not None:
                self.state.detections = list(detections)
            elif caption == "too dark to see":
                self.state.detections = []
            self.state.understood_seq = self.state.frame_seq
            self.state.updated = time.time()
            self.state.error = error
            self.state.model = self.model
            write_state(self.state, self.root)
        print(f"[eye] SEE {caption}", flush=True)

    def _caption_stale(self) -> bool:
        text = " ".join(self.state.caption.split())
        if not text or text == "too dark to see":
            return True
        stamp = float(self.state.updated or 0.0)
        if stamp <= 0:
            return True
        return (time.time() - stamp) >= STALE_CAPTION_S

    def _detect_boxes(self, jpeg: bytes, objects: list[str]) -> list[dict[str, Any]] | None:
        """GPU boxes for the names we just saw. None keeps the previous list."""
        if self.detect is None or not objects:
            return []
        from bruhos.detect.types import bboxes_to_detections

        try:
            resp = self.detect.detect_jpeg(
                jpeg, queries=list(objects[:12]), frame_id="eye",
            )
        except (urllib.error.URLError, OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError) as e:
            msg = str(e)[:200]
            self._detect_error = msg
            if "404" in msg:
                self.detect = None
                print("[eye] DETECT off — this server has no /v1/detect", flush=True)
            else:
                print(f"[eye] DETECT fail {msg}", flush=True)
            return None
        if getattr(resp, "wire_only", False):
            print("[eye] DETECT wire_only", flush=True)
            return None
        spatial = bboxes_to_detections(resp.detections, self.camera)
        by_label = {d.label: d for d in spatial}
        out: list[dict[str, Any]] = []
        for box in resp.detections:
            hit = by_label.get(box.label.casefold().strip())
            out.append({
                "label": box.label,
                "score": float(box.confidence),
                "bbox": [box.x1, box.y1, box.x2, box.y2],
                "bearing_deg": float(hit.bearing_deg) if hit is not None else 0.0,
            })
        return out

    def _vision_busy(self) -> bool:
        thread = self._vision_thread
        return thread is not None and thread.is_alive()

    def _understand_job(self, jpeg: bytes, mean: float) -> None:
        """Caption + detect. Runs off the camera thread so frame.jpg keeps moving."""
        try:
            caption, objects = self._understand(jpeg)
            if not caption:
                caption = "too dark to see" if mean < self.min_mean else ""
            if caption:
                self._detect_error = ""
                found = self._detect_boxes(jpeg, objects) if caption != "too dark to see" else []
                self._set_caption(caption, objects, detections=found, error=self._detect_error)
        except (urllib.error.URLError, OSError, RuntimeError, KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            with self._state_lock:
                self.state.error = str(e)[:200]
                write_state(self.state, self.root)
            print(f"[eye] SEE fail {e}", flush=True)

    def _maybe_understand(self, jpeg: bytes, mean: float, motion: float, first_lit: bool) -> None:
        if self._vision_busy():
            return
        now = time.monotonic()
        due = (now - self._last_sent) >= self.cooldown_s
        changed = motion >= MOTION_FRACTION
        stale = mean >= self.min_mean and self._caption_stale()
        if not due or not (first_lit or changed or stale):
            return
        self._last_sent = now
        if mean < self.min_mean:
            self._set_caption("too dark to see", [])
            return
        self._vision_thread = threading.Thread(
            target=self._understand_job,
            args=(bytes(jpeg), float(mean)),
            name="bruhos-eye-vision",
            daemon=True,
        )
        self._vision_thread.start()

    def _lift_if_dark(self, cap) -> None:
        import cv2

        from bruhos.eye.expose import lift_dark
        from bruhos.scan.camera_tune import _read_prop, _set_prop

        def measure() -> float | None:
            pair = self._grab_pair(cap)
            if pair is None:
                return None
            return float(pair[1])

        mean, settings = lift_dark(
            lambda name, value: _set_prop(cap, cv2, name, value),
            measure,
            min_mean=self.min_mean,
            settle_s=0.05,
            read_prop=lambda name: _read_prop(cap, cv2, name),
        )
        self._last_lift = time.monotonic()
        if settings:
            print(f"[eye] exposure {settings} mean={mean:.1f}", flush=True)

    def _grab_pair(self, cap) -> tuple[bytes, float] | None:
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        encoded = self._encode(frame)
        if encoded is None:
            return None
        _gray, jpeg, mean = encoded
        return jpeg, mean

    def _run_boot_survey(self, cap) -> None:
        if not self.survey or not self.head_url or self._survey_done:
            self._survey_done = True
            return
        from bruhos.eye.survey import MEMORY_PATH, run_survey

        def grab() -> tuple[bytes, float] | None:
            pair = self._grab_pair(cap)
            if pair is not None:
                jpeg, mean = pair
                self._publish_frame(
                    jpeg, mean, 0.0,
                    {"direction": "still", "flow_x": 0.0, "flow_y": 0.0, "movers": []},
                )
            return pair

        try:
            report = run_survey(
                head_url=self.head_url,
                grab=grab,
                understand=self._understand,
                root=self.root,
                min_mean=self.min_mean,
                memory_path=self.memory_path or MEMORY_PATH,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[eye] SURVEY fail {e}", flush=True)
            self._survey_done = True
            return
        with self._state_lock:
            self.state.focus = str(report.get("focus") or "")
            self.state.survey_changed = list(report.get("changed") or [])
        center = None
        for stop in report.get("stops") or []:
            if stop.get("pose") == "C1" and stop.get("caption"):
                center = stop
        if center:
            self._set_caption(str(center.get("caption") or ""), list(center.get("objects") or []))
        elif report.get("dark"):
            self._set_caption("too dark to see", [])
        with self._state_lock:
            write_state(self.state, self.root)
        self._survey_done = True

    def _loop(self) -> None:
        period = 1.0 / PUBLISH_HZ
        cap = None
        print(
            f"[eye] watching /dev/video{self.device} model={self.model} dir={self.root}",
            flush=True,
        )
        while not self._stop.is_set():
            t0 = time.monotonic()
            if cap is None:
                try:
                    cap = self._open()
                except Exception as e:  # noqa: BLE001
                    print(f"[eye] camera open {e}", flush=True)
                    cap = None
                if cap is None:
                    print("[eye] camera not open", flush=True)
                    self._stop.wait(2)
                    continue
                if not self._survey_done:
                    self._lift_if_dark(cap)
                    self._run_boot_survey(cap)
            try:
                ok, frame = cap.read()
            except Exception as e:  # noqa: BLE001
                print(f"[eye] camera read {e}", flush=True)
                try:
                    cap.release()
                except Exception:  # noqa: BLE001
                    pass
                cap = None
                self._stop.wait(1)
                continue
            if not ok or frame is None:
                try:
                    cap.release()
                except Exception:  # noqa: BLE001
                    pass
                cap = None
                self._stop.wait(1)
                continue
            encoded = self._encode(frame)
            if encoded is None:
                self._stop.wait(period)
                continue
            gray, jpeg, mean = encoded
            if (
                mean < self.min_mean
                and self._survey_done
                and (time.monotonic() - self._last_lift) >= LIFT_EVERY_S
            ):
                self._lift_if_dark(cap)
                continue
            motion = motion_fraction(self._prev, gray)
            flow = motion_flow(self._prev, gray, width=80, height=60)
            self._prev = gray
            first_lit = (not self._saw_lit) and mean >= self.min_mean
            if mean >= self.min_mean:
                self._saw_lit = True
            self._publish_frame(jpeg, mean, motion, flow)
            if self.follow is not None and self._survey_done:
                self.follow.consider(flow, motion, mean)
            # Dark first frame still gets a local caption once.
            if not self.state.caption and mean < self.min_mean and (time.monotonic() - self._last_sent) >= self.cooldown_s:
                self._last_sent = time.monotonic()
                self._set_caption("too dark to see", [])
            else:
                self._maybe_understand(jpeg, mean, motion, first_lit)
            elapsed = time.monotonic() - t0
            self._stop.wait(max(0.0, period - elapsed))
        if cap is not None:
            try:
                cap.release()
            except Exception:  # noqa: BLE001
                pass


def start_from_config(cfg: dict[str, Any]) -> EyePublisher | None:
    cam = cfg.get("camera")
    if not isinstance(cam, dict) or not cam.get("enabled", True):
        return None
    speech = cfg.get("speech") if isinstance(cfg.get("speech"), dict) else {}
    scan = cfg.get("scan") if isinstance(cfg.get("scan"), dict) else {}
    eye = cfg.get("eye") if isinstance(cfg.get("eye"), dict) else {}
    base = str(eye.get("base_url") or speech.get("base_url") or "").rstrip("/")
    if not base:
        print("[eye] no base_url", flush=True)
        return None
    from bruhos.detect import make_remote_client
    from bruhos.eye.survey import MEMORY_PATH, head_url_from_config
    from bruhos.eye.track import follow_from_config

    detect = make_remote_client(cfg)
    if detect is not None:
        detect.timeout = min(float(detect.timeout), 20.0)
    width = int(cam.get("width", 640))
    height = int(cam.get("height", 480))
    publisher = EyePublisher(
        base_url=base,
        model=str(eye.get("model") or scan.get("model") or "ministral-3:3b"),
        device=int(cam.get("device", 0)),
        width=width,
        height=height,
        root=eye.get("dir") or DEFAULT_DIR,
        min_mean=float(scan.get("vl_min_mean", 40.0)),
        follow=follow_from_config(cfg),
        survey=eye.get("survey", True) is not False and bool(cfg.get("head")),
        head_url=head_url_from_config(cfg) if cfg.get("head") else None,
        memory_path=Path(eye["memory"]) if eye.get("memory") else MEMORY_PATH,
        detect=detect,
        camera=CameraModel(width=width, height=height, hfov_deg=float(cam.get("hfov_deg", 62.0))),
        jpeg_quality=int(eye.get("jpeg_quality", JPEG_QUALITY)),
    )
    publisher.start()
    return publisher
