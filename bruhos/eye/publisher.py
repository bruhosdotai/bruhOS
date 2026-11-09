"""Hold the USB camera and publish frame.jpg + state.json."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
from pathlib import Path
from typing import Any, Callable

from bruhos.eye.flow import motion_flow
from bruhos.eye.state import (
    DEFAULT_DIR,
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
JPEG_QUALITY = 75


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
        self.state = EyeState(model=model)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._prev = b""
        self._last_sent = 0.0
        self._saw_lit = False
        self._busy = False
        self._direction = "still"
        self._survey_done = False

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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (80, 60))
        mean = float(gray.mean())
        out = frame
        h, w = frame.shape[:2]
        if w > 480:
            out = cv2.resize(frame, (480, max(1, int(h * 480 / w))))
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return None
        return small.tobytes(), bytes(buf), mean

    def _publish_frame(self, jpeg: bytes, mean: float, motion: float, flow: dict[str, Any]) -> None:
        write_frame(jpeg, self.root)
        self.state.frame_seq += 1
        self.state.mean = mean
        self.state.motion = motion
        self.state.direction = str(flow.get("direction") or "still")
        self.state.flow_x = float(flow.get("flow_x") or 0.0)
        self.state.flow_y = float(flow.get("flow_y") or 0.0)
        self.state.movers = list(flow.get("movers") or [])
        write_state(self.state, self.root)
        if self.state.direction != self._direction:
            self._direction = self.state.direction
            if self._direction != "still":
                print(
                    f"[eye] MOVE {self._direction} dx={self.state.flow_x} dy={self.state.flow_y}",
                    flush=True,
                )

    def _set_caption(self, caption: str, objects: list[str], *, error: str = "") -> None:
        caption = " ".join(caption.split())[:240]
        self.state.caption = caption
        self.state.objects = list(objects)
        self.state.understood_seq = self.state.frame_seq
        self.state.updated = time.time()
        self.state.error = error
        self.state.model = self.model
        write_state(self.state, self.root)
        print(f"[eye] SEE {caption}", flush=True)

    def _maybe_understand(self, jpeg: bytes, mean: float, motion: float, first_lit: bool) -> None:
        if self._busy:
            return
        now = time.monotonic()
        due = (now - self._last_sent) >= self.cooldown_s
        changed = motion >= MOTION_FRACTION
        if not due or not (first_lit or changed):
            return
        self._last_sent = now
        if mean < self.min_mean:
            self._set_caption("too dark to see", [])
            return
        self._busy = True
        try:
            caption, objects = self._understand(jpeg)
            if not caption:
                caption = "too dark to see" if mean < self.min_mean else ""
            if caption:
                self._set_caption(caption, objects)
        except (urllib.error.URLError, OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
            self.state.error = str(e)[:200]
            write_state(self.state, self.root)
            print(f"[eye] SEE fail {e}", flush=True)
        finally:
            self._busy = False

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
            return self._grab_pair(cap)

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
    from bruhos.eye.survey import MEMORY_PATH, head_url_from_config
    from bruhos.eye.track import follow_from_config

    publisher = EyePublisher(
        base_url=base,
        model=str(eye.get("model") or scan.get("model") or "ministral-3:3b"),
        device=int(cam.get("device", 0)),
        width=int(cam.get("width", 640)),
        height=int(cam.get("height", 480)),
        root=eye.get("dir") or DEFAULT_DIR,
        min_mean=float(scan.get("vl_min_mean", 40.0)),
        follow=follow_from_config(cfg),
        survey=eye.get("survey", True) is not False and bool(cfg.get("head")),
        head_url=head_url_from_config(cfg) if cfg.get("head") else None,
        memory_path=Path(eye["memory"]) if eye.get("memory") else MEMORY_PATH,
    )
    publisher.start()
    return publisher
