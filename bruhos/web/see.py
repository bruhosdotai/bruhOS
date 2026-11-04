"""Async single-shot camera see pipeline for the web UI.

Flow (quality over speed): capture → score → retune → burst → VL inventory → grounded boxes.
Each stage updates progress so the UI can show bars while longer steps run.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from bruhos.body.perception import CameraModel
from bruhos.scan.bench import curl_transport
from bruhos.scan.camera_tune import burst_best, retune_pose, score_frame, sweeps_from_config
from bruhos.scan.ground import draw_overlay, ground_image
from bruhos.scan.scan import DEFAULT_JPEG_QUALITY, DEFAULT_RETUNE_SETTLE_S, _grab_jpeg, _open_camera
from bruhos.scan.vision import VisionLister

GrabFn = Callable[[], bytes]
OpenCamFn = Callable[[dict[str, Any]], Any]
GroundFn = Callable[..., Any]

STAGE_SPECS: list[tuple[str, str]] = [
    ("capture", "Take picture"),
    ("score", "Measure focus / exposure"),
    ("retune", "Clean up camera settings"),
    ("burst", "Pick sharpest still"),
    ("inventory", "Name what's in view"),
    ("ground", "Locate objects (boxes + bearing)"),
]


@dataclass
class Stage:
    id: str
    label: str
    status: str = "pending"  # pending | running | done | error | skipped
    progress: float = 0.0
    detail: str = ""
    seconds: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "status": self.status,
            "progress": round(self.progress, 1),
            "detail": self.detail,
            "seconds": round(self.seconds, 2),
            "error": self.error,
        }


@dataclass
class SeeJob:
    id: str
    root: Path
    cfg: dict[str, Any]
    stages: list[Stage] = field(default_factory=list)
    state: str = "queued"  # queued | running | done | error
    error: str | None = None
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    camera: dict[str, Any] = field(default_factory=dict)
    inventory: list[str] = field(default_factory=list)
    detections: list[dict[str, Any]] = field(default_factory=list)
    images: dict[str, str] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if not self.stages:
            self.stages = [Stage(i, lab) for i, lab in STAGE_SPECS]

    def _touch(self) -> None:
        self.updated = time.time()

    def stage(self, sid: str) -> Stage:
        for s in self.stages:
            if s.id == sid:
                return s
        raise KeyError(sid)

    def set_stage(
        self,
        sid: str,
        *,
        status: str | None = None,
        progress: float | None = None,
        detail: str | None = None,
        error: str | None = None,
        seconds: float | None = None,
    ) -> None:
        with self._lock:
            s = self.stage(sid)
            if status is not None:
                s.status = status
            if progress is not None:
                s.progress = max(0.0, min(100.0, float(progress)))
            if detail is not None:
                s.detail = detail
            if error is not None:
                s.error = error
            if seconds is not None:
                s.seconds = seconds
            self._touch()

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "state": self.state,
                "error": self.error,
                "created": self.created,
                "updated": self.updated,
                "stages": [s.to_dict() for s in self.stages],
                "images": dict(self.images),
                "camera": dict(self.camera),
                "inventory": list(self.inventory),
                "detections": list(self.detections),
            }


class SeeHub:
    """One active camera job at a time; history kept on disk under http_dir/see/."""

    def __init__(
        self,
        http_dir: Path,
        cfg: dict[str, Any],
        *,
        grab: GrabFn | None = None,
        open_camera: OpenCamFn | None = None,
        vision: VisionLister | None = None,
        ground_fn: GroundFn | None = None,
    ):
        self.http_dir = Path(http_dir)
        self.cfg = cfg
        self._grab = grab
        self._open_camera = open_camera
        self._vision = vision
        # None → ground_image (GPU /v1/detect then VL). Injected for tests.
        self._ground_fn = ground_fn
        self._lock = threading.Lock()
        # Held only while the UVC device is open (not during VL). Head previews
        # can run during inventory/ground once the camera is released.
        self._cam_lock = threading.Lock()
        self._jobs: dict[str, SeeJob] = {}
        self._current: str | None = None
        self.see_root = self.http_dir / "see"
        self.see_root.mkdir(parents=True, exist_ok=True)

    def get(self, job_id: str) -> SeeJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def latest(self) -> SeeJob | None:
        with self._lock:
            if not self._jobs:
                return None
            return max(self._jobs.values(), key=lambda j: j.created)

    def start(self) -> SeeJob:
        with self._lock:
            if self._current and self._jobs.get(self._current) and self._jobs[self._current].state == "running":
                raise RuntimeError("a see job is already running — wait for it to finish")
            job_id = uuid.uuid4().hex[:12]
            root = self.see_root / job_id
            root.mkdir(parents=True, exist_ok=True)
            job = SeeJob(id=job_id, root=root, cfg=self.cfg)
            self._jobs[job_id] = job
            self._current = job_id
        t = threading.Thread(target=self._run, args=(job,), name=f"see-{job_id}", daemon=True)
        t.start()
        return job

    def preview_still(self, *, settle_s: float = 0.55) -> dict[str, Any]:
        """Grab one JPEG for the UI after a head move — no retune / VL.

        Writes ``/see/live/still.jpg`` (cache-busted by the client).
        """
        cam_cfg = self.cfg.get("camera", {})
        scan_cfg = self.cfg.get("scan", {})
        jpeg_q = int(cam_cfg.get("jpeg_quality", scan_cfg.get("jpeg_quality", DEFAULT_JPEG_QUALITY)))
        live = self.see_root / "live"
        live.mkdir(parents=True, exist_ok=True)
        # Short wait: analysis holds the cam only during capture/retune/burst.
        if not self._cam_lock.acquire(timeout=2.5):
            raise RuntimeError("camera busy retuning — try again in a second")
        cap = None
        try:
            time.sleep(max(0.0, float(settle_s)))
            if self._grab is not None:
                jpeg = self._grab()
            else:
                open_fn = self._open_camera or _open_camera
                cap = open_fn(cam_cfg)
                jpeg = _grab_jpeg(cap, quality=jpeg_q)
            path = live / "still.jpg"
            path.write_bytes(jpeg)
            return {
                "url": "/see/live/still.jpg",
                "bytes": len(jpeg),
                "settle_s": round(float(settle_s), 3),
            }
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:  # noqa: BLE001
                    pass
            self._cam_lock.release()

    def _url(self, job: SeeJob, name: str) -> str:
        return f"/see/{job.id}/{name}"

    def _write_jpeg(self, job: SeeJob, name: str, data: bytes) -> str:
        path = job.root / name
        path.write_bytes(data)
        url = self._url(job, name)
        with job._lock:
            job.images[name.replace(".jpg", "")] = url
            # also map stem keys: raw, best, overlay
            stem = Path(name).stem
            job.images[stem] = url
            job._touch()
        return url

    def _run(self, job: SeeJob) -> None:
        job.state = "running"
        job._touch()
        cap = None
        own_cap = False
        best = b""
        scan_cfg = job.cfg.get("scan", {})
        cam_cfg = job.cfg.get("camera", {})
        brain = job.cfg.get("brain", {})
        jpeg_q = int(cam_cfg.get("jpeg_quality", scan_cfg.get("jpeg_quality", DEFAULT_JPEG_QUALITY)))
        try:
            # Camera section — release before slow VL so head previews still work.
            with self._cam_lock:
                # --- capture ---
                t0 = time.perf_counter()
                job.set_stage("capture", status="running", progress=10, detail="Opening camera…")
                if self._grab is not None:
                    grab = self._grab
                else:
                    open_fn = self._open_camera or _open_camera
                    cap = open_fn(cam_cfg)
                    own_cap = True
                    grab = lambda: _grab_jpeg(cap, quality=jpeg_q)
                job.set_stage("capture", progress=40, detail="Grabbing still…")
                raw = grab()
                self._write_jpeg(job, "raw.jpg", raw)
                self._write_jpeg(job, "best.jpg", raw)
                job.set_stage(
                    "capture", status="done", progress=100,
                    detail=f"{len(raw)} bytes", seconds=time.perf_counter() - t0,
                )

                # --- score ---
                t0 = time.perf_counter()
                job.set_stage("score", status="running", progress=20, detail="Scoring frame…")
                sc, metrics = score_frame(raw, "see")
                with job._lock:
                    job.camera = {
                        "score": sc,
                        "metrics": metrics,
                        "settings": {},
                        "diagnosis": [],
                    }
                job.set_stage(
                    "score", status="done", progress=100,
                    detail=f"score={sc:.0f} lap={metrics.get('lap_var', 0):.0f} mean={metrics.get('mean', 0):.0f}",
                    seconds=time.perf_counter() - t0,
                )

                # --- retune ---
                t0 = time.perf_counter()
                do_retune = bool(scan_cfg.get("retune", True)) and cap is not None
                if do_retune:
                    job.set_stage("retune", status="running", progress=5, detail="Guided camera cleanup…")
                    retune_settle = float(scan_cfg.get("retune_settle_ms", DEFAULT_RETUNE_SETTLE_S * 1000)) / 1000.0
                    order, sweeps = sweeps_from_config(cam_cfg)
                    result = retune_pose(
                        cap, grab, pose="see", settle_s=retune_settle,
                        order=order, sweeps=sweeps,
                        mode=str(scan_cfg.get("retune_mode", "guided")),
                        good_enough=float(scan_cfg.get("retune_good_enough", 75.0)),
                        max_trials=int(scan_cfg.get("retune_max_trials", 12)),
                    )
                    best = result.jpeg
                    self._write_jpeg(job, "best.jpg", best)
                    with job._lock:
                        job.camera = {
                            "score": result.score,
                            "metrics": result.metrics,
                            "settings": dict(result.settings),
                            "diagnosis": list(result.diagnosis),
                            "trials": len(result.trials),
                            "mode": result.mode,
                        }
                    job.set_stage(
                        "retune", status="done", progress=100,
                        detail=f"score={result.score:.0f}; settings={result.settings or '{}'}",
                        seconds=time.perf_counter() - t0,
                    )
                else:
                    job.set_stage(
                        "retune", status="skipped", progress=100,
                        detail="no live camera handle (or retune disabled)",
                        seconds=time.perf_counter() - t0,
                    )
                    best = raw

                # --- burst ---
                t0 = time.perf_counter()
                burst_n = int(scan_cfg.get("burst_frames", 5))
                if burst_n > 1:
                    job.set_stage("burst", status="running", progress=10, detail=f"Burst of {burst_n}…")
                    best, bscore, bmeta = burst_best(grab, n=burst_n, pose="see", pause_s=0.08)
                    self._write_jpeg(job, "best.jpg", best)
                    with job._lock:
                        job.camera["score"] = bscore
                        job.camera["metrics"] = bmeta
                        job.camera["burst"] = burst_n
                    job.set_stage(
                        "burst", status="done", progress=100,
                        detail=f"kept score={bscore:.0f}", seconds=time.perf_counter() - t0,
                    )
                else:
                    job.set_stage("burst", status="skipped", progress=100, detail="burst_frames≤1")

                if own_cap and cap is not None:
                    try:
                        cap.release()
                    except Exception:  # noqa: BLE001
                        pass
                    cap = None
                    own_cap = False


            # --- inventory (no camera) ---
            t0 = time.perf_counter()
            base_url = scan_cfg.get("base_url") or brain.get("base_url")
            model = scan_cfg.get("model", "ministral-3:3b")
            api_key = scan_cfg.get("api_key") or brain.get("api_key") or ""
            timeout = float(scan_cfg.get("timeout", 180.0))
            if base_url:
                job.set_stage("inventory", status="running", progress=15, detail=f"VL inventory via {model}…")
                vision = self._vision or VisionLister(
                    base_url=base_url, model=model, api_key=api_key or None,
                    timeout=timeout, transport=curl_transport,
                )
                objs = vision.list_objects([("see", best)])
                with job._lock:
                    job.inventory = list(objs)
                (job.root / "inventory.json").write_text(json.dumps({"objects": objs}, indent=2) + "\n")
                job.set_stage(
                    "inventory", status="done", progress=100,
                    detail=", ".join(objs) if objs else "(none)",
                    seconds=time.perf_counter() - t0,
                )
            else:
                job.set_stage(
                    "inventory", status="skipped", progress=100,
                    detail="no [scan].base_url",
                )

            # --- ground (GPU /v1/detect, VL boxes as fallback) ---
            t0 = time.perf_counter()
            det_cfg = job.cfg.get("detect", {})
            has_detect = bool(
                det_cfg.get("detect_url") or det_cfg.get("base_url")
                or os.environ.get("ZEROLLAMA_DETECT_URL")
                or os.environ.get("BRUHOS_DETECT_URL")
            )
            if base_url or has_detect or self._ground_fn is not None:
                backend_hint = "detect→VL" if has_detect and base_url else (
                    "detect" if has_detect else str(scan_cfg.get("ground_model") or model)
                )
                job.set_stage(
                    "ground", status="running", progress=10,
                    detail=f"Grounding via {backend_hint}…",
                )
                w = int(cam_cfg.get("width", 640))
                h = int(cam_cfg.get("height", 480))
                hfov = float(cam_cfg.get("hfov_deg", 62.0))
                hints = list(job.inventory)
                cam = CameraModel(width=w, height=h, hfov_deg=hfov)
                if self._ground_fn is not None:
                    fr = self._ground_fn(
                        label="see", jpeg=best, base_url=base_url or "",
                        model=str(scan_cfg.get("ground_model") or model),
                        timeout=timeout, api_key=api_key,
                        cam=cam, transport=curl_transport, hint_objects=hints,
                    )
                else:
                    fr = ground_image(
                        label="see", jpeg=best, cam=cam, cfg=job.cfg,
                        hint_objects=hints, transport=curl_transport,
                    )
                if fr.raw_preview:
                    (job.root / "ground_raw.txt").write_text(fr.raw_preview + "\n")
                if not fr.ok:
                    job.set_stage(
                        "ground", status="error", progress=100,
                        detail=fr.error or "ground failed", error=fr.error,
                        seconds=time.perf_counter() - t0,
                    )
                else:
                    dets = [d.to_dict() for d in fr.detections]
                    with job._lock:
                        job.detections = dets
                    if dets:
                        overlay = draw_overlay(best, fr.detections, title="see")
                        self._write_jpeg(job, "overlay.jpg", overlay)
                        labs = ", ".join(d["label"] for d in dets)
                        detail = f"{labs} · {fr.backend}"
                    elif hints:
                        detail = (
                            f"named {len(hints)} object(s) but {fr.backend} "
                            "returned no boxes"
                        )
                    else:
                        detail = f"(none) · {fr.backend}"
                    (job.root / "detections.json").write_text(
                        json.dumps({"backend": fr.backend, "detections": dets}, indent=2) + "\n"
                    )
                    job.set_stage(
                        "ground", status="done", progress=100,
                        detail=detail, seconds=time.perf_counter() - t0,
                    )
            else:
                job.set_stage(
                    "ground", status="skipped", progress=100,
                    detail="no [detect].detect_url and no [scan].base_url",
                )

            job.state = "done"
            job._touch()
            (job.root / "status.json").write_text(json.dumps(job.to_dict(), indent=2) + "\n")
        except Exception as e:  # noqa: BLE001
            job.state = "error"
            job.error = str(e)[:500]
            # mark current running stage as error
            for s in job.stages:
                if s.status == "running":
                    job.set_stage(s.id, status="error", progress=100, error=str(e)[:300], detail=str(e)[:200])
                    break
            job._touch()
            try:
                (job.root / "status.json").write_text(json.dumps(job.to_dict(), indent=2) + "\n")
            except OSError:
                pass
        finally:
            if own_cap and cap is not None:
                try:
                    cap.release()
                except Exception:  # noqa: BLE001
                    pass
            with self._lock:
                if self._current == job.id:
                    self._current = None
