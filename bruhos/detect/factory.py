"""Build Pi hot-loop + optional GPU detect clients from config."""

from __future__ import annotations

import os
from typing import Any

from bruhos.body.perception import CameraModel
from bruhos.detect.local import HsvBlobBackend, LocalDetectHelper, OnnxCocoBackend
from bruhos.detect.remote import RemoteDetectClient


def camera_from_cfg(cfg: dict[str, Any]) -> CameraModel:
    cam = cfg.get("camera", {})
    return CameraModel(
        width=int(cam.get("width", 640)),
        height=int(cam.get("height", 480)),
        hfov_deg=float(cam.get("hfov_deg", 62.0)),
    )


def make_local_backend(cfg: dict[str, Any]):
    """Hot-loop backend: ``hsv`` (default) or ``onnx`` (nano COCO)."""
    det = cfg.get("detect", {})
    backend = str(det.get("backend", det.get("local", "hsv"))).casefold()
    cam = camera_from_cfg(cfg)
    if backend in ("hsv", "blob", "color"):
        hsv = cfg.get("perception", {}).get("hsv")
        return HsvBlobBackend(camera=cam, hsv=hsv)
    if backend in ("onnx", "onnx-coco", "yolo", "nano"):
        path = det.get("onnx_path") or det.get("model_path")
        if not path:
            raise ValueError("[detect] onnx_path required for backend=onnx")
        return OnnxCocoBackend(
            model_path=path,
            camera=cam,
            confidence=float(det.get("confidence", 0.25)),
            input_size=int(det.get("input_size", 640)),
        )
    raise ValueError(f"unknown [detect].backend {backend!r} (use hsv|onnx)")


def make_local_helper(cfg: dict[str, Any]) -> LocalDetectHelper:
    """In-process helper with the same JSON shape as GPU ``/v1/detect``."""
    return LocalDetectHelper(backend=make_local_backend(cfg))


def make_remote_client(cfg: dict[str, Any]) -> RemoteDetectClient | None:
    """GPU detect client, or None if no URL configured.

    Env ``ZEROLLAMA_DETECT_URL`` wins over ``[detect].detect_url``.
    """
    det = cfg.get("detect", {})
    url = (
        os.environ.get("ZEROLLAMA_DETECT_URL")
        or os.environ.get("BRUHOS_DETECT_URL")
        or det.get("detect_url")
        or det.get("base_url")
    )
    if not url:
        return None
    return RemoteDetectClient(
        detect_url=str(url),
        model=str(det.get("remote_model") or det.get("model") or "detect"),
        timeout=float(det.get("timeout", 60.0)),
        api_key=det.get("api_key") or cfg.get("brain", {}).get("api_key"),
        api_key_env=det.get("api_key_env", "OPENAI_API_KEY"),
        coordinate_space=str(det.get("coordinate_space", "normalized")),
    )


class _FrameAdapter:
    def __init__(self, backend):
        self._b = backend

    def detect(self, frame):
        return self._b.detect_frame(frame)


def make_hot_loop_detector(cfg: dict[str, Any]):
    """Always-local detector for control ticks (never HTTP / never zerollama)."""
    backend = make_local_backend(cfg)
    if isinstance(backend, HsvBlobBackend):
        return backend._blob
    return _FrameAdapter(backend)
