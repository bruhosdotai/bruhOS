"""Pi-local detectors for the control-rate hot loop.

- ``HsvBlobBackend`` — existing tabletop color blobs (fast, always available)
- ``OnnxCocoBackend`` — nano COCO ONNX when ``onnxruntime`` + weights are present
- ``LocalDetectHelper`` — same JSON shape as GPU ``/v1/detect``, in-process

Never starts ``zerollama serve``. Heavy open-vocab (OWLv2, Grounding DINO) stays
on the GPU box behind ``ZEROLLAMA_DETECT_URL``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from bruhos.body.perception import CameraModel, ColorBlobDetector
from bruhos.detect.types import BBox, DetectRequest, DetectResponse, frame_to_jpeg
from bruhos.schemas import Detection


class FrameDetector(Protocol):
    """Hot-loop: BGR frame → bruhOS Detection list."""

    def detect(self, frame) -> list[Detection]: ...


class DetectBackend(Protocol):
    """Same contract as remote ``/v1/detect`` (request/response JSON)."""

    def detect(self, req: DetectRequest) -> DetectResponse: ...


@dataclass
class HsvBlobBackend:
    """Wrap ``ColorBlobDetector`` in the shared detect JSON shape."""

    camera: CameraModel = field(default_factory=CameraModel)
    hsv: dict | None = None
    label: str = "block"
    size_m: float = 0.04

    def __post_init__(self) -> None:
        kw: dict[str, Any] = {"camera": self.camera, "label": self.label, "size_m": self.size_m}
        if self.hsv:
            kw["hsv"] = self.hsv
        self._blob = ColorBlobDetector(**kw)

    def detect_frame(self, frame) -> list[Detection]:
        return self._blob.detect(frame)

    def detect(self, req: DetectRequest) -> DetectResponse:
        import math

        import cv2
        import numpy as np

        t0 = time.perf_counter()
        arr = np.frombuffer(req.image_jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("bad jpeg")
        h, w = img.shape[:2]
        dets = self._blob.detect(img)
        # HSV path has no xyxy today — approximate a small box on the bearing ray.
        boxes: list[BBox] = []
        space = (req.coordinate_space or "normalized").casefold()
        for d in dets:
            cx = w / 2.0 + self.camera.focal_px * math.tan(math.radians(-d.bearing_deg))
            half = 24.0
            x1, y1, x2, y2 = cx - half, h / 2 - half, cx + half, h / 2 + half
            if space == "normalized":
                boxes.append(BBox(
                    label=d.name, x1=x1 / w, y1=y1 / h, x2=x2 / w, y2=y2 / h,
                    confidence=d.confidence, normalized=True,
                ))
            else:
                boxes.append(BBox(
                    label=d.name, x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=d.confidence, normalized=False,
                ))
        return DetectResponse(
            detections=boxes, model=req.model or "hsv-blob", status="ok",
            width=w, height=h, coordinate_space=space,
            backend="local:hsv", seconds=round(time.perf_counter() - t0, 3),
        )


@dataclass
class OnnxCocoBackend:
    """Nano COCO ONNX (YOLOv8n-class) via onnxruntime — optional dep.

    Install: ``pip install onnxruntime`` and point ``model_path`` at a ``.onnx`` export.
    """

    model_path: str | Path
    camera: CameraModel = field(default_factory=CameraModel)
    confidence: float = 0.25
    input_size: int = 640
    _session: Any = field(default=None, repr=False, init=False)
    _labels: list[str] = field(default_factory=list, repr=False)

    def _ensure(self) -> None:
        if self._session is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime not installed — Pi hot-loop ONNX needs "
                "`pip install onnxruntime` (or use backend=hsv)"
            ) from e
        path = Path(self.model_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {path}")
        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"],
        )
        # COCO-80 default names if sidecar missing
        names_path = path.with_suffix(".names")
        if names_path.is_file():
            self._labels = [ln.strip() for ln in names_path.read_text().splitlines() if ln.strip()]
        else:
            self._labels = _COCO80

    def detect_frame(self, frame) -> list[Detection]:
        from bruhos.detect.types import bboxes_to_detections

        jpeg = frame_to_jpeg(frame)
        resp = self.detect(DetectRequest(
            image_jpeg=jpeg, queries=list(_COCO80), threshold=self.confidence,
        ))
        return bboxes_to_detections(resp.detections, self.camera)

    def detect(self, req: DetectRequest) -> DetectResponse:
        import cv2
        import numpy as np

        self._ensure()
        t0 = time.perf_counter()
        arr = np.frombuffer(req.image_jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("bad jpeg")
        h0, w0 = img.shape[:2]
        conf_min = float(req.threshold if req.threshold is not None else self.confidence)
        size = self.input_size
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (size, size), swapRB=True, crop=False)
        inp = self._session.get_inputs()[0].name
        out = self._session.run(None, {inp: blob})[0]
        boxes = _parse_yolo_onnx(out, w0, h0, size, conf_min, self._labels)
        if req.queries:
            allow = {c.casefold() for c in req.queries}
            boxes = [b for b in boxes if b.label.casefold() in allow]
        space = (req.coordinate_space or "normalized").casefold()
        if space == "normalized":
            boxes = [
                BBox(b.label, b.x1 / w0, b.y1 / h0, b.x2 / w0, b.y2 / h0, b.confidence, True)
                for b in boxes
            ]
        if req.max_detections is not None and req.max_detections > 0:
            boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)[: int(req.max_detections)]
        return DetectResponse(
            detections=boxes,
            model=req.model or str(self.model_path),
            status="ok",
            width=w0, height=h0,
            coordinate_space=space,
            backend="local:onnx-coco",
            seconds=round(time.perf_counter() - t0, 3),
        )


def _parse_yolo_onnx(
    out, w0: int, h0: int, size: int, conf_min: float, labels: list[str],
) -> list[BBox]:
    """Best-effort YOLOv8 ONNX output parse (1, 84, N) or (1, N, 84)."""
    import numpy as np

    x = np.asarray(out)
    if x.ndim == 3:
        x = x[0]
    # (84, N) vs (N, 84)
    if x.shape[0] < x.shape[1] and x.shape[0] <= 85:
        x = x.T
    boxes: list[BBox] = []
    sx, sy = w0 / size, h0 / size
    for row in x:
        if row.shape[0] < 6:
            continue
        # cx, cy, w, h, class scores...
        cx, cy, bw, bh = row[:4]
        scores = row[4:]
        cls_id = int(scores.argmax())
        conf = float(scores[cls_id])
        if conf < conf_min:
            continue
        x1 = (float(cx) - float(bw) / 2) * sx
        y1 = (float(cy) - float(bh) / 2) * sy
        x2 = (float(cx) + float(bw) / 2) * sx
        y2 = (float(cy) + float(bh) / 2) * sy
        label = labels[cls_id] if 0 <= cls_id < len(labels) else str(cls_id)
        boxes.append(BBox(label, x1, y1, x2, y2, conf, normalized=False))
    return boxes


@dataclass
class LocalDetectHelper:
    """In-process ``/v1/detect`` twin — same JSON as the GPU sibling."""

    backend: DetectBackend

    def detect(self, req: DetectRequest) -> DetectResponse:
        return self.backend.detect(req)


# Minimal COCO-80 for nano YOLO when no .names sidecar
_COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
