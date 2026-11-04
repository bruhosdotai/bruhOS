"""Shared detect wire format matching zerollama ``POST /v1/detect`` (OpenAPI).

Pi hot-loop helpers reuse the same request/response shape in-process.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

from bruhos.body.perception import CameraModel
from bruhos.schemas import Detection


@dataclass
class BBox:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    normalized: bool = False  # True if coords are 0..1

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": self.confidence,
            "confidence": self.confidence,
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "normalized": self.normalized,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, default_normalized: bool = False) -> BBox:
        bbox = raw.get("bbox") or raw.get("xyxy") or raw.get("box")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            raise ValueError("bbox needs 4 numbers")
        label = raw.get("label") or raw.get("name") or raw.get("class") or "object"
        conf = raw.get("score", raw.get("confidence", raw.get("conf", 1.0)))
        return cls(
            label=str(label),
            x1=float(bbox[0]), y1=float(bbox[1]), x2=float(bbox[2]), y2=float(bbox[3]),
            confidence=float(conf),
            normalized=bool(raw.get("normalized", default_normalized)),
        )


@dataclass
class DetectRequest:
    """``POST /v1/detect`` body (zerollama robot detect)."""

    image_jpeg: bytes
    queries: list[str]
    model: str = "detect"
    threshold: float | None = None
    max_detections: int | None = None
    coordinate_space: str = "normalized"  # normalized | pixel
    frame_id: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        if not self.queries:
            raise ValueError("queries must be non-empty")
        out: dict[str, Any] = {
            "model": self.model or "detect",
            "queries": list(self.queries),
            "image": base64.b64encode(self.image_jpeg).decode("ascii"),
            "coordinate_space": self.coordinate_space or "normalized",
        }
        if self.threshold is not None:
            out["threshold"] = float(self.threshold)
        if self.max_detections is not None:
            out["max_detections"] = int(self.max_detections)
        if self.frame_id:
            out["frame_id"] = self.frame_id
        return out


@dataclass
class DetectResponse:
    detections: list[BBox] = field(default_factory=list)
    model: str | None = None
    status: str | None = None  # ok | wire_only
    backend: str | None = None
    coordinate_space: str | None = None
    width: int | None = None
    height: int | None = None
    frame_id: str | None = None
    threshold: float | None = None
    engine: dict[str, Any] = field(default_factory=dict)
    seconds: float | None = None

    @property
    def wire_only(self) -> bool:
        return (self.status or "").casefold() == "wire_only"

    @property
    def ok_real(self) -> bool:
        """True when a real detector answered (empty boxes may be a miss)."""
        return (self.status or "ok").casefold() == "ok"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "detections": [b.to_dict() for b in self.detections],
            "coordinate_space": self.coordinate_space or "normalized",
            "image_size": {
                "width": self.width or 0,
                "height": self.height or 0,
            },
        }
        if self.model is not None:
            d["model"] = self.model
        if self.status is not None:
            d["status"] = self.status
        if self.backend is not None:
            d["backend"] = self.backend
        if self.frame_id is not None:
            d["frame_id"] = self.frame_id
        if self.threshold is not None:
            d["threshold"] = self.threshold
        if self.engine:
            d["engine"] = dict(self.engine)
        if self.seconds is not None:
            d["seconds"] = self.seconds
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DetectResponse:
        space = raw.get("coordinate_space") or "normalized"
        default_norm = str(space).casefold() == "normalized"
        items = raw.get("detections") or raw.get("objects") or []
        dets: list[BBox] = []
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict):
                    try:
                        dets.append(BBox.from_dict(it, default_normalized=default_norm))
                    except (TypeError, ValueError):
                        continue
        size = raw.get("image_size") or {}
        w = raw.get("width", size.get("width") if isinstance(size, dict) else None)
        h = raw.get("height", size.get("height") if isinstance(size, dict) else None)
        engine = raw.get("engine") if isinstance(raw.get("engine"), dict) else {}
        return cls(
            detections=dets,
            model=raw.get("model"),
            status=raw.get("status"),
            backend=raw.get("backend"),
            coordinate_space=space,
            width=int(w) if w is not None else None,
            height=int(h) if h is not None else None,
            frame_id=raw.get("frame_id"),
            threshold=float(raw["threshold"]) if raw.get("threshold") is not None else None,
            engine=dict(engine or {}),
            seconds=float(raw["seconds"]) if raw.get("seconds") is not None else None,
        )


def bboxes_to_detections(
    boxes: list[BBox],
    cam: CameraModel,
    *,
    size_m: float | None = None,
) -> list[Detection]:
    """Map pixel/normalized boxes → bruhOS ``Detection`` (bearing / optional range)."""
    out: list[Detection] = []
    for b in boxes:
        x1, y1, x2, y2 = b.x1, b.y1, b.x2, b.y2
        if b.normalized or max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
            x1, x2 = x1 * cam.width, x2 * cam.width
            y1, y2 = y1 * cam.height, y2 * cam.height
        x1, x2 = sorted((float(x1), float(x2)))
        y1, y2 = sorted((float(y1), float(y2)))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        cx = (x1 + x2) / 2.0
        dist = cam.distance(max(x2 - x1, y2 - y1), size_m) if size_m else None
        out.append(Detection(
            label=b.label.casefold().strip(),
            color=None,
            bearing_deg=round(cam.bearing(cx), 2),
            distance_m=dist,
            confidence=float(b.confidence),
        ))
    return out


def frame_to_jpeg(frame, quality: int = 85) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return bytes(buf)
