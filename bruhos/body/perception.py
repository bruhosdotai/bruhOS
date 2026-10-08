"""On-robot perception: camera frame -> compressed Detections. Raw video never leaves this module.

The default detector is HSV color-blob segmentation (OpenCV), which is what the TonyPi tabletop
tasks need. YOLO / MediaPipe detectors plug in by implementing ``Detector``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from bruhos.schemas import Detection, LineReading

# OpenCV HSV ranges (H 0-180). Tune per lighting; override in configs/*.toml.
DEFAULT_HSV = {
    "red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    "green": [((40, 80, 60), (85, 255, 255))],
    "blue": [((95, 120, 60), (130, 255, 255))],
    "yellow": [((20, 120, 100), (35, 255, 255))],
}


@dataclass
class CameraModel:
    width: int = 640
    height: int = 480
    hfov_deg: float = 62.0

    @property
    def focal_px(self) -> float:
        return (self.width / 2) / math.tan(math.radians(self.hfov_deg / 2))

    def bearing(self, cx: float) -> float:
        """+ left / - right, degrees."""
        return -math.degrees(math.atan((cx - self.width / 2) / self.focal_px))

    def distance(self, size_px: float, size_m: float) -> float | None:
        return round(size_m * self.focal_px / size_px, 3) if size_px > 0 else None


class Detector(Protocol):
    def detect(self, frame) -> list[Detection]: ...


@dataclass
class ColorBlobDetector:
    camera: CameraModel = field(default_factory=CameraModel)
    hsv: dict = field(default_factory=lambda: dict(DEFAULT_HSV))
    label: str = "block"
    size_m: float = 0.04                 # TonyPi tabletop blocks are ~4 cm
    min_area_px: int = 300

    def detect(self, frame) -> list[Detection]:
        import cv2
        import numpy as np

        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        out = []
        for color, ranges in self.hsv.items():
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area < self.min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(c)
            out.append(Detection(self.label, color, round(self.camera.bearing(x + w / 2), 2),
                                 self.camera.distance(max(w, h), self.size_m),
                                 round(min(1.0, area / (w * h)), 3)))
        return out


@dataclass
class LineDetector:
    """Dark tape on a light floor, read from the bottom band of the frame."""

    threshold: int = 70
    band: float = 0.25

    def read(self, frame) -> LineReading:
        import cv2

        h, w = frame.shape[:2]
        roi = cv2.cvtColor(frame[int(h * (1 - self.band)):, :], cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(roi, self.threshold, 255, cv2.THRESH_BINARY_INV)
        m = cv2.moments(mask)
        if m["m00"] < 255 * w * 2:
            return LineReading(False, 0.0)
        cx = m["m10"] / m["m00"]
        return LineReading(True, round((cx - w / 2) / (w / 2), 3))
