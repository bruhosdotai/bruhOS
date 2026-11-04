"""Monocular metric depth via Depth Anything V2 (indoor Small).

Upstream: https://github.com/DepthAnything/Depth-Anything-V2 (metric_depth)
Checkpoint: depth_anything_v2_metric_hypersim_vits.pth → depth in metres.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_VENDOR_METRIC = _REPO / "vendor" / "Depth-Anything-V2" / "metric_depth"
_DEFAULT_CKPT = _REPO / "models" / "depth" / "depth_anything_v2_metric_hypersim_vits.pth"

_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


@dataclass(frozen=True)
class DepthBand:
    """Summary distances (m) over image bands. Lower = closer."""

    center_m: float
    left_m: float
    right_m: float
    min_m: float
    elapsed_s: float


class MetricDepth:
    """Lazy-loaded indoor metric Depth-Anything-V2 Small."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        encoder: str = "vits",
        max_depth: float = 20.0,
        input_size: int = 308,
        device: str | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else _DEFAULT_CKPT
        self.encoder = encoder
        self.max_depth = max_depth
        self.input_size = input_size
        self._device = device
        self._model: Any = None

    def _ensure(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"depth checkpoint missing: {self.checkpoint} "
                "(download Depth-Anything-V2-Metric-Hypersim-Small)"
            )
        if not _VENDOR_METRIC.is_dir():
            raise FileNotFoundError(f"vendor tree missing: {_VENDOR_METRIC}")
        try:
            import torch
        except ImportError as e:
            raise ImportError("depth needs torch (+ torchvision); pip install 'bruhos[depth]'") from e

        root = str(_VENDOR_METRIC)
        if root not in sys.path:
            sys.path.insert(0, root)
        from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = {**_MODEL_CONFIGS[self.encoder], "max_depth": self.max_depth}
        model = DepthAnythingV2(**cfg)
        state = torch.load(str(self.checkpoint), map_location="cpu")
        model.load_state_dict(state)
        model = model.to(device).eval()
        self._device = device
        self._model = model
        return model

    def infer(self, bgr: np.ndarray) -> np.ndarray:
        """HxW float32 depth map in metres."""
        model = self._ensure()
        return model.infer_image(bgr, self.input_size)

    def bands(
        self,
        bgr: np.ndarray,
        *,
        mid_frac: float = 0.4,
        row_lo: float = 0.35,
        row_hi: float = 0.85,
    ) -> DepthBand:
        """Median metres in left / center / right strips of the lower FOV."""
        t0 = time.monotonic()
        depth = self.infer(bgr)
        h, w = depth.shape
        r0, r1 = int(h * row_lo), int(h * row_hi)
        c0 = int(w * (0.5 - mid_frac / 2))
        c1 = int(w * (0.5 + mid_frac / 2))
        strip = depth[r0:r1, :]
        left = float(np.median(strip[:, :c0]))
        center = float(np.median(strip[:, c0:c1]))
        right = float(np.median(strip[:, c1:]))
        return DepthBand(
            center_m=center,
            left_m=left,
            right_m=right,
            min_m=float(np.min(strip)),
            elapsed_s=time.monotonic() - t0,
        )


def grab_bgr(cap: Any) -> np.ndarray | None:
    """Read one frame from an OpenCV VideoCapture (drop a buffered frame)."""
    if cap is None:
        return None
    cap.read()
    ok, frame = cap.read()
    return frame if ok else None
