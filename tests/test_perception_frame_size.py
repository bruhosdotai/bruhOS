"""Detections must be computed against the frame the camera actually delivered.

``TonyPiBody`` asks the camera for ``[camera].width x height`` and builds a ``CameraModel`` from
the same numbers, but V4L2 cameras silently keep their native size when a request is unsupported.
On ``main`` the detector used the configured centre and focal length on whatever frame arrived:
a block dead ahead in a 1280x960 frame read as 31 degrees to the right at half its distance.
"""

from __future__ import annotations

import math

import pytest

from bruhos.body.perception import CameraModel


def test_for_frame_rescales_the_model_and_keeps_the_lens():
    cam = CameraModel(640, 480, 62.0)
    big = cam.for_frame(1280, 960)
    assert (big.width, big.height, big.hfov_deg) == (1280, 960, 62.0)
    assert big.focal_px == pytest.approx(2 * cam.focal_px)
    assert big.bearing(640) == 0.0                         # centre of the delivered frame
    assert big.bearing(640 + 200) == pytest.approx(cam.bearing(320 + 100))   # same ray, twice the pixels
    assert big.distance(82, 0.04) == pytest.approx(cam.distance(41, 0.04), abs=1e-3)


def test_for_frame_is_identity_when_the_size_matches():
    cam = CameraModel(640, 480, 62.0)
    assert cam.for_frame(640, 480) is cam


# ---- with OpenCV (optional "vision" extra; skipped in CI) ----

def _frame(w, h, cx, cy, size, bgr=(0, 0, 255)):
    np = pytest.importorskip("numpy")
    cv2 = pytest.importorskip("cv2")
    f = np.full((h, w, 3), (200, 200, 200), np.uint8)
    cv2.rectangle(f, (cx - size // 2, cy - size // 2), (cx + size // 2, cy + size // 2), bgr, -1)
    return f


@pytest.mark.parametrize("w,h,size", [(1280, 960, 80), (320, 240, 20), (640, 480, 40)])
def test_centred_block_reads_dead_ahead_at_any_delivered_resolution(w, h, size):
    from bruhos.body.perception import ColorBlobDetector

    det = ColorBlobDetector(CameraModel(640, 480, 62.0))
    (d,) = det.detect(_frame(w, h, w // 2, h // 2, size))
    assert d.name == "red_block"
    assert abs(d.bearing_deg) < 0.5, d
    # 40 px of a 4 cm block at 640 px wide is 0.52 m; the same scene must read the same at any size
    assert d.distance_m == pytest.approx(0.52, abs=0.02), d


def test_off_centre_block_keeps_its_bearing_across_resolutions():
    from bruhos.body.perception import ColorBlobDetector

    det = ColorBlobDetector(CameraModel(640, 480, 62.0))
    (small,) = det.detect(_frame(640, 480, 320 + 100, 240, 40))
    (big,) = det.detect(_frame(1280, 960, 640 + 200, 480, 80))
    assert big.bearing_deg == pytest.approx(small.bearing_deg, abs=0.3)
    assert small.bearing_deg < -10                          # to the right, as before this change
