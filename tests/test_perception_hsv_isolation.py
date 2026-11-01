"""Detector calibration must not change another detector's color recognition."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from bruhos.body.perception import ColorBlobDetector, DEFAULT_HSV


def test_tuning_one_detector_does_not_change_other_detectors():
    frame = np.zeros((128, 128, 3), dtype=np.uint8)
    frame[24:104, 24:104] = (0, 0, 255)
    tuned, independent = ColorBlobDetector(), ColorBlobDetector()
    original_ranges = list(DEFAULT_HSV["red"])
    try:
        assert [d.name for d in independent.detect(frame)] == ["red_block"]
        tuned.hsv["red"].clear()  # Disable red recognition only on this calibrated detector.
        assert tuned.detect(frame) == []
        assert [d.name for d in independent.detect(frame)] == ["red_block"]
        assert [d.name for d in ColorBlobDetector().detect(frame)] == ["red_block"]
    finally:
        # The base revision aliases this list globally; avoid contaminating other tests.
        DEFAULT_HSV["red"][:] = original_ranges
