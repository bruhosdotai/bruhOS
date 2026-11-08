"""ColorBlobDetector must report every visible blob, not just the largest per color."""

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from bruhos.body.perception import ColorBlobDetector


def _frame(*rects):
    """Black 480x640 BGR frame with the given (x, y, w, h, bgr) rectangles."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for x, y, w, h, bgr in rects:
        frame[y:y + h, x:x + w] = bgr
    return frame


def test_two_blocks_of_the_same_color_are_both_reported():
    frame = _frame((100, 100, 100, 100, (0, 0, 255)),   # red, 100x100
                   (400, 300, 60, 60, (0, 0, 255)),     # red, 60x60
                   (400, 100, 80, 80, (255, 0, 0)))     # blue, 80x80
    dets = ColorBlobDetector().detect(frame)
    reds = [d for d in dets if d.color == "red"]
    assert len(reds) == 2
    assert len(dets) == 3


def test_single_block_detection_is_unchanged():
    frame = _frame((100, 100, 100, 100, (0, 0, 255)))
    dets = ColorBlobDetector().detect(frame)
    assert [d.name for d in dets] == ["red_block"]


def test_specks_below_min_area_are_still_ignored():
    frame = _frame((100, 100, 100, 100, (0, 0, 255)),
                   (400, 300, 10, 10, (0, 0, 255)))     # red speck, 100 px < 300 min area
    dets = ColorBlobDetector().detect(frame)
    assert [d.name for d in dets] == ["red_block"]
