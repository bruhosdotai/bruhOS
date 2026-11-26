"""An unusable camera width must fail at load, not crash the first bearing."""

import math

import pytest

from bruhos.body.perception import CameraModel


@pytest.mark.parametrize("bad", [0, -640, -1, 1.5, "abc", True, None, float("nan"), float("inf"), -float("inf")])
def test_unusable_width_is_rejected(bad):
    with pytest.raises(ValueError):
        CameraModel(width=bad)


@pytest.mark.parametrize("good,expected", [(640, 640), (640.0, 640), ("640", 640), (1, 1)])
def test_valid_width_is_kept_as_an_int(good, expected):
    cam = CameraModel(width=good)
    assert cam.width == expected
    assert math.isfinite(cam.focal_px)
    assert cam.bearing(expected / 4) > 0  # left of centre reads positive
