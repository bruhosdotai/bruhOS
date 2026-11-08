"""A camera model must reject a horizontal field of view the pinhole model cannot use.

Failing-test-first for a reproduced defect: ``CameraModel`` accepts any
``hfov_deg`` and only breaks later, inside ``bearing``/``distance``:

- ``hfov_deg=0`` constructs fine, then ``focal_px`` raises ``ZeroDivisionError``
  the first time a detection is computed -- a crash mid-observation, not a
  clean error at load.
- ``hfov_deg=180`` constructs fine and reports a block dead ahead at 90 degrees
  (``focal_px`` collapses to ~0), so the robot turns away from a centered target.
- A negative ``hfov_deg`` flips the sign of ``focal_px``, so bearings come out
  mirrored: left reads as right and the robot aims the wrong way.
- ``hfov_deg=nan`` constructs fine and poisons every bearing and distance with
  NaN, which ``look_at``/``walk_toward`` then compare against thresholds that
  never match.

The pinhole model is only defined for ``0 < hfov_deg < 180``. Reject anything
else at construction -- the same load-time rule open PR #120 applies to
``turn_step_deg`` -- so a bad ``[camera].hfov_deg`` fails fast instead of
steering the robot wrong.
"""

import math

import pytest

from bruhos.body.perception import CameraModel


@pytest.mark.parametrize("bad", [0.0, 180.0, 270.0, -10.0, float("nan"), float("inf")])
def test_unusable_hfov_deg_is_rejected_at_construction(bad):
    with pytest.raises(ValueError):
        CameraModel(640, 480, bad)


def test_hfov_zero_does_not_crash_mid_observation():
    # Before the fix this constructed fine and raised ZeroDivisionError from
    # focal_px the first time a bearing was computed.
    with pytest.raises(ValueError):
        CameraModel(640, 480, 0.0)


@pytest.mark.parametrize("good", [1.0, 62.0, 179.9])
def test_usable_hfov_deg_still_constructs(good):
    m = CameraModel(640, 480, good)
    assert math.isfinite(m.focal_px) and m.focal_px > 0


def test_bearing_sign_and_centre_for_valid_hfov():
    m = CameraModel(640, 480, 62.0)
    assert m.bearing(320) == 0.0  # frame centre is straight ahead
    assert m.bearing(100) > 0.0  # left of centre reads positive (left)
    assert m.bearing(540) < 0.0  # right of centre reads negative (right)
    assert m.distance(100, 0.04) is not None and m.distance(100, 0.04) > 0
