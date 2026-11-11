"""A camera model must reject a frame width the pinhole model cannot use.

Failing-test-first for a reproduced defect: ``CameraModel`` accepts any
``width`` and only breaks later, inside ``bearing``/``distance``:

- ``width=0`` constructs fine, then ``focal_px`` is 0.0 and ``bearing``
  raises ``ZeroDivisionError`` the first time a blob is detected -- a crash
  mid-observation, uncaught by ``Runtime._run``, so ``bruhos run`` dies.
- A negative ``width`` constructs fine and reports wrong bearings: the frame
  centre sits off-frame at ``width / 2`` and right-side objects read with the
  wrong sign, so the robot aims the wrong way.
- ``width="abc"`` (wrong TOML type) constructs fine and raises ``TypeError``
  from ``focal_px`` the first time a bearing is computed.
- ``width=nan`` constructs fine and poisons every bearing with NaN, which
  ``look_at``/``walk_toward`` then compare against thresholds that never match.

The pinhole model is only defined for ``width > 0``. Reject anything else at
construction -- the same load-time rule open PR #222 applies to ``hfov_deg`` --
so a bad ``[camera].width`` fails fast instead of crashing or steering wrong.
"""

import math

import pytest

from bruhos.body.perception import CameraModel


@pytest.mark.parametrize("bad", [0, 0.0, -640, -1, float("nan"), float("inf"), "abc", True, None])
def test_unusable_width_is_rejected_at_construction(bad):
    with pytest.raises(ValueError):
        CameraModel(bad, 480, 62.0)


def test_width_zero_does_not_crash_mid_observation():
    # Before the fix this constructed fine and raised ZeroDivisionError from
    # bearing() the first time a detection was computed.
    with pytest.raises(ValueError):
        CameraModel(0, 480, 62.0)


@pytest.mark.parametrize("good", [1, 320, 640, 1920, "640"])
def test_usable_width_still_constructs(good):
    m = CameraModel(good, 480, 62.0)
    assert math.isfinite(m.focal_px) and m.focal_px > 0


def test_bearing_sign_and_centre_for_valid_width():
    m = CameraModel(640, 480, 62.0)
    assert m.bearing(320) == 0.0  # frame centre is straight ahead
    assert m.bearing(100) > 0.0  # left of centre reads positive (left)
    assert m.bearing(540) < 0.0  # right of centre reads negative (right)
    assert m.distance(100, 0.04) is not None and m.distance(100, 0.04) > 0
