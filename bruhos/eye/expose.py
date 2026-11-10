"""Short exposure climb while eye owns the camera.

A black frame with the lights off stays black. A stuck auto-exposure does not.
Try a few UVC knobs, keep the brightest, and stop once the frame is lit.
"""

from __future__ import annotations

import time
from typing import Callable

Measure = Callable[[], float | None]
SetProp = Callable[[str, float], bool]
ReadProp = Callable[[str], float | None]

# V4L2 auto-exposure: 1 is manual on this camera; 3 and 0.75 are auto modes
# OpenCV already accepts. Manual exposure only matters after auto is off.
def signal_mean(gray) -> float:
    """Mean brightness, or 0 when the frame is a flat field.

    Raising UVC brightness on a dark sensor can pin every pixel to the same
    value. That mean is not light.
    """
    import numpy as np

    arr = np.asarray(gray)
    if arr.size == 0 or float(arr.std()) < 2.0:
        return 0.0
    return float(arr.mean())


TRIES: tuple[tuple[str, float], ...] = (
    ("auto_exposure", 3.0),
    ("auto_exposure", 0.75),
    ("brightness", 180.0),
    ("brightness", 220.0),
    ("auto_exposure", 1.0),
    ("exposure", 200.0),
    ("exposure", 500.0),
)


def lift_dark(
    set_prop: SetProp,
    measure: Measure,
    *,
    min_mean: float,
    settle_s: float = 0.0,
    tries: tuple[tuple[str, float], ...] = TRIES,
    read_prop: ReadProp | None = None,
) -> tuple[float, dict[str, float]]:
    """Return ``(best_mean, settings)``. ``settings`` is empty when already lit.

    If no knob beats the starting frame, the starting controls are put back
    so a dark room does not get left on manual exposure.
    """
    baseline: dict[str, float] = {}
    if read_prop is not None:
        for name in ("auto_exposure", "exposure", "brightness"):
            raw = read_prop(name)
            if raw is not None:
                baseline[name] = float(raw)
    base = measure()
    start = float(base or 0.0)
    best_mean = start
    if best_mean >= min_mean:
        return best_mean, {}
    best: dict[str, float] = {}
    applied: dict[str, float] = {}
    for name, value in tries:
        if name == "exposure":
            if set_prop("auto_exposure", 1.0):
                applied["auto_exposure"] = 1.0
        if not set_prop(name, float(value)):
            continue
        applied[name] = float(value)
        if settle_s > 0:
            time.sleep(settle_s)
        measure()
        got = measure()
        if got is None:
            continue
        if float(got) > best_mean + 1.0:
            best_mean = float(got)
            best = dict(applied)
        if best_mean >= min_mean:
            break
    if not best:
        for name, value in baseline.items():
            set_prop(name, value)
        return start, {}
    for name, value in best.items():
        set_prop(name, value)
    return best_mean, best
