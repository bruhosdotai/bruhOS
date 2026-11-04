"""Per-pose camera retune driven by image feedback (few targeted steps, not full sweeps)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from bruhos.scan.focus import analyze_jpeg

GrabFn = Callable[[], bytes]

# Kept for optional ``mode=sweep`` / config overrides.
DEFAULT_SWEEPS: dict[str, list[float]] = {
    "auto_exposure": [0.75, 0.25, 3, 1],
    "exposure": [-13, -10, -7, -5, -3, 50, 100, 150, 200],
    "brightness": [64, 96, 128, 160, 192],
    "contrast": [64, 96, 128, 160, 192],
    "saturation": [64, 128, 192],
    "sharpness": [32, 64, 128, 192, 255],
    "gamma": [80, 100, 120, 160, 200],
    "backlight": [0, 1, 2],
    "wb_temperature": [2800, 4000, 4600, 5500, 6500],
}

DEFAULT_ORDER = (
    "auto_exposure",
    "exposure",
    "brightness",
    "contrast",
    "sharpness",
    "gamma",
    "saturation",
    "backlight",
    "wb_temperature",
)

# Stop early when a still is already good enough for VL.
GOOD_ENOUGH_SCORE = 75.0        # quality-first: only stop when still is clearly strong
# Hard cap on grabs after the baseline (guided mode).
DEFAULT_MAX_TRIALS = 12
DEFAULT_BURST = 5               # pick sharpest of N stills after retune


def _cv_prop(cv2, name: str) -> int | None:
    key = {
        "brightness": "CAP_PROP_BRIGHTNESS",
        "contrast": "CAP_PROP_CONTRAST",
        "saturation": "CAP_PROP_SATURATION",
        "hue": "CAP_PROP_HUE",
        "sharpness": "CAP_PROP_SHARPNESS",
        "gamma": "CAP_PROP_GAMMA",
        "backlight": "CAP_PROP_BACKLIGHT",
        "wb_temperature": "CAP_PROP_WB_TEMPERATURE",
        "auto_exposure": "CAP_PROP_AUTO_EXPOSURE",
        "exposure": "CAP_PROP_EXPOSURE",
    }.get(name)
    if not key:
        return None
    return int(getattr(cv2, key))


def score_frame(jpeg: bytes, pose: str = "x") -> tuple[float, dict[str, Any]]:
    """Combined sharpness + exposure health score (higher is better, ~0..100)."""
    m = analyze_jpeg(pose, jpeg)
    if m.get("verdict") == "unreadable":
        return 0.0, m
    lap = float(m["lap_var"])
    mean = float(m["mean"])
    p99 = float(m["p99"])
    sharp = min(100.0, 100.0 * lap / 120.0)
    if mean < 15 or mean > 230:
        exp = 0.0
    else:
        exp = 100.0 * (1.0 - min(1.0, abs(mean - 105.0) / 105.0))
    if p99 >= 250:
        exp *= 0.65
    if p99 < 60:
        exp *= 0.35
    if mean < 40:
        exp *= 0.4
    total = round(0.62 * sharp + 0.38 * exp, 2)
    m["tune_score"] = total
    return total, m


def diagnose(metrics: dict[str, Any], props: dict[str, float]) -> tuple[list[str], list[tuple[str, float]]]:
    """From one frame's metrics, return (issues, ordered (control, value) tries)."""
    mean = float(metrics.get("mean") or 0)
    p99 = float(metrics.get("p99") or 0)
    lap = float(metrics.get("lap_var") or 0)
    issues: list[str] = []
    actions: list[tuple[str, float]] = []

    def cur(name: str, default: float) -> float:
        return float(props.get(name, default))

    if mean < 45 or p99 < 70:
        issues.append("dark")
        b = cur("brightness", 128)
        actions.extend([
            ("brightness", min(220.0, b + 48)),
            ("brightness", min(240.0, max(160.0, b + 80))),
            ("gamma", min(220.0, cur("gamma", 100) + 40)),
            ("auto_exposure", 0.75),
            ("backlight", 1),
        ])
    elif mean > 175 or p99 >= 252:
        issues.append("bright")
        b = cur("brightness", 128)
        actions.extend([
            ("brightness", max(40.0, b - 48)),
            ("brightness", max(32.0, b - 80)),
            ("backlight", 0),
            ("auto_exposure", 0.25),
            ("exposure", -10),
            ("gamma", max(60.0, cur("gamma", 100) - 30)),
        ])

    if lap < 40 and mean >= 35:
        issues.append("soft")
        s = cur("sharpness", 128)
        c = cur("contrast", 128)
        actions.extend([
            ("sharpness", min(255.0, max(180.0, s + 64))),
            ("sharpness", 255),
            ("contrast", min(220.0, c + 40)),
        ])

    if not issues:
        issues.append("ok")

    # De-dupe while preserving order.
    seen: set[tuple[str, float]] = set()
    uniq: list[tuple[str, float]] = []
    for a in actions:
        key = (a[0], round(a[1], 3))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(a)
    return issues, uniq


@dataclass
class TuneTrial:
    control: str
    value: float
    score: float
    lap_var: float
    mean: float
    issue: str = ""


@dataclass
class TuneResult:
    jpeg: bytes
    score: float
    settings: dict[str, float] = field(default_factory=dict)
    trials: list[TuneTrial] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    diagnosis: list[str] = field(default_factory=list)
    mode: str = "guided"


def _read_prop(cap, cv2, name: str) -> float | None:
    pid = _cv_prop(cv2, name)
    if pid is None:
        return None
    return float(cap.get(pid))


def _set_prop(cap, cv2, name: str, value: float) -> bool:
    pid = _cv_prop(cv2, name)
    if pid is None:
        return False
    return bool(cap.set(pid, float(value)))


def _snapshot_props(cap, cv2, names: tuple[str, ...] = DEFAULT_ORDER) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in names:
        v = _read_prop(cap, cv2, name)
        if v is not None:
            out[name] = v
    return out


def sweeps_from_config(cam_cfg: dict[str, Any]) -> tuple[list[str], dict[str, list[float]]]:
    rt = cam_cfg.get("retune") if isinstance(cam_cfg.get("retune"), dict) else {}
    if not isinstance(rt, dict):
        rt = {}
    order = list(rt.get("order") or DEFAULT_ORDER)
    sweeps: dict[str, list[float]] = {}
    for name in order:
        raw = rt.get(name, DEFAULT_SWEEPS.get(name))
        if raw is None:
            continue
        sweeps[name] = [float(x) for x in raw]
    order = [n for n in order if n in sweeps]
    return order, sweeps


def retune_pose(
    cap,
    grab: GrabFn,
    *,
    pose: str = "pose",
    settle_s: float = 0.25,
    order: list[str] | None = None,
    sweeps: dict[str, list[float]] | None = None,
    mode: str = "guided",
    good_enough: float = GOOD_ENOUGH_SCORE,
    max_trials: int = DEFAULT_MAX_TRIALS,
    cv2_mod=None,
) -> TuneResult:
    """Retune using image feedback (default) or legacy full OFAT ``mode='sweep'``."""
    if mode == "sweep":
        return _retune_sweep(
            cap, grab, pose=pose, settle_s=settle_s,
            order=order, sweeps=sweeps, cv2_mod=cv2_mod,
        )
    return _retune_guided(
        cap, grab, pose=pose, settle_s=settle_s,
        good_enough=good_enough, max_trials=max_trials, cv2_mod=cv2_mod,
    )


def _retune_guided(
    cap,
    grab: GrabFn,
    *,
    pose: str,
    settle_s: float,
    good_enough: float,
    max_trials: int,
    cv2_mod=None,
) -> TuneResult:
    import cv2 as _cv2

    cv2 = cv2_mod or _cv2
    props = _snapshot_props(cap, cv2)

    jpeg = grab()
    score, metrics = score_frame(jpeg, pose)
    best_jpeg, best_score, best_metrics = jpeg, score, metrics
    chosen: dict[str, float] = {}
    issues, actions = diagnose(metrics, props)
    trials: list[TuneTrial] = [
        TuneTrial("baseline", 0.0, score, metrics.get("lap_var", 0.0), metrics.get("mean", 0.0),
                  issue=",".join(issues))
    ]

    # Quality-first: only skip work when the baseline is excellent *and* clean.
    if score >= good_enough and issues == ["ok"]:
        return TuneResult(
            jpeg=best_jpeg, score=best_score, settings=chosen, trials=trials,
            metrics=best_metrics, diagnosis=issues, mode="guided",
        )

    tried = 0
    # Up to three diagnose→act rounds; keep going while score still climbs.
    for _round in range(3):
        if tried >= max_trials:
            break
        props = _snapshot_props(cap, cv2)
        issues, actions = diagnose(best_metrics, props)
        if issues == ["ok"] and best_score >= good_enough:
            break
        round_best = best_score
        for control, value in actions:
            if tried >= max_trials:
                break
            prev = props.get(control)
            if prev is not None and abs(prev - value) < 1e-3:
                continue
            if not _set_prop(cap, cv2, control, value):
                continue
            time.sleep(max(0.0, settle_s))
            jpeg = grab()
            sc, met = score_frame(jpeg, pose)
            tried += 1
            trials.append(TuneTrial(
                control, value, sc, met.get("lap_var", 0.0), met.get("mean", 0.0),
                issue=",".join(issues),
            ))
            if sc > best_score + 0.25:
                best_score, best_jpeg, best_metrics = sc, jpeg, met
                chosen[control] = value
                props[control] = value
            elif prev is not None:
                _set_prop(cap, cv2, control, prev)
                time.sleep(max(0.0, settle_s * 0.25))
        if best_score >= good_enough and diagnose(best_metrics, props)[0] == ["ok"]:
            break
        if best_score <= round_best + 0.25:
            break  # this round did not improve — stop

    # Leave device on winning settings.
    for control, value in chosen.items():
        _set_prop(cap, cv2, control, value)

    return TuneResult(
        jpeg=best_jpeg,
        score=best_score,
        settings=chosen,
        trials=trials,
        metrics=best_metrics,
        diagnosis=issues,
        mode="guided",
    )


def _retune_sweep(
    cap,
    grab: GrabFn,
    *,
    pose: str,
    settle_s: float,
    order: list[str] | None,
    sweeps: dict[str, list[float]] | None,
    cv2_mod=None,
) -> TuneResult:
    """Legacy one-factor-at-a-time sweep (many steps)."""
    import cv2 as _cv2

    cv2 = cv2_mod or _cv2
    order = list(order or DEFAULT_ORDER)
    sweeps = dict(sweeps or DEFAULT_SWEEPS)
    baseline = _snapshot_props(cap, cv2)

    jpeg0 = grab()
    score0, metrics0 = score_frame(jpeg0, pose)
    best_jpeg, best_score, best_metrics = jpeg0, score0, metrics0
    chosen: dict[str, float] = {}
    trials: list[TuneTrial] = [
        TuneTrial("baseline", 0.0, score0, metrics0.get("lap_var", 0.0), metrics0.get("mean", 0.0))
    ]

    for control in order:
        values = sweeps.get(control) or []
        if not values:
            continue
        if control == "exposure":
            ae = chosen.get("auto_exposure", baseline.get("auto_exposure"))
            if ae is None or ae not in (0.25, 1.0, 1):
                continue

        local_best_score = best_score
        local_best_val: float | None = None
        local_best_jpeg = best_jpeg
        local_best_metrics = best_metrics
        start_val = chosen.get(control, baseline.get(control))

        for val in values:
            if not _set_prop(cap, cv2, control, val):
                continue
            time.sleep(max(0.0, settle_s))
            jpeg = grab()
            sc, met = score_frame(jpeg, pose)
            trials.append(TuneTrial(control, val, sc, met.get("lap_var", 0.0), met.get("mean", 0.0)))
            if sc > local_best_score:
                local_best_score = sc
                local_best_val = val
                local_best_jpeg = jpeg
                local_best_metrics = met

        if local_best_val is not None and local_best_score > best_score + 0.25:
            _set_prop(cap, cv2, control, local_best_val)
            time.sleep(max(0.0, settle_s * 0.5))
            chosen[control] = local_best_val
            best_score = local_best_score
            best_jpeg = local_best_jpeg
            best_metrics = local_best_metrics
        elif start_val is not None:
            _set_prop(cap, cv2, control, start_val)
            time.sleep(max(0.0, settle_s * 0.3))

    return TuneResult(
        jpeg=best_jpeg, score=best_score, settings=chosen, trials=trials,
        metrics=best_metrics, diagnosis=["sweep"], mode="sweep",
    )


def burst_best(grab: GrabFn, *, n: int = DEFAULT_BURST, pose: str = "pose",
               pause_s: float = 0.05) -> tuple[bytes, float, dict[str, Any]]:
    """Grab ``n`` stills and return the highest-scoring one (beats single-frame noise)."""
    n = max(1, int(n))
    best_jpeg: bytes | None = None
    best_score = -1.0
    best_metrics: dict[str, Any] = {}
    for i in range(n):
        if i:
            time.sleep(max(0.0, pause_s))
        jpeg = grab()
        sc, met = score_frame(jpeg, pose)
        if sc > best_score:
            best_score, best_jpeg, best_metrics = sc, jpeg, met
    assert best_jpeg is not None
    return best_jpeg, best_score, best_metrics


def format_retune_summary(per_pose: list[dict[str, Any]]) -> str:
    if not per_pose:
        return "(no retune)"
    lines = ["RETUNE (image-feedback, few steps)", "-" * 64]
    lines.append(f"{'pose':6}  {'score':>6}  {'n':>3}  diagnosis     settings")
    for row in per_pose:
        settings = row.get("settings") or {}
        s = ", ".join(f"{k}={v:g}" for k, v in settings.items()) or "(defaults)"
        diag = ",".join(row.get("diagnosis") or []) or "-"
        lines.append(
            f"{row['pose']:6}  {row['score']:6.1f}  {row.get('trials', 0):3d}  {diag:<12}  {s}"
        )
    return "\n".join(lines)
