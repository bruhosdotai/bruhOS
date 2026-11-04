"""Per-frame focus / sharpness analysis + overall score for a fixed lens setting."""

from __future__ import annotations

from typing import Any

# Map median Laplacian variance → 0..100 for this class of USB cam.
# ~120 lapVar on a well-lit still ≈ 100; tune if the sensor changes.
_SCORE_REF_LAP = 120.0


def analyze_jpeg(name: str, jpeg: bytes) -> dict[str, Any]:
    """Return sharpness metrics for one JPEG. Higher ``lap_var`` = sharper."""
    import cv2
    import numpy as np

    arr = np.frombuffer(jpeg, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return {
            "pose": name,
            "lap_var": 0.0,
            "lap_roi": 0.0,
            "tenengrad": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p99": 0.0,
            "verdict": "unreadable",
        }

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    ten = float((gx * gx + gy * gy).mean())
    h, w = gray.shape
    roi = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    lap_roi = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    mean = float(gray.mean())
    p01, p50, p99 = (float(x) for x in np.percentile(gray, [1, 50, 99]))
    return {
        "pose": name,
        "lap_var": round(lap_var, 1),
        "lap_roi": round(lap_roi, 1),
        "tenengrad": round(ten, 1),
        "mean": round(mean, 1),
        "p50": round(p50, 1),
        "p99": round(p99, 1),
        "verdict": "",
        "_dark": mean < 40 or p99 < 80,
    }


def _peer_verdict(lap_var: float, dark: bool, median: float) -> str:
    if dark and lap_var < 50:
        return "DARK+SOFT (underexposed; focus hard to judge)"
    if median <= 0:
        return "ok"
    if lap_var < 0.35 * median:
        return "SOFTEST (likely blur / motion / distance)"
    if lap_var < 0.7 * median:
        return "softer than median"
    if lap_var > 1.4 * median:
        return "sharpest of set"
    return "ok vs peers"


def _overall_verdict(score: float) -> str:
    # Absolute labels for the fixed manual focus setting (same for all poses).
    if score < 20:
        return "poor — twist the lens ring; mostly soft"
    if score < 40:
        return "soft — usable up close; ring may need a tweak"
    if score < 60:
        return "fair — ok for VL; not crisp"
    if score < 80:
        return "good — fixed focus is in a solid place"
    return "sharp — fixed focus looks well set"


def overall_focus(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """One score for the shared manual focus setting (ignores dark / unreadable)."""
    usable = [
        r for r in frames
        if r.get("verdict") != "unreadable" and r.get("mean", 0) >= 40
    ]
    excluded = len(frames) - len(usable)
    if not usable:
        return {
            "score": 0.0,
            "lap_var_median": 0.0,
            "usable_frames": 0,
            "excluded_frames": excluded,
            "verdict": "no usable frames (too dark / unreadable)",
        }
    laps = sorted(r["lap_var"] for r in usable)
    med = float(laps[len(laps) // 2])
    # Drop the softest quartile when scoring the lens (near-floor / motion outliers).
    if len(laps) >= 4:
        cut = max(1, len(laps) // 4)
        core = laps[cut:]
        core_med = float(core[len(core) // 2])
    else:
        core_med = med
    score = round(min(100.0, 100.0 * core_med / _SCORE_REF_LAP), 1)
    return {
        "score": score,
        "lap_var_median": round(med, 1),
        "lap_var_score_basis": round(core_med, 1),
        "usable_frames": len(usable),
        "excluded_frames": excluded,
        "verdict": _overall_verdict(score),
    }


def analyze_images(images: list[tuple[str, bytes]]) -> dict[str, Any]:
    """Return ``{"frames": [...], "overall": {...}}`` for a fixed-focus capture set."""
    rows = [analyze_jpeg(name, jpeg) for name, jpeg in images]
    peer_laps = [r["lap_var"] for r in rows if r.get("verdict") != "unreadable"]
    peer_med = float(sorted(peer_laps)[len(peer_laps) // 2]) if peer_laps else 0.0
    frames: list[dict[str, Any]] = []
    for r in rows:
        dark = bool(r.pop("_dark", False))
        if r.get("verdict") == "unreadable":
            frames.append(r)
            continue
        r["verdict"] = _peer_verdict(r["lap_var"], dark, peer_med)
        frames.append(r)
    return {"frames": frames, "overall": overall_focus(frames)}


def select_for_vl(
    images: list[tuple[str, bytes]],
    *,
    min_lap_var: float = 25.0,
    min_mean: float = 40.0,
    min_score: float = 30.0,
    max_images: int = 6,
) -> tuple[list[tuple[str, bytes]], list[str]]:
    """Keep the sharpest / best-lit frames for the VL call; drop junk.

    Returns ``(selected_images, dropped_pose_names)``. If everything fails the
    gate, falls back to the top ``max_images`` by score so VL still runs.
    """
    from bruhos.scan.camera_tune import score_frame

    scored: list[tuple[float, float, float, str, bytes]] = []
    for name, jpeg in images:
        sc, meta = score_frame(jpeg, name)
        lap = float(meta.get("lap_var") or 0.0)
        mean = float(meta.get("mean") or 0.0)
        scored.append((sc, lap, mean, name, jpeg))

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    kept: list[tuple[str, bytes]] = []
    dropped: list[str] = []
    for sc, lap, mean, name, jpeg in scored:
        ok = lap >= min_lap_var and mean >= min_mean and sc >= min_score
        if ok and len(kept) < max_images:
            kept.append((f"{name} (score={sc:.0f}, lap={lap:.0f})", jpeg))
        else:
            dropped.append(name)

    if not kept:
        for sc, lap, _mean, name, jpeg in scored[:max_images]:
            kept.append((f"{name} (score={sc:.0f}, lap={lap:.0f}, fallback)", jpeg))
        dropped = [n for _, _, _, n, _ in scored[max_images:]]
    return kept, dropped


def format_focus_table(report: dict[str, Any] | list[dict[str, Any]]) -> str:
    if isinstance(report, list):
        frames, overall = report, overall_focus(report)
    else:
        frames = report.get("frames") or []
        overall = report.get("overall") or overall_focus(frames)
    if not frames:
        return "(no frames)"
    header = (
        f"{'pose':6}  {'lapVar':>8}  {'lapROI':>8}  {'tenen':>10}  "
        f"{'mean':>5}  {'p50':>5}  {'p99':>5}  verdict"
    )
    lines = [header, "-" * len(header)]
    for r in frames:
        if r.get("verdict") == "unreadable":
            lines.append(
                f"{r['pose']:6}  {'—':>8}  {'—':>8}  {'—':>10}  "
                f"{'—':>5}  {'—':>5}  {'—':>5}  unreadable"
            )
            continue
        lines.append(
            f"{r['pose']:6}  {r['lap_var']:8.1f}  {r['lap_roi']:8.1f}  {r['tenengrad']:10.1f}  "
            f"{r['mean']:5.1f}  {r['p50']:5.1f}  {r['p99']:5.1f}  {r['verdict']}"
        )
    lines.append("")
    lines.append(
        f"OVERALL FOCUS  score={overall['score']:.1f}/100  "
        f"(lapVar≈{overall.get('lap_var_score_basis', overall['lap_var_median']):.1f}, "
        f"usable={overall['usable_frames']}, excluded={overall['excluded_frames']})"
    )
    lines.append(f"               {overall['verdict']}")
    return "\n".join(lines)
