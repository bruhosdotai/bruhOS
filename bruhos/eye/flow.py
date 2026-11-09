"""Which way the picture is moving, from two gray frames.

Positive ``flow_x`` means image content moved right. Positive ``flow_y``
means it moved down. Expansion is ``closer``; contraction is ``farther``.
No model and no extra camera open.
"""

from __future__ import annotations

from typing import Any

JUMP = 18
MIN_CELL = 0.08


def motion_flow(
    prev: bytes,
    curr: bytes,
    *,
    width: int,
    height: int,
    jump: int = JUMP,
) -> dict[str, Any]:
    still = {"direction": "still", "flow_x": 0.0, "flow_y": 0.0, "movers": []}
    if width < 8 or height < 6 or len(prev) != width * height or len(curr) != len(prev):
        return still
    cols, rows = 4, 3
    cw, ch = width // cols, height // rows
    if cw < 4 or ch < 4:
        return still
    radius = max(2, min(4, cw // 3))
    samples: list[tuple[float, float, float, float, float, float]] = []
    for gy in range(rows):
        for gx in range(cols):
            x0, y0 = gx * cw, gy * ch
            if _changed(prev, curr, width, x0, y0, cw, ch, jump) < MIN_CELL:
                continue
            shift = _shift(prev, curr, width, height, x0, y0, cw, ch, radius)
            if shift is None:
                continue
            vx, vy = shift
            cx = (x0 + cw / 2) - width / 2
            cy = (y0 + ch / 2) - height / 2
            samples.append((vx, vy, cx, cy, x0 + cw / 2, y0 + ch / 2))
    if not samples:
        return still
    n = len(samples)
    fx = sum(s[0] for s in samples) / n
    fy = sum(s[1] for s in samples) / n
    div = sum(s[0] * s[2] + s[1] * s[3] for s in samples) / n
    span = max(width, height) / 2
    movers = []
    ranked = sorted(samples, key=lambda s: s[0] * s[0] + s[1] * s[1], reverse=True)
    for vx, vy, _cx, _cy, px, py in ranked[:3]:
        if abs(vx) < 0.4 and abs(vy) < 0.4:
            continue
        movers.append({
            "x": round(px / width, 3),
            "y": round(py / height, 3),
            "dx": round(vx, 2),
            "dy": round(vy, 2),
        })
    return {
        "direction": _label(fx, fy, div, span),
        "flow_x": round(fx, 2),
        "flow_y": round(fy, 2),
        "movers": movers,
    }


def _changed(prev: bytes, curr: bytes, width: int, x0: int, y0: int, cw: int, ch: int, jump: int) -> float:
    changed = 0
    total = cw * ch
    for y in range(y0, y0 + ch):
        row = y * width
        for x in range(x0, x0 + cw):
            if abs(prev[row + x] - curr[row + x]) > jump:
                changed += 1
    return changed / total


def _sad(
    prev: bytes,
    curr: bytes,
    width: int,
    height: int,
    x0: int,
    y0: int,
    cw: int,
    ch: int,
    dx: int,
    dy: int,
) -> float:
    total = 0
    count = 0
    for y in range(y0, y0 + ch, 2):
        sy = y - dy
        if sy < 0 or sy >= height:
            continue
        row = y * width
        prow = sy * width
        for x in range(x0, x0 + cw, 2):
            sx = x - dx
            if sx < 0 or sx >= width:
                continue
            total += abs(curr[row + x] - prev[prow + sx])
            count += 1
    if count < 4:
        return 1e9
    return total / count


def _shift(
    prev: bytes,
    curr: bytes,
    width: int,
    height: int,
    x0: int,
    y0: int,
    cw: int,
    ch: int,
    radius: int,
) -> tuple[float, float] | None:
    base = _sad(prev, curr, width, height, x0, y0, cw, ch, 0, 0)
    best_dx, best_dy, best = 0, 0, base
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            score = _sad(prev, curr, width, height, x0, y0, cw, ch, dx, dy)
            if score < best:
                best_dx, best_dy, best = dx, dy, score
    if best > base * 0.92:
        return None
    return float(best_dx), float(best_dy)


def _label(fx: float, fy: float, div: float, span: float) -> str:
    trans = max(abs(fx), abs(fy))
    expanding = div / max(span, 1.0)
    if trans < 0.45 and abs(expanding) < 0.35:
        return "still"
    if abs(expanding) > 0.55 and abs(expanding) > trans * 0.8:
        return "closer" if expanding > 0 else "farther"
    parts: list[str] = []
    if abs(fx) >= 0.45:
        parts.append("right" if fx > 0 else "left")
    if abs(fy) >= 0.45:
        parts.append("down" if fy > 0 else "up")
    return " ".join(parts) if parts else "still"
