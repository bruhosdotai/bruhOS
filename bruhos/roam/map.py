"""Dead-reckoned occupancy sketch for roam."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


FREE = 1
BLOCKED = 2
ASSUMED = 3


@dataclass
class Pose:
    x: float
    y: float
    heading_deg: float
    kind: str = "measured"   # measured | assumed | bump | fall


@dataclass
class OccupancyMap:
    size_m: float = 4.0
    cell_m: float = 0.05
    origin_x: float = 0.0
    origin_y: float = 0.0
    cells: dict[tuple[int, int], int] = field(default_factory=dict)
    trail: list[Pose] = field(default_factory=list)
    bumps: list[tuple[float, float]] = field(default_factory=list)
    aps: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.trail:
            self.trail.append(Pose(self.origin_x, self.origin_y, 0.0))

    @property
    def pose(self) -> Pose:
        return self.trail[-1]

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return (int(math.floor(x / self.cell_m)), int(math.floor(y / self.cell_m)))

    def mark(self, x: float, y: float, kind: int) -> None:
        c = self._cell(x, y)
        prev = self.cells.get(c)
        if prev == BLOCKED and kind != BLOCKED:
            return
        self.cells[c] = kind

    def mark_line(self, x0: float, y0: float, x1: float, y1: float, kind: int) -> None:
        dist = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(dist / (self.cell_m * 0.5))))
        for i in range(n + 1):
            t = i / n
            self.mark(x0 + t * (x1 - x0), y0 + t * (y1 - y0), kind)

    def advance(self, forward_m: float, lateral_m: float = 0.0, yaw_deg: float = 0.0,
                kind: str = "measured") -> Pose:
        p = self.pose
        h = math.radians(p.heading_deg)
        # +x forward at heading 0, +y left
        dx = forward_m * math.cos(h) - lateral_m * math.sin(h)
        dy = forward_m * math.sin(h) + lateral_m * math.cos(h)
        x1, y1 = p.x + dx, p.y + dy
        cell_kind = ASSUMED if kind == "assumed" else FREE
        self.mark_line(p.x, p.y, x1, y1, cell_kind)
        new_h = (p.heading_deg + yaw_deg) % 360.0
        pose = Pose(x1, y1, new_h, kind)
        self.trail.append(pose)
        return pose

    def mark_bump_ahead(self, step_m: float | None = None) -> tuple[float, float]:
        p = self.pose
        d = step_m if step_m is not None else self.cell_m
        h = math.radians(p.heading_deg)
        bx = p.x + d * math.cos(h)
        by = p.y + d * math.sin(h)
        self.mark(bx, by, BLOCKED)
        self.bumps.append((bx, by))
        return bx, by

    def turn(self, yaw_deg: float) -> Pose:
        p = self.pose
        pose = Pose(p.x, p.y, (p.heading_deg + yaw_deg) % 360.0, "turn")
        self.trail.append(pose)
        return pose

    def apply_pose_updates(self, xs: list[float], ys: list[float], headings: list[float]) -> None:
        """Replace trail poses (same length) after a WiFi fit; rebuild free cells from trail."""
        if len(xs) != len(self.trail):
            return
        new_trail = [
            Pose(xs[i], ys[i], headings[i] % 360.0, self.trail[i].kind)
            for i in range(len(self.trail))
        ]
        self._rebuild_from_trail(new_trail, bump_deltas=None)

    def apply_keyframe_warp(
        self,
        old_xy: list[tuple[float, float]],
        new_xy: list[tuple[float, float]],
    ) -> bool:
        """Warp the dense IMU trail by piecewise deltas from WiFi keyframe poses.

        Each trail pose / bump gets the delta of the nearest old keyframe (by XY).
        Relative step shape between keyframes is only approximately preserved.
        """
        if len(old_xy) != len(new_xy) or len(old_xy) < 2:
            return False
        deltas = [(nx - ox, ny - oy) for (ox, oy), (nx, ny) in zip(old_xy, new_xy)]

        def nearest_delta(x: float, y: float) -> tuple[float, float]:
            best_i, best_d = 0, float("inf")
            for i, (ox, oy) in enumerate(old_xy):
                d = (x - ox) * (x - ox) + (y - oy) * (y - oy)
                if d < best_d:
                    best_d, best_i = d, i
            return deltas[best_i]

        new_trail = []
        for p in self.trail:
            dx, dy = nearest_delta(p.x, p.y)
            new_trail.append(Pose(p.x + dx, p.y + dy, p.heading_deg, p.kind))
        bump_xy = []
        for bx, by in self.bumps:
            dx, dy = nearest_delta(bx, by)
            bump_xy.append((bx + dx, by + dy))
        self._rebuild_from_trail(new_trail, bump_xy)
        return True

    def _rebuild_from_trail(
        self,
        new_trail: list[Pose],
        bump_deltas: list[tuple[float, float]] | None,
    ) -> None:
        aps = list(self.aps)
        bumps = list(bump_deltas) if bump_deltas is not None else list(self.bumps)
        self.trail = new_trail
        self.cells.clear()
        self.bumps = []
        for i in range(1, len(self.trail)):
            a, b = self.trail[i - 1], self.trail[i]
            kind = ASSUMED if b.kind == "assumed" else FREE
            if b.kind in ("bump", "fall", "turn"):
                continue
            self.mark_line(a.x, a.y, b.x, b.y, kind)
        for bx, by in bumps:
            self.mark(bx, by, BLOCKED)
            self.bumps.append((bx, by))
        self.aps = aps

    def ascii(self, width: int = 41, height: int = 21) -> str:
        if not self.trail and not self.cells:
            return "(empty)"
        half_w = width // 2
        half_h = height // 2
        ox, oy = self._cell(self.pose.x, self.pose.y)
        chars = []
        ap_cells = {self._cell(a["x"], a["y"]) for a in self.aps if "x" in a}
        for row in range(half_h, -half_h - 1, -1):
            line = []
            for col in range(-half_w, half_w + 1):
                c = (ox + col, oy + row)
                if c == (ox, oy):
                    line.append("o")
                elif c in ap_cells:
                    line.append("A")
                else:
                    v = self.cells.get(c)
                    if v == BLOCKED:
                        line.append("#")
                    elif v == FREE:
                        line.append(".")
                    elif v == ASSUMED:
                        line.append(",")
                    else:
                        line.append(" ")
            chars.append("".join(line))
        return "\n".join(chars)

    def to_dict(self) -> dict[str, Any]:
        return {
            "size_m": self.size_m,
            "cell_m": self.cell_m,
            "trail": [{"x": p.x, "y": p.y, "heading_deg": p.heading_deg, "kind": p.kind} for p in self.trail],
            "bumps": [{"x": x, "y": y} for x, y in self.bumps],
            "cells": [{"ix": k[0], "iy": k[1], "v": v} for k, v in sorted(self.cells.items())],
            "aps": list(self.aps),
            "pose": {"x": self.pose.x, "y": self.pose.y, "heading_deg": self.pose.heading_deg},
        }
