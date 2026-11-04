"""Shared head pose control for the web UI (grid + nudge)."""

from __future__ import annotations

import threading
from typing import Any, Callable

from bruhos.body.head import Head, HeadPose

MakeHead = Callable[[dict[str, Any]], Head]

DEFAULT_NUDGE_US = 80


class HeadControl:
    """Lazy Head wrapper with current pose tracking and a display grid."""

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        head: Head | None = None,
        make_head: MakeHead | None = None,
        nudge_us: int = DEFAULT_NUDGE_US,
    ):
        self.cfg = cfg
        self._head = head
        self._make_head = make_head or (lambda c: Head.from_config(c))
        self.nudge_us = abs(int(nudge_us))
        self._lock = threading.RLock()
        self._yaw: int | None = None
        self._pitch: int | None = None
        self._pose_name: str | None = None
        self._error: str | None = None

    def _ensure(self) -> Head:
        if self._head is None:
            self._head = self._make_head(self.cfg)
            if self._yaw is None:
                self._yaw = self._head.yaw_center
                self._pitch = self._head.pitch_center
                self._pose_name = self._nearest_name(self._yaw, self._pitch)
        return self._head

    def poses(self) -> list[HeadPose]:
        return self._ensure().poses()

    def _nearest_name(self, yaw: int, pitch: int) -> str | None:
        best = None
        best_d = 10**9
        for p in self._ensure().poses():
            d = abs(p.yaw_us - yaw) + abs(p.pitch_us - pitch)
            if d < best_d:
                best_d = d
                best = p.name
        return best if best_d <= 60 else None

    def grid(self) -> list[list[dict[str, Any]]]:
        """3×3 (or fewer) looking out through the camera: top = look-up."""
        poses = self.poses()
        by_name = {p.name: p for p in poses}
        rows = sorted({int(p.name[1:]) for p in poses if len(p.name) >= 2 and p.name[1:].isdigit()})
        display_rows = list(reversed(rows))
        cols = ["L", "C", "R"]
        out: list[list[dict[str, Any]]] = []
        for r in display_rows:
            line = []
            for c in cols:
                name = f"{c}{r}"
                p = by_name.get(name)
                if p is None:
                    line.append({"name": name, "available": False})
                else:
                    line.append({
                        "name": p.name,
                        "available": True,
                        "yaw_us": p.yaw_us,
                        "pitch_us": p.pitch_us,
                    })
            out.append(line)
        return out

    def status(self) -> dict[str, Any]:
        with self._lock:
            try:
                h = self._ensure()
            except Exception as e:  # noqa: BLE001
                self._error = str(e)[:300]
                return {
                    "ok": False,
                    "error": self._error,
                    "poses": [],
                    "grid": [],
                    "current": None,
                    "nudge_us": self.nudge_us,
                }
            if self._yaw is None:
                self._yaw = h.yaw_center
                self._pitch = h.pitch_center
                self._pose_name = self._nearest_name(self._yaw, self._pitch)
            return {
                "ok": True,
                "error": self._error,
                "nudge_us": self.nudge_us,
                "move_ms": h.move_ms,
                "limits": {
                    "yaw_min": h.yaw_min_us,
                    "yaw_max": h.yaw_max_us,
                    "pitch_min": h.pitch_min_us,
                    "pitch_max": h.pitch_max_us,
                    "yaw_center": h.yaw_center,
                    "pitch_center": h.pitch_center,
                },
                "poses": [
                    {"name": p.name, "yaw_us": p.yaw_us, "pitch_us": p.pitch_us}
                    for p in h.poses()
                ],
                "grid": self.grid(),
                "current": {
                    "name": self._pose_name,
                    "yaw_us": self._yaw,
                    "pitch_us": self._pitch,
                },
            }

    def go_pose(self, name: str) -> dict[str, Any]:
        with self._lock:
            h = self._ensure()
            match = next((p for p in h.poses() if p.name == name), None)
            if match is None:
                raise ValueError(f"unknown pose {name!r}")
            h.go_pose(match)
            self._yaw, self._pitch = match.yaw_us, match.pitch_us
            self._pose_name = match.name
            self._error = None
            return self.status()

    def center(self) -> dict[str, Any]:
        with self._lock:
            h = self._ensure()
            h.recenter()
            self._yaw, self._pitch = h.yaw_center, h.pitch_center
            self._pose_name = self._nearest_name(self._yaw, self._pitch)
            self._error = None
            return self.status()

    def nudge(self, direction: str) -> dict[str, Any]:
        d = direction.casefold().strip()
        with self._lock:
            h = self._ensure()
            if self._yaw is None or self._pitch is None:
                self._yaw, self._pitch = h.yaw_center, h.pitch_center
            yaw, pitch = self._yaw, self._pitch
            step = self.nudge_us
            # Match [head].yaw_sign so left/right are camera-POV (same as L*/R* poses).
            if d == "left":
                yaw -= h.yaw_sign * step
            elif d == "right":
                yaw += h.yaw_sign * step
            elif d == "up":
                pitch += step
            elif d == "down":
                pitch -= step
            else:
                raise ValueError("nudge must be left|right|up|down")
            h.go_pulse(yaw, pitch)
            self._yaw = max(h.yaw_min_us, min(h.yaw_max_us, yaw))
            self._pitch = max(h.pitch_min_us, min(h.pitch_max_us, pitch))
            self._pose_name = self._nearest_name(self._yaw, self._pitch)
            self._error = None
            return self.status()

    def apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("center"):
            return self.center()
        if payload.get("pose"):
            return self.go_pose(str(payload["pose"]))
        if payload.get("nudge"):
            return self.nudge(str(payload["nudge"]))
        raise ValueError("need pose, nudge, or center")
