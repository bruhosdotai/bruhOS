"""TonyPi head (PWM servos 1=pitch, 2=yaw) — pose grid inside Hiwonder safe stops."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


PITCH_SERVO = 1
YAW_SERVO = 2
DEFAULT_YAW_OFFSET_US = 400
# servo1 in servo_config is table-down (~914). KickBall only searches up to 1200 (still
# floor-ish). Head_Control nod uses ~1800 for the high end — that is real look-up.
DEFAULT_PITCH_MAX_US = 1800
PITCH_RANGE_EPS_US = 50          # collapse to yaw-only when travel is this small
PITCH_THREE_ROW_US = 500         # span large enough for down / mid / up rows
DEFAULT_MOVE_MS = 500            # servo trajectory duration; scan waits this + settle


class HeadController(Protocol):
    def set_pwm_servo_pulse(self, servo_id: int, pulse: int, use_time: int) -> None: ...


@dataclass(frozen=True)
class HeadPose:
    name: str
    yaw_us: int
    pitch_us: int


def load_servo_centers(path: str | Path) -> tuple[int, int]:
    """Parse Hiwonder ``servo_config.yaml`` (``servo1: N`` / ``servo2: N``) without PyYAML."""
    pitch = yaw = None
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if key == "servo1":
            pitch = int(float(val))
        elif key == "servo2":
            yaw = int(float(val))
    if pitch is None or yaw is None:
        raise ValueError(f"servo1/servo2 not found in {path}")
    return pitch, yaw


def scan_poses(
    pitch_center: int,
    yaw_center: int,
    *,
    yaw_offset_us: int = DEFAULT_YAW_OFFSET_US,
    pitch_min_us: int | None = None,
    pitch_max_us: int = DEFAULT_PITCH_MAX_US,
    yaw_sign: int = 1,
) -> list[HeadPose]:
    """Yaw × pitch grid in snake order (row0 = down / low pulse → last row = look-up).

    ``servo1`` is calibrated table-down; higher pitch pulse looks up (toward ~1800).
    """
    p_min = pitch_center if pitch_min_us is None else int(pitch_min_us)
    p_max = int(pitch_max_us)
    if p_max < p_min:
        p_min, p_max = p_max, p_min
    off = abs(int(yaw_offset_us))
    # yaw_sign: +1 = lower pulse is camera-left; -1 = mirrored servo (TonyPi here).
    sign = 1 if yaw_sign >= 0 else -1
    left = yaw_center - sign * off
    right = yaw_center + sign * off
    yaw_cols = [("L", left), ("C", yaw_center), ("R", right)]

    span = p_max - p_min
    if span <= PITCH_RANGE_EPS_US:
        pitch_rows = [p_min]
    elif span < PITCH_THREE_ROW_US:
        pitch_rows = [p_min, p_max]
    else:
        pitch_rows = [p_min, (p_min + p_max) // 2, p_max]

    out: list[HeadPose] = []
    for row_i, pitch in enumerate(pitch_rows):
        cols = yaw_cols if row_i % 2 == 0 else list(reversed(yaw_cols))
        for lab, yaw in cols:
            out.append(HeadPose(f"{lab}{row_i}", yaw, pitch))
    return out


def clamp_pulse(pulse: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(pulse)))


class Head:
    """PWM head with calibrated centers and a FOV-covering scan grid."""

    def __init__(
        self,
        ctl: HeadController,
        *,
        pitch_center: int,
        yaw_center: int,
        yaw_offset_us: int = DEFAULT_YAW_OFFSET_US,
        pitch_min_us: int | None = None,
        pitch_max_us: int = DEFAULT_PITCH_MAX_US,
        yaw_sign: int = 1,
        move_ms: int = DEFAULT_MOVE_MS,
        board: Any = None,
    ):
        self.ctl = ctl
        self.board = board
        self.pitch_center = int(pitch_center)
        self.yaw_center = int(yaw_center)
        self.yaw_offset_us = abs(int(yaw_offset_us))
        self.pitch_min_us = self.pitch_center if pitch_min_us is None else int(pitch_min_us)
        self.pitch_max_us = int(pitch_max_us)
        if self.pitch_max_us < self.pitch_min_us:
            self.pitch_min_us, self.pitch_max_us = self.pitch_max_us, self.pitch_min_us
        self.yaw_sign = 1 if yaw_sign >= 0 else -1
        self.move_ms = int(move_ms)
        self.yaw_min_us = self.yaw_center - self.yaw_offset_us
        self.yaw_max_us = self.yaw_center + self.yaw_offset_us

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctl: HeadController | None = None) -> Head:
        head = cfg.get("head", {})
        pitch = head.get("servo1")
        yaw = head.get("servo2")
        if pitch is None or yaw is None:
            path = head.get("servo_config", "/home/pi/TonyPi/servo_config.yaml")
            pitch, yaw = load_servo_centers(path)
        if ctl is None:
            ctl, board = _open_hiwonder(head.get("sdk_path", "/home/pi/TonyPi"))
        else:
            board = None
        return cls(
            ctl,
            pitch_center=int(pitch),
            yaw_center=int(yaw),
            yaw_offset_us=int(head.get("yaw_offset_us", DEFAULT_YAW_OFFSET_US)),
            pitch_min_us=head.get("pitch_min_us"),
            pitch_max_us=int(head.get("pitch_max_us", DEFAULT_PITCH_MAX_US)),
            yaw_sign=int(head.get("yaw_sign", 1)),
            move_ms=int(head.get("move_ms", DEFAULT_MOVE_MS)),
            board=board,
        )

    def poses(self) -> list[HeadPose]:
        return scan_poses(
            self.pitch_center,
            self.yaw_center,
            yaw_offset_us=self.yaw_offset_us,
            pitch_min_us=self.pitch_min_us,
            pitch_max_us=self.pitch_max_us,
            yaw_sign=self.yaw_sign,
        )

    def go_pulse(self, yaw_us: int, pitch_us: int, move_ms: int | None = None) -> None:
        ms = self.move_ms if move_ms is None else int(move_ms)
        yaw = clamp_pulse(yaw_us, self.yaw_min_us, self.yaw_max_us)
        pitch = clamp_pulse(pitch_us, self.pitch_min_us, self.pitch_max_us)
        self.ctl.set_pwm_servo_pulse(PITCH_SERVO, pitch, ms)
        self.ctl.set_pwm_servo_pulse(YAW_SERVO, yaw, ms)

    def go_pose(self, pose: HeadPose, move_ms: int | None = None) -> None:
        self.go_pulse(pose.yaw_us, pose.pitch_us, move_ms=move_ms)

    def recenter(self, move_ms: int | None = None) -> None:
        self.go_pulse(self.yaw_center, self.pitch_center, move_ms=move_ms)

    def close(self) -> None:
        close = getattr(self.board, "close", None) or getattr(self.board, "disconnect", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def _open_hiwonder(sdk_path: str | None) -> tuple[HeadController, Any]:
    if sdk_path and sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)
    hi_sdk = str(Path(sdk_path) / "HiwonderSDK") if sdk_path else None
    if hi_sdk and hi_sdk not in sys.path:
        sys.path.insert(0, hi_sdk)
    try:
        import hiwonder.ros_robot_controller_sdk as rrc
        from hiwonder.Controller import Controller
    except ImportError as e:
        raise ImportError(
            "Hiwonder Controller not found; set [head].sdk_path (e.g. /home/pi/TonyPi). "
            "Stop other TonyPi apps if /dev/ttyAMA0 is busy."
        ) from e
    try:
        board = rrc.Board()
    except Exception as e:
        raise RuntimeError(
            f"could not open robot board ({e}); is /dev/ttyAMA0 free? "
            "Stop TonyPi RPC/Joystick and retry."
        ) from e
    return Controller(board), board
