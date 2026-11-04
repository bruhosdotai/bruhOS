"""IMU readers: smbus MPU-6050 (legacy) and Hiwonder controller-board stream (TonyPi roam).

Hiwonder Fall_and_Stand uses ``atan2(ax, az)``: near 0° face-down, near 180° on back,
standing around ±90° — chip X is up. Yaw while standing is rotation about X (gx).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

from bruhos.schemas import Imu

PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
ACCEL_SCALE = 16384.0          # LSB/g at the default +-2g range


@dataclass(frozen=True)
class ImuSample:
    """One host-timestamped accel+gyro packet. Accel in sensor units; normalize later."""

    t: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float

    @classmethod
    def from_board(cls, packet: Sequence[float], t: float | None = None) -> "ImuSample":
        if len(packet) < 6:
            raise ValueError(f"IMU packet needs 6 floats, got {len(packet)}")
        return cls(
            t if t is not None else time.monotonic(),
            float(packet[0]), float(packet[1]), float(packet[2]),
            float(packet[3]), float(packet[4]), float(packet[5]),
        )


def hiwonder_lean_deg(ax: float, az: float) -> float:
    """Hiwonder Fall_and_Stand angle: ~0 face-down, ~±180 on back, ~±90 standing."""
    return math.degrees(math.atan2(ax, az))


def sample_to_imu(
    s: ImuSample,
    *,
    fall_forward_deg: float = 15.0,
    fall_back_deg: float = 160.0,
) -> Imu:
    """Map a board sample to the compressed Imu used by skills (Hiwonder axes)."""
    lean = hiwonder_lean_deg(s.ax, s.az)
    face_down = abs(lean) < fall_forward_deg
    on_back = abs(lean) > fall_back_deg
    fallen = face_down or on_back
    return Imu(
        roll_deg=round(math.degrees(math.atan2(s.ay, s.ax)), 2),
        pitch_deg=round(lean, 2),
        fallen=fallen,
        fell_forward=True if face_down else (False if on_back else None),
    )


class BoardImu:
    """Drain ``board.get_imu()`` from the ActionGroupControl serial board."""

    def __init__(
        self,
        board: Any,
        *,
        fall_forward_deg: float = 15.0,
        fall_back_deg: float = 160.0,
        forward_sign: int = 1,
    ):
        self.board = board
        self.fall_forward_deg = fall_forward_deg
        self.fall_back_deg = fall_back_deg
        self.forward_sign = forward_sign
        self._enabled = False

    def enable(self) -> None:
        en = getattr(self.board, "enable_reception", None)
        if en:
            en(True)
        self._enabled = True

    def poll(self) -> ImuSample | None:
        get = getattr(self.board, "get_imu", None)
        if not get:
            return None
        data = get()
        if data is None:
            return None
        return ImuSample.from_board(data)

    def read(self) -> Imu:
        s = self.poll()
        if s is None:
            return Imu()
        return sample_to_imu(s, fall_forward_deg=self.fall_forward_deg, fall_back_deg=self.fall_back_deg)

    def close(self) -> None:
        pass


class MPU6050:
    def __init__(self, bus: int = 1, address: int = 0x68, fall_deg: float = 55.0, forward_sign: int = 1):
        from smbus2 import SMBus  # optional extra: pip install bruhos[imu]

        self.bus = SMBus(bus)
        self.addr = address
        self.fall_deg = fall_deg
        self.forward_sign = forward_sign      # flip if the board is mounted reversed
        self.bus.write_byte_data(self.addr, PWR_MGMT_1, 0)

    def _word(self, reg: int) -> int:
        hi, lo = self.bus.read_i2c_block_data(self.addr, reg, 2)
        v = (hi << 8) | lo
        return v - 65536 if v > 32767 else v

    def read(self) -> Imu:
        ax, ay, az = (self._word(ACCEL_XOUT_H + 2 * i) / ACCEL_SCALE for i in range(3))
        roll = math.degrees(math.atan2(ay, az))
        pitch = math.degrees(math.atan2(-ax, math.hypot(ay, az))) * self.forward_sign
        fallen = abs(pitch) > self.fall_deg or abs(roll) > self.fall_deg
        return Imu(round(roll, 2), round(pitch, 2), fallen, (pitch > 0) if fallen else None)

    def close(self) -> None:
        self.bus.close()
