"""MPU-6050 fall detection over I2C (smbus2). Tilt from the accelerometer, no fusion needed for falls."""

from __future__ import annotations

import math

from bruhos.schemas import Imu

PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
ACCEL_SCALE = 16384.0          # LSB/g at the default +-2g range


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
