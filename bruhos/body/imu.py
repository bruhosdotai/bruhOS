"""MPU-6050 fall detection over I2C (smbus2). Tilt from the accelerometer, no fusion needed for falls."""

from __future__ import annotations

import math
import time

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


class ControllerIMU:
    """Adapt Hiwonder's existing board IMU stream to the same read/close interface.

    The action SDK owns the board connection; this reader does not open or close it.
    Axes and zero offsets describe the sensor mounting on the robot.
    """

    def __init__(self, board, fall_deg=55.0, forward_sign=1, upright_axis=0,
                 forward_axis=2, upright_sign=1, pitch_zero_deg=0.0, roll_zero_deg=0.0):
        self.board = board
        self.fall_deg = fall_deg
        self.upright_axis, self.forward_axis = upright_axis, forward_axis
        self.side_axis, = {0, 1, 2} - {upright_axis, forward_axis}
        self.upright_sign, self.forward_sign = upright_sign, forward_sign
        self.pitch_zero_deg, self.roll_zero_deg = pitch_zero_deg, roll_zero_deg
        board.enable_reception()

    def read(self) -> Imu:
        deadline = time.monotonic() + 0.2
        sample = self.board.get_imu()
        while sample is None and time.monotonic() < deadline:
            time.sleep(0.01)
            sample = self.board.get_imu()
        if sample is None:
            raise OSError("No IMU sample received from Hiwonder controller")
        if len(sample) != 6 or not all(math.isfinite(v) for v in sample) or not any(sample[:3]):
            raise ValueError("Invalid controller IMU sample")
        up = sample[self.upright_axis] * self.upright_sign
        forward = sample[self.forward_axis] * self.forward_sign
        side = sample[self.side_axis]
        pitch = (math.degrees(math.atan2(forward, up)) - self.pitch_zero_deg + 180) % 360 - 180
        roll = math.degrees(math.atan2(side, math.hypot(up, forward))) - self.roll_zero_deg
        fallen = abs(pitch) > self.fall_deg or abs(roll) > self.fall_deg
        direction = (pitch > 0) if fallen and abs(pitch) >= abs(roll) else None
        return Imu(round(roll, 2), round(pitch, 2), fallen, direction)

    def close(self) -> None:
        pass  # The action SDK retains ownership of its shared board connection.
