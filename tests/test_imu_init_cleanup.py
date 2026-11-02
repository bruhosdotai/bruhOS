import sys
import types

import pytest


class FakeBus:
    instances: list["FakeBus"] = []

    def __init__(self, bus):
        self.closed = False
        FakeBus.instances.append(self)

    def write_byte_data(self, addr, reg, value):
        raise OSError(121, "Remote I/O error")  # no device answers at this address

    def close(self):
        self.closed = True


@pytest.fixture
def fake_smbus2(monkeypatch):
    FakeBus.instances.clear()
    monkeypatch.setitem(sys.modules, "smbus2", types.SimpleNamespace(SMBus=FakeBus))
    return FakeBus


def test_failed_wake_up_closes_the_i2c_bus(fake_smbus2):
    from bruhos.body.imu import MPU6050

    with pytest.raises(OSError):
        MPU6050(bus=1, address=0x69)
    assert len(fake_smbus2.instances) == 1
    assert fake_smbus2.instances[0].closed
