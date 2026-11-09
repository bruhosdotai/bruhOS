import os
from pathlib import Path

from bruhos.input.drive import stick_command
from bruhos.input.joysticks import scan_joysticks


def test_stick_deadzone_and_signs():
    assert stick_command(0.0, 0.0) is None
    assert stick_command(0.2, -0.2) is None
    assert stick_command(0.0, -0.8) == "forward"
    assert stick_command(0.0, 0.8) == "back"
    assert stick_command(-0.9, 0.1) == "turn_left"
    assert stick_command(0.9, 0.1) == "turn_right"
    # forward wins when both are pushed and Y is stronger
    assert stick_command(0.4, -0.9) == "forward"


def test_scan_all_js_nodes(tmp_path: Path):
    dev = tmp_path / "input"
    by_id = dev / "by-id"
    by_id.mkdir(parents=True)
    js0 = dev / "js0"
    js1 = dev / "js1"
    js0.write_bytes(b"")
    js1.write_bytes(b"")
    (dev / "js-not-a-device").write_bytes(b"")
    os.symlink("../js1", by_id / "usb-MAGIC-S-joystick")

    def query(path: str):
        name = "ShanWan" if path.endswith("js0") else "MAGIC-S Ultimate"
        return name, 8, 14

    found = scan_joysticks(dev, query=query)
    assert [d.index for d in found] == [0, 1]
    assert found[0].name == "ShanWan"
    assert found[0].by_id is None
    assert found[1].name == "MAGIC-S Ultimate"
    assert found[1].by_id == "usb-MAGIC-S-joystick"
    assert found[1].axes == 8 and found[1].buttons == 14
