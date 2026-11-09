"""Enumerate every joystick the kernel exposes.

The MAGIC-S Ultimate shows up as its own js node (a DualShock 4), next to any
other pad. Stock TonyPi code only opens js0, so a second controller is invisible.
"""

from __future__ import annotations

import array
import fcntl
import os
import struct
from dataclasses import dataclass
from pathlib import Path

# linux/joystick.h — _IOR('j', nr, uint8) / _IOC_READ name
JSIOCGAXES = (2 << 30) | (1 << 16) | (ord("j") << 8) | 0x11
JSIOCGBUTTONS = (2 << 30) | (1 << 16) | (ord("j") << 8) | 0x12

JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
JS_EVENT_SIZE = 8  # uint32 time, int16 value, uint8 type, uint8 number


def _jsiocgname(length: int) -> int:
    return (2 << 30) | (length << 16) | (ord("j") << 8) | 0x13


@dataclass(frozen=True)
class JoystickDevice:
    index: int
    path: str
    name: str
    axes: int
    buttons: int
    by_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "path": self.path,
            "name": self.name,
            "axes": self.axes,
            "buttons": self.buttons,
            "by_id": self.by_id,
        }


def query_joystick(path: str) -> tuple[str, int, int]:
    """Return (name, axis count, button count) for one js device."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        axes = array.array("B", [0])
        buttons = array.array("B", [0])
        name = array.array("B", bytes(128))
        fcntl.ioctl(fd, JSIOCGAXES, axes, True)
        fcntl.ioctl(fd, JSIOCGBUTTONS, buttons, True)
        fcntl.ioctl(fd, _jsiocgname(len(name)), name, True)
        label = bytes(name).split(b"\x00", 1)[0].decode("utf-8", "replace").strip()
        return label or Path(path).name, int(axes[0]), int(buttons[0])
    finally:
        os.close(fd)


def _by_id_map(dev_dir: Path) -> dict[str, str]:
    """Map resolved js path → by-id symlink name."""
    found: dict[str, str] = {}
    by_id = dev_dir / "by-id"
    if not by_id.is_dir():
        return found
    for link in sorted(by_id.iterdir()):
        if not link.is_symlink():
            continue
        target = (link.parent / os.readlink(link)).resolve()
        if target.name.startswith("js"):
            found[str(target)] = link.name
    return found


def scan_joysticks(dev_dir: str | Path = "/dev/input", query=query_joystick) -> list[JoystickDevice]:
    """Every js* node, including pads behind a MAGIC-S and any other controller."""
    root = Path(dev_dir)
    ids = _by_id_map(root)
    devices: list[JoystickDevice] = []
    for path in sorted(root.glob("js*"), key=lambda p: p.name):
        if not path.name[2:].isdigit():
            continue
        try:
            name, axes, buttons = query(str(path))
        except OSError:
            continue
        devices.append(
            JoystickDevice(
                index=int(path.name[2:]),
                path=str(path),
                name=name,
                axes=axes,
                buttons=buttons,
                by_id=ids.get(str(path.resolve()), ids.get(str(path))),
            )
        )
    return devices


@dataclass
class JsEvent:
    time_ms: int
    kind: str          # axis | button
    number: int
    value: int         # axis: -32767..32767, button: 0 or 1
    init: bool


class JoystickReader:
    """Non-blocking reader for one js device. Safe to open every scanned node."""

    def __init__(self, device: JoystickDevice):
        self.device = device
        self.fd = os.open(device.path, os.O_RDONLY | os.O_NONBLOCK)
        self.axes = [0] * device.axes
        self.buttons = [0] * device.buttons

    def poll(self) -> list[JsEvent]:
        events: list[JsEvent] = []
        while True:
            try:
                data = os.read(self.fd, JS_EVENT_SIZE)
            except BlockingIOError:
                break
            if len(data) < JS_EVENT_SIZE:
                break
            time_ms, value, typ, number = struct.unpack("IhBB", data)
            init = bool(typ & JS_EVENT_INIT)
            kind_n = typ & ~JS_EVENT_INIT
            if kind_n == JS_EVENT_AXIS:
                kind = "axis"
                if 0 <= number < len(self.axes):
                    self.axes[number] = value
            elif kind_n == JS_EVENT_BUTTON:
                kind = "button"
                if 0 <= number < len(self.buttons):
                    self.buttons[number] = value
            else:
                continue
            events.append(JsEvent(time_ms, kind, number, value, init))
        return events

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:
            pass
