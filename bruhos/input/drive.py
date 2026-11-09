"""Hold a stick on any connected joystick to step the body.

Left stick is axis 0 (strafe / turn) and axis 1 (forward / back). That layout
is what both the ShanWan pad and a MAGIC-S (DualShock 4 emulation) expose.
Several pads can be plugged in; the strongest deflection wins.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from bruhos.body.base import Body
from bruhos.input.joysticks import JoystickReader, scan_joysticks

DEADZONE = 0.50
AXIS_FULL = 32767.0


@dataclass
class DriveConfig:
    deadzone: float = DEADZONE
    rescan_s: float = 1.0


def stick_command(lx: float, ly: float, deadzone: float = DEADZONE) -> str | None:
    """Map one left stick to a body primitive. ly < 0 is forward (Linux gamepads)."""
    ax, ay = abs(lx), abs(ly)
    if ay >= ax and ay >= deadzone:
        return "forward" if ly < 0 else "back"
    if ax >= deadzone:
        return "turn_left" if lx < 0 else "turn_right"
    return None


def _strongest(readers: list[JoystickReader], deadzone: float) -> tuple[str | None, str | None]:
    best: tuple[float, str, str] | None = None
    for reader in readers:
        if len(reader.axes) < 2:
            continue
        lx = reader.axes[0] / AXIS_FULL
        ly = reader.axes[1] / AXIS_FULL
        cmd = stick_command(lx, ly, deadzone)
        if cmd is None:
            continue
        mag = max(abs(lx), abs(ly))
        if best is None or mag > best[0]:
            best = (mag, cmd, reader.device.name)
    if best is None:
        return None, None
    return best[1], best[2]


def run_drive(body: Body, cfg: DriveConfig | None = None) -> None:
    """Block, rescan joysticks, and step the body while a stick is held."""
    cfg = cfg or DriveConfig()
    readers: dict[str, JoystickReader] = {}
    last_scan = 0.0
    print("[joy] scanning all joysticks; hold the left stick to step. Ctrl-C to stop.", flush=True)
    try:
        while True:
            now = time.monotonic()
            if now - last_scan >= cfg.rescan_s:
                last_scan = now
                found = scan_joysticks()
                paths = {d.path for d in found}
                for path in list(readers):
                    if path not in paths:
                        print(f"[joy] removed {readers[path].device.name} {path}", flush=True)
                        readers[path].close()
                        del readers[path]
                for dev in found:
                    if dev.path not in readers:
                        try:
                            readers[dev.path] = JoystickReader(dev)
                        except OSError as e:
                            print(f"[joy] cannot open {dev.path}: {e}", flush=True)
                            continue
                        via = f" via {dev.by_id}" if dev.by_id else ""
                        print(
                            f"[joy] found js{dev.index} {dev.name!r} "
                            f"axes={dev.axes} buttons={dev.buttons}{via}",
                            flush=True,
                        )
            for reader in readers.values():
                reader.poll()
            command, source = _strongest(list(readers.values()), cfg.deadzone)
            if command:
                try:
                    ok = body.act(command)
                except Exception as e:  # noqa: BLE001
                    ok = False
                    print(f"[joy] {command} error: {e}", flush=True)
                print(f"[joy] {source} -> {command} {'ok' if ok else 'failed'}", flush=True)
            else:
                time.sleep(0.05)
    finally:
        for reader in readers.values():
            reader.close()
            print("[joy] closed", flush=True)
