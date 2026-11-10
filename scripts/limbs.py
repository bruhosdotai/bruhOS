#!/usr/bin/env python3
"""Drop or restore torque on TonyPi limbs without commanding a new pose.

Why torque, not a new pulse: an LX-824HV holds its last target with a closed
position loop, and that hold is what heats the case. The joints were already
at the pose, so a move command would only clamp harder. ``off`` sends unload
and no target. ``on`` sends load, which locks that same last target. A limb
that sagged while unloaded drives back. This does not play a stand file.

Why the head stays on: pitch and yaw are PWM servos. The expansion board
latches a pulse and has no unload for them.

Why this opens the serial port itself: TonyPi.py already owns ``/dev/ttyAMA0``.
Constructing ``Board()`` starts a second reader and can pulse DTR, which
resets the controller. These frames are the same bytes ``unload_bus_servo``
writes, and they need no reply.

    scripts/limbs.py off left-hand
    scripts/limbs.py off right-hand
    scripts/limbs.py off left-arm
    scripts/limbs.py off right-arm
    scripts/limbs.py off left-leg
    scripts/limbs.py off right-leg
    scripts/limbs.py on left-arm

See docs/body.md, docs/findings-body.md, and docs/roadmap-body.md.
"""

from __future__ import annotations

import argparse
import sys
import time

PORT = "/dev/ttyAMA0"
BAUD = 1_000_000  # expansion-board link, not the 115200 servo bus
BUS_FUNC = 5      # PacketFunction.PACKET_FUNC_BUS_SERVO
# The SDK names the argument "enable" and then sends 0x0B when it is true.
# unload_bus_servo passes 1, and that is the phone app's power-down path,
# so 0x0B is unload. 0x0C is load. On the servo, command 31 with parameter
# 0 is the same unload, and the servo stores it, which is why a reboot of
# TonyPi does not by itself turn the motor back on.
UNLOAD = 0x0B
LOAD = 0x0C

# Ids come from diffs of this image's action files, because the SDK never
# names a joint. left_hand.d6a moves 8 alone and right_hand.d6a moves 16
# alone, so the hands are separate groups. There is no combined "hands"
# name. Each arm is those plus the other two joints of that side. Legs are
# separate for the same reason: unloading either one folds that side.
GROUPS: dict[str, tuple[int, ...]] = {
    "left-hand": (8,),
    "right-hand": (16,),
    "left-arm": (6, 7, 8),
    "right-arm": (14, 15, 16),
    "arms": (6, 7, 8, 14, 15, 16),
    "left-leg": (1, 2, 3, 4, 5),
    "right-leg": (9, 10, 11, 12, 13),
    "legs": (1, 2, 3, 4, 5, 9, 10, 11, 12, 13),
}

LEG_IDS = frozenset(GROUPS["legs"])

_CRC8 = (
    0, 94, 188, 226, 97, 63, 221, 131, 194, 156, 126, 32, 163, 253, 31, 65,
    157, 195, 33, 127, 252, 162, 64, 30, 95, 1, 227, 189, 62, 96, 130, 220,
    35, 125, 159, 193, 66, 28, 254, 160, 225, 191, 93, 3, 128, 222, 60, 98,
    190, 224, 2, 92, 223, 129, 99, 61, 124, 34, 192, 158, 29, 67, 161, 255,
    70, 24, 250, 164, 39, 121, 155, 197, 132, 218, 56, 102, 229, 187, 89, 7,
    219, 133, 103, 57, 186, 228, 6, 88, 25, 71, 165, 251, 120, 38, 196, 154,
    101, 59, 217, 135, 4, 90, 184, 230, 167, 249, 27, 69, 198, 152, 122, 36,
    248, 166, 68, 26, 153, 199, 37, 123, 58, 100, 134, 216, 91, 5, 231, 185,
    140, 210, 48, 110, 237, 179, 81, 15, 78, 16, 242, 172, 47, 113, 147, 205,
    17, 79, 173, 243, 112, 46, 204, 146, 211, 141, 111, 49, 178, 236, 14, 80,
    175, 241, 19, 77, 206, 144, 114, 44, 109, 51, 209, 143, 12, 82, 176, 238,
    50, 108, 142, 208, 83, 13, 239, 177, 240, 174, 76, 18, 145, 207, 45, 115,
    202, 148, 118, 40, 171, 245, 23, 73, 8, 86, 180, 234, 105, 55, 213, 139,
    87, 9, 235, 181, 54, 104, 138, 212, 149, 203, 41, 119, 244, 170, 72, 22,
    233, 183, 85, 11, 136, 214, 52, 106, 43, 117, 151, 201, 74, 20, 246, 168,
    116, 42, 200, 150, 21, 75, 169, 247, 182, 232, 10, 84, 215, 137, 107, 53,
)


def checksum(data: bytes) -> int:
    check = 0
    for b in data:
        check = _CRC8[check ^ b]
    return check & 0xFF


def packet(subcommand: int, servo_id: int) -> bytes:
    """One bus-servo frame.

    The comment above ``PacketControllerState`` says length comes before the
    function byte. ``Board.buf_write`` does the opposite, and the receiver
    state machine reads function then length. Matching the writer is why
    the body is ``func, len, subcommand, id``.
    """
    body = bytes((BUS_FUNC, 2, subcommand, servo_id))
    return bytes((0xAA, 0x55)) + body + bytes((checksum(body),))


def ids_for(names: list[str]) -> list[int]:
    chosen: list[int] = []
    seen: set[int] = set()
    for name in names:
        for servo_id in GROUPS[name]:
            if servo_id not in seen:
                seen.add(servo_id)
                chosen.append(servo_id)
    return chosen


def send(payloads: list[bytes], port: str = PORT) -> None:
    import serial

    ser = serial.Serial()
    ser.baudrate = BAUD
    ser.timeout = 0.05
    ser.write_timeout = 0.5
    # Set these before open. The SDK does the same thing because asserting
    # DTR or RTS resets the controller mid-command.
    ser.rts = False
    ser.dtr = False
    ser.port = port
    try:
        ser.open()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"could not open {port}: {e}") from e
    try:
        for payload in payloads:
            ser.write(payload)
            ser.flush()
            # The SDK waits 20 ms after every torque write so two short
            # frames are not fused on the half-duplex servo chain.
            time.sleep(0.02)
    finally:
        ser.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("off", "on"))
    parser.add_argument("groups", nargs="+", choices=sorted(GROUPS))
    parser.add_argument("--dry-run", action="store_true", help="print the servo ids and do not open the port")
    parser.add_argument("--port", default=PORT)
    args = parser.parse_args(argv)

    servo_ids = ids_for(args.groups)
    sub = UNLOAD if args.action == "off" else LOAD
    # A leg unload is the same packet as a hand unload. The warning exists
    # because that leg is what holds the body up, so it folds.
    if args.action == "off" and LEG_IDS.intersection(servo_ids):
        print("legs off: those joints will fold.", file=sys.stderr)
    # Load does not sample the live angle. The servo still has the old
    # target, so a limb that moved while limp drives back to it.
    if args.action == "on":
        print("on locks the last commanded pose. A limb that moved while off will drive back there.", file=sys.stderr)
    print(f"{args.action} {' '.join(args.groups)}: servos {', '.join(str(i) for i in servo_ids)}")
    if args.dry_run:
        return 0
    send([packet(sub, servo_id) for servo_id in servo_ids], port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
