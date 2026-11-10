# TonyPi body: what we can tell it

The lab robot is a Hiwonder TonyPi on a Pi 5. Sixteen body joints are LX-824HV bus servos. The head is two PWM servos. Both kinds are driven by expansion board B, the microcontroller on `/dev/ttyAMA0`.

This page is the operator's map. Why each conclusion was drawn is in [findings-body.md](findings-body.md). What is still unfinished is in [roadmap-body.md](roadmap-body.md).

## Why there are two kinds of servo

A bus servo is a small closed-loop joint on a shared serial chain. It knows its own angle, and it can be told to let go. A PWM servo only remembers the last pulse the board sent it. The head is the second kind, which is why limb power-down leaves the head holding.

## Bus joints

Pulse 0 is 0°, 500 is 120°, 1000 is 240°. One count is 0.24°. A move also carries a time, from 0 to 30000 ms. Asking for less time than the motor can manage just runs it at full speed: about 0.20 s per 60° at 11.1 V, so a full sweep cannot finish in under about 0.8 s.

| Command | What it does | Why it matters |
| --- | --- | --- |
| Move | Drive to a pulse in the given time. | Also turns the motor back on. Playing any action file does this for every joint in the file. |
| Unload | Drop torque. No new target. | The hold is what heats the case. The joint stays at the angle it already has, then sags under gravity. The servo stores this, so rebooting TonyPi does not turn the motor back on. |
| Load | Hold the last commanded pulse. | A limb that sagged while unloaded drives back to that pulse. |
| Stop | Cut the move short. | Torque stays on. This is not power-down. |
| Read | Position, voltage, temperature, torque on or off, offset. | Needs a reply. Several processes already share the port, so a read is often stolen. A write does not need a reply. |

The ids were taken from this image's action files. The SDK never names a joint.

| Group | Ids | What they are |
| --- | --- | --- |
| `left-leg` | 1–5 | Foot roll, ankle pitch, knee, hip pitch, hip roll |
| `right-leg` | 9–13 | Same order |
| `left-arm` | 6, 7, 8 | 8 is the hand alone |
| `right-arm` | 14, 15, 16 | 16 is the hand alone |
| `left-hand` | 8 | The joint `left_hand.d6a` moves by itself |
| `right-hand` | 16 | The joint `right_hand.d6a` moves by itself |
| `arms` | 6, 7, 8, 14, 15, 16 | Both arms, hands included |
| `legs` | 1–5 and 9–13 | Both legs |

There is no `hands` group. The two hands are separate because that is how they are used.

```bash
scripts/limbs.py off left-hand
scripts/limbs.py off right-hand
scripts/limbs.py off left-arm
scripts/limbs.py off right-arm
scripts/limbs.py off left-leg      # that leg folds
scripts/limbs.py off right-leg     # that leg folds
scripts/limbs.py on left-arm
```

`on` does not sample the live angle first. Why: the reliable read path is the board's reply queue, and TonyPi's reader is already consuming it. Load is the command that needs no reply. The cost is the snap back to the old target.

The chip can also spin continuously, blink its own LED, and queue a move until a broadcast start. Expansion board B's Python API does not forward those. Why we leave them alone: a leg in spin mode is not a joint anymore, and we have no wrapper that speaks the servo's own 115200 baud protocol. The Pi only speaks to the controller.

## Head

PWM port 1 is pitch, port 2 is yaw. 500–2500 µs covers 0–180°, and 1500 µs is 90°. On this robot the safe window is narrower. Pitch center is the calibrated table-down pulse (about 914) and look-up is toward 1800. Yaw center is about 1477, and the scan grid only travels about 400 µs either side, with the sign flipped because this servo is mirrored.

`Head.go_pulse`, `go_pose`, and `recenter` send those pulses. They cannot unload the head.

## Controller

Expansion board B is a Cortex-class microcontroller on the Pi's UART at 1,000,000 baud. Frames are `0xAA 0x55`, a function byte, a length, the payload, and a CRC8. DTR and RTS stay off because asserting them resets the chip. Reception has to be enabled before a report is kept, and each report queue holds one sample. Why that shape: the board streams the IMU continuously, and a slow reader would otherwise block the serial thread.

| Part | Use |
| --- | --- |
| Six bus ports | One chain of the 16 body servos. |
| Two PWM ports | The head. |
| Buzzer | Frequency, on time, off time, repeat. Repeat 0 keeps going. TonyPi beeps once at startup. |
| IMU | Six floats, acceleration and gyro. On this robot X is vertical while standing. The units are not labeled, so a still window has to scale them before a distance means anything. |
| Two buttons | A long hold of one clears saved Wi-Fi. A long hold of the other halts the Pi. |
| Battery and charger | The pack plugs in here. The library can print a battery word when the chip sends one. The scale is not documented in the Python. |
| Fan and spare pins | Raspberry Pi GPIO, not a serial command. |

The same Python file also has calls for four DC motors, RGB lamps, an OLED, a gamepad packet, and SBUS. Those connectors are on expansion board A and on Hiwonder's separate ROS control board. Sending them here has nowhere to land.

## Who owns the port

TonyPi, the joystick process, and the web app already have `/dev/ttyAMA0` open. `scripts/limbs.py` writes its frames and closes the port so it does not add another reader. A second `Board()` steals reply bytes. That is why unload works and a status read usually does not.

Any later gait undoes an unload. `runAction` sends position frames, and a position frame turns torque back on for every servo in the file.
