# Findings: servos and the controller

What we learned on this TonyPi, and why each claim is believed. The command map that follows from it is [body.md](body.md).

## The holding current is the heat

An LX-824HV in servo mode closes a loop on its potentiometer until it is unloaded. Sixteen of those loops, plus two PWM servos that latch their pulse, sit there after `stand_slow` at TonyPi startup. The cases get hot with no motion left to do.

Why unload rather than a rest pose: the joints were already at the pose. A new pulse would enable torque again and hold a different angle just as hard. Command 31 parameter 0 (board subcommand `0x0B`) drops the motor drive and leaves the electronics on, so the joint can still report an angle while it hangs.

Why the head could not join them: the head servos are LFD-01M PWM parts on ports 1 and 2. The board's PWM command is "go to this pulse." There is no unload in that function. They stay locked.

## The SDK flag is backwards

`Controller.unload_bus_servo` calls `bus_servo_enable_torque(id, 1)`. In the SDK, a true flag packs subcommand `0x0B` and a false flag packs `0x0C`. The phone RPC `UnloadBusServo` uses that same call for ids 1–15. So `0x0B` is unload, whatever the parameter name says.

Why this had to be checked against the caller: trusting the English name `enable_torque(id, 1)` would have turned the motors harder on while we were trying to cool them.

The servo stores load and unload. Why a TonyPi restart does not restore torque by itself: the bit lives in the servo, and the process that exits is only the sender. The next position command, or an explicit load (`0x0C`), is what turns the motor back on. Load aims at the last target, which is why a limb that sagged while off drives back when it is powered up.

## Joint ids come from the action files

Nothing in the SDK labels servo 8 as a hand. The names were read off this image's `.d6a` files by comparing each clip with the last frame of `stand.d6a`.

Why that method: a published diagram can be for another revision. These files are what this robot actually plays.

- `left_hand.d6a` moves only servo 8. `right_hand.d6a` moves only 16. That is why the hands are two groups and there is no `hands` group.
- `lift_left_hand.d6a` moves 6 and 8. `grab_stand_left.d6a` moves 7 and 8. `grab_stand_right.d6a` moves 15 and 16. `wave.d6a` moves 14 and 16. So the left arm is 6, 7, 8 and the right arm is 14, 15, 16.
- `bow.d6a` bends both pitch chains: 2 with 10, 3 with 11, 4 with 12. Those are ankle, knee, and hip pitch. `left_kick.d6a` swings 5, `right_kick.d6a` swings 13, and the sidesteps use 1 and 9. Those are the roll joints. Left leg is 1–5, right leg is 9–13, in order foot roll, ankle pitch, knee, hip pitch, hip roll.
- Standing pulses mirror around 500 across those pairs (7 at 800 with 15 at 200, 8 at 725 with 16 at 275). That is the same pairing.

`pose_control.py` drives 6, 7, 14, and 15 from the camera person's arms and never touches 8 or 16. Why we did not use it as the arm map: it is a two-joint mimic, and the gait files show a third joint per arm.

## What an LX-824HV actually is

TonyPi's Pi 4/5 spec calls the body servos LX-824HV, on the 11.1 V pack. The protocol below is the Hiwonder / LewanSoul bus protocol those parts speak. We did not read a label on a horn.

| Fact | Why it changes a command |
| --- | --- |
| 0–1000 maps to 0–240°. Accuracy claimed 0.3°. | The sensor is a potentiometer, one turn. Past the stop, the reading is meaningless. |
| 17 kg·cm and 0.20 s/60° at 11.1 V. Supply 9–12.6 V. | Time-to-target is the only rate. A requested time shorter than the motor can do is just full speed. |
| Move time is 0–30000 ms. | The board shares one duration across every id in the packet. |
| Offset ±125 counts, about ±30°. Adjust is RAM. Save is kept. | Centering a joint does not require moving the horn. |
| Voltage and temperature limits are kept. Out of range, the servo unloads itself and blinks. Temperature alarm is 50–100 °C. Clearing the alarm restores torque. | A hot servo can let go on its own, then clamp again when it cools. |
| Spin mode is a speed from −1000 to 1000 and is forgotten on reboot. | The chip can do it. Board B's Python API has no call for it. |
| Id 0–253, 254 means every servo on the chain. | A broadcast write is how a whole chain is addressed. We address one id at a time so a limb can drop alone. |

Stall current on the HV part is about 1.7–2 A. Why sixteen holding servos matter: the idle current of a closed loop, times the chain, is the heat and the battery load even when the robot is still.

## The controller is a translator

The Pi never opens the servo bus. That bus is 115200 baud, half duplex, behind the expansion board. The Pi talks to the board at 1,000,000 baud. Why the script still builds the board frame: the board is the only device on `/dev/ttyAMA0`.

The published comment above the state machine says the length byte comes first. `Board.buf_write` writes the function byte first, and the parser reads it that way. The limb frames follow `buf_write`. Why: a frame that matches the comment is a frame the chip ignores.

Board B's own connectors are six bus sockets, two PWM sockets, a buzzer, an MPU6050, two buttons, the pack connector and charger, a fan, and a few GPIO pins. The fan and the buttons are reached through the Pi's GPIO. The installed `button_scan.py` treats a long hold of one key as a Wi-Fi reset and a long hold of the other as `halt`. The wiki text about a short-press self-check is a different description than the script on this image.

`get_imu()` returns six floats. Why roam does not treat them as degrees per second out of the box: the unit is undocumented, the still-window magnitude is what turns acceleration into g, and an in-place turn is what turns the gyro into degrees. On this board, chip X is the vertical axis when the robot is standing. Hiwonder's own fall check is `atan2(ax, az)`.

The SDK also implements motors, RGB, OLED, a gamepad, and SBUS. Why those are not part of this robot's API: the wiki assigns them to expansion board A and to the separate ROS control board. Board B's connector table does not list them.

## The port is already shared

`fuser` on `/dev/ttyAMA0` showed TonyPi, the joystick process, multi_control, and the bruhOS web app all holding the device. Why a second `Board()` is the wrong way to unload: it starts a reader thread, and opening the port with DTR asserted resets the microcontroller. The SDK clears DTR and RTS before `open` for that reason. The limb script does the same, writes, and closes.

A torque-state read after the arm unload came back mixed with IMU frames (function byte 7). Why the write is still believed: unload does not need a reply, the port stayed up and kept streaming IMU, and the frames match `buf_write` byte for byte in `tests/test_limbs.py`.

The stock RPC power-down stops at id 15. Servo 16 is the right hand, so that RPC leaves it holding. The script includes 16.

## What was run

On this robot the script unloaded servos 1–16. The head was left holding, because that is the PWM pair. Legs fold when they are unloaded. An action file played afterwards will torque the joints in that file back on.
