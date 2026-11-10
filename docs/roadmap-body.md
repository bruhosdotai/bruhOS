# Roadmap: body torque and the controller

The limb script is a manual tool. It is not a skill. The duplex ear still leaves the board closed. The harness ear opens `/dev/ttyAMA0` when a skill runs, and that races the IMU reader in the same process. Why the unload stays a script: TonyPi already owns the port, a second reader steals replies, and an unload of a leg folds the robot. A voice skill that can do that by accident is worse than a script you run on purpose. What the ear learned about that port is in [findings-ear.md](findings-ear.md).

## Now

`scripts/limbs.py` can unload or load each hand, each arm, and each leg, plus both arms and both legs. Packets match Hiwonder's `buf_write`. The head is out of scope because PWM has no unload.

## Next

1. **One owner of the port.** Why: reads are unreliable while TonyPi, the joystick process, and the web app all hold the device. A torque read and a live position read have to land in the process that already has the reader thread. Until then, `on` cannot see where a sagged limb actually is.
2. **Load at the live angle.** Why: today's `on` locks the last target, so a limb that folded while off drives back there. The safe power-up is read position, then command that same pulse. That needs the single owner from step 1. It matters most for the legs.
3. **Remember who is unloaded.** Why: the bit lives in the servo, not in bruhOS. After a reboot we do not know which limbs are limp unless we read torque. The next gait should refuse to start while a leg it needs is unloaded, or it should load that leg on purpose before the first frame.
4. **Teach the action player about torque.** Why: `runAction` turns torque back on for every joint in the file, which silently undoes `off`. A walk after a leg unload should be an explicit load of those ids, not a surprise inside the gait.

## Later

- Per-limb load as the first step of a skill that uses that limb. Why this still waits: the harness ear already opens the board when a skill runs, and that races the IMU reader plus TonyPi. Load-on-use belongs in the one process that owns the port, which the ear is not.
- A battery reading with a known scale, from the controller's system packet, once one process owns the replies. Why it is not in the script: the Python prints an integer and does not document the unit.
- Leave spin mode, servo LED, and broadcast queued moves unused. Why: they are real on the LX-824HV and absent from board B's Python API, and a joint in spin mode is no longer a joint.

## Not this

GR00T joint targets still do not map onto these 16 servos. Why: the runtime's job is whitelisted skills and action files, and a raw 16-wide action chunk skips the torque and the fall recovery this body actually has.
