# Changelog

## 2026-11-10

### Ear stack defaults to the Pi

The ear's spoken turn defaults to `local`. Hearing is `qwen3-asr` and the reply is `tinyllama`, both through llama-server on this Pi. The voice is Piper. `remote` keeps the zerollama URLs in the config and is how you get a faster reply.

Why local is the default: the robot should answer when the other machine is down. Why remote is the upgrade rather than the baseline: that server holds a larger model already loaded, and the Pi's llama-server keeps only one model, so a local turn unloads the hearing model before the reply can start. Measured here, Qwen3-ASR took about 4 seconds on a 5-second clip and TinyLlama generated at about 14 tokens per second. The `GEN` line after `REPLY` is that generation rate, for the completion that produced the spoken text.

Why Piper is a separate program: the voice is ONNX. llama-server's speech route expects a GGUF voice. Why local does not use the duplex socket: that socket is zerollama's pipeline, which llama-server does not speak.

A `{ "go": { "direction": "forward", "steps": 3 } }` object is now a skill call. Why it used to be read aloud: the old scraper required the skill name before the brace.

The Python extra is `ear` (pyserial). Hearing, TinyLlama, and Piper are files and binaries, not pip packages, because the voice is ONNX and llama-server is a native build. `contrib/check-ear-local.sh` prints what is still missing. The commands are in [docs/ear.md](docs/ear.md).

The hearing transcript is the words. Qwen3-ASR answers as `language …<asr_text>` plus the words, and that tag was being handed to TinyLlama, which started the reply `languaage:`. Why it is parsed rather than prefilled: llama.cpp does not put the tag on the assistant turn, and a hardcoded language would ignore the one in the config. `language None` with nothing after the tag is silence.

TinyLlama's prompt now ends `<|assistant|>` plus the newline it was trained on. Why the GGUF was not left as shipped: its template omits that newline, so the model was continuing from the wrong position. The preset is `contrib/llama-models.ini`, on the `tinyllama` model only, so the hearing model's ChatML template stays. `response_format` still cannot start on this model, because the first token of that tag is a space glued to `<` and the sampler drops it. The plan request then uses a JSON grammar that is not prefilled with the tag.

Docs: [docs/ear.md](docs/ear.md), [docs/findings-ear.md](docs/findings-ear.md), [docs/roadmap-ear.md](docs/roadmap-ear.md).

### Limb torque script

`scripts/limbs.py` can drop or restore torque on this TonyPi without sending a new pose.

Why a script, and why no new pulse: the LX-824HV servos were already at their pose and heating up from the hold. A position command turns the motor back on. Unload (`0x0B`) lets the joint go limp. Load (`0x0C`) locks the last target again.

Groups are `left-hand`, `right-hand`, `left-arm`, `right-arm`, `arms`, `left-leg`, `right-leg`, and `legs`. There is no `hands` group, because the two hands are separate joints (servos 8 and 16). Leg commands warn that the leg will fold.

The script writes the controller frame itself and closes the port. Why it does not construct `Board()`: TonyPi already owns `/dev/ttyAMA0`, a second reader steals replies, and DTR must stay low or the controller resets.

Ids come from this image's action files. The SDK flag on `enable_torque` is inverted relative to its name; the script follows `unload_bus_servo`, which is the path the phone app calls power-down.

The head is unchanged. Pitch and yaw are PWM servos and have no unload command.

On this robot the script was run for both arms and then for both legs, so bus servos 1–16 were unloaded. The head kept holding.

Docs: [docs/body.md](docs/body.md), [docs/findings-body.md](docs/findings-body.md), [docs/roadmap-body.md](docs/roadmap-body.md).
