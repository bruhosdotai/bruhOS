# Findings: the ear

What we learned by running the ear on this TonyPi, and why each claim is believed. The operator's map is [ear.md](ear.md). What is still unfinished is in [roadmap-ear.md](roadmap-ear.md).

## The Pi can hold a whole turn

Qwen3-ASR 0.6B Q4 and TinyLlama 1.1B Q4 both run on the Pi 5 under llama.cpp built here. The prebuilt llama.cpp binary needed a newer glibc than Debian on this board has, which is why the server is a native build at `/opt/llama.cpp` rather than the release tarball.

A five-second clip transcribed by `qwen3-asr` in about 4 seconds. TinyLlama then generated the spoken reply at about 14 tokens per second (`predicted_per_second` on the completion that produced the text, 130 tokens). Piper Lessac synthesized a short line in about 0.2 seconds, roughly eight times faster than realtime.

Why this is the default: a robot that can only answer when another machine is up is down whenever that machine or the network is. Local is the floor. Remote zerollama is the upgrade for a faster reply, because that server keeps a larger model loaded and does not spend the turn swapping weights on four CPU cores.

## One model slot is why local feels slow

llama-server was started with `--models-max 1`. The hearing model and the reply model cannot both stay resident. On the turn that asked who is on the twenty-dollar bill, the server unloaded `qwen3-asr` and spawned TinyLlama before the first token of the reply. The log line `GEN` reports generation speed after that load. The `+45s` on `REPLY` includes the swap and a first completion that was only the plan request.

Why the plan request exists: the harness asks for a JSON plan before it asks for speech, so a chat question still costs two completions. The spoken paragraph was the second one. TinyLlama also invented the bill (a bald eagle, a ten, "the penny"). That is the quality cost of the floor model, separate from the tokens per second.

## The hearing model writes a wrapper, and the reply template was short a newline

`qwen3-asr` returned `language en<asr_text>Who is on the twenty-dollar bill?`. The words were right. Why the wrapper is the model and not the microphone: the same `<asr_text>` marker showed up in an earlier file transcription of a Piper clip, and the remote `whisper-base` turn of the same question came back as plain text. Qwen3-ASR is trained to answer in that shape. Its own toolkit splits on `<asr_text>` and keeps the words, and treats `language None` with nothing after the tag as silence. llama.cpp's transcription prompt does not prefill the tag, so the tag arrives as text. The ear now parses it before the reply. Why not a fixed `language English<asr_text>` prefill on the hearing model: the transcription API cannot see the configured language as a template variable, and a hardcoded English prefix would be wrong for Chinese, Malay, Thai, or Vietnamese.

The reply then started `languaage:`. TinyLlama's GGUF ends the prompt at `<|assistant|>` with no newline. The model was trained to continue after `<|assistant|>` plus a newline. The router preset `contrib/llama-models.ini` gives `tinyllama` that newline. Why it is not a router-wide `--chat-template`: Qwen3-ASR is ChatML, and one template for both models would break hearing. A leading `languaage:` on a reply is dropped so a turn that still echoes the tag is not spoken that way.

The JSON plan request still cannot use `response_format`. Why, after the newline was put back: TinyLlama tokenizes `<|assistant|>` as a space glued to `<`, and llama.cpp drops that first token before the grammar sees it, so the sampler refuses the tag. The retry sends a JSON grammar of its own. That grammar is not prefilled with the tag, so it only constrains the words the model still has to write. A server that rejected the format for a different reason does not receive the grammar.

## Piper is not a llama.cpp model

The installed voice is `en_US-lessac-medium.onnx`, 63 MB, under `/var/lib/bruhos/voices`. llama-server does expose `/v1/audio/speech`, and its router would forward a JSON body. That route is for a GGUF voice. This disk has none. Why the ear shells out to the `piper` binary instead of posting speech at port 11435: an ONNX file never becomes a model id in the models directory.

The Ollama-shaped front on port 11434 only translates `/api/chat`. Why the ear talks to port 11435 directly: transcriptions and chat completions are already on llama-server, and the front does not add them.

## Skill names inside the braces were spoken, not run

The reply model wrote `{ "wave": {} }` and `{ "go": { "direction": "forward", "steps": 3 } }` and the ear logged `DECIDE chat-only`, then Piper read the braces. Why the old scraper missed them: it required the skill name before the `{`, as in `go { "direction": "forward" }`. In the new shape the name is the object's key. The ear now treats an object as calls when every key is a known skill, and then speaks from what the skills did.

Why a chatty model makes this dangerous: "How are you doing?" produced a wave and a walk that the model invented. Once those objects are calls, the body will do them. The harness body is the TonyPi, not the sim named in `[body]`.

## The gait and the IMU share one UART

The failure on the first real `go` was `Attempting to use a port that is not open`, logged as `LLM_FAIL` because the ear wraps the whole turn. The microphone serial never closed. `/dev/ttyUSB0` stayed with the ear, and the process went back to waiting for the wake word.

`/dev/ttyAMA0` is the controller board. The gait writes servo frames on it. The ear also turns on the board IMU at startup, and that reader uses the same `Board` object. The Hiwonder receive loop treats a failed read as fatal: it closes the port, waits, and tries to open it again. The next servo frame then hits pyserial's "port that is not open." Why the timing fits a close rather than a busy open: the walk was logged, and the exception came about two seconds later, mid-step, which is the gap while the reader has the port shut.

Four other programs already had that UART open: `TonyPi.py` (phone-app backend, camera stream, and the demo modes), `Joystick.py` (USB gamepad to gait), `multi_control_client.py` (the lab relay's action groups), and `bruhos web` (the head). None of the first three are bruhOS. They are the vendor units from `/home/pi/TonyPi`. Their traffic on the wire is what makes the ear's reader drop the port.

The microphone and the IMU do not share a serial port. Audio is USB. The board is the Pi's UART.
