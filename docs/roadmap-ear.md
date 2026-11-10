# Roadmap: the ear

Local is the default because the Pi can finish a turn alone. Remote zerollama is the upgrade when the reply has to come back faster than TinyLlama on four cores. Why that split stays: pulling the remote in as the default makes the robot mute whenever that machine is down, and pretending TinyLlama is as fast as a loaded GPU model is how the `+45s` turn surprised us.

## Now

- `[ear] stack` is `local` or `remote`. Missing means `local`.
- Local hears with `qwen3-asr`, answers with `tinyllama`, and speaks with Piper. Both models go through llama-server on `127.0.0.1:11435`. The server keeps one model.
- Remote uses the speech and brain URLs already in the config file the service opened.
- A reply log is followed by `GEN`, the generation rate of the completion that produced the spoken text.
- Skill objects with the name inside the braces are dispatched.
- Qwen3-ASR's `language …<asr_text>` tag is parsed off before the reply sees the turn. TinyLlama is prompted with the newline its GGUF dropped.

## Next

1. **One completion when nothing is going to move.** Why: a chat question currently spends a plan request and then a speech request, and each one waits on a model that may have just been loaded. If the plan comes back with no skill, the plan text should be the reply, or the plan request should be skipped for a turn that cannot be a motion.
2. **Keep the hearing model and the reply model from evicting each other.** Why: `--models-max 1` makes every local turn pay a load. Two slots, or a hearing path that does not occupy the reply slot, is the difference between a 4-second transcript plus a 14 token/second answer and a turn that runs most of a minute. Memory is the constraint, not the idea.
3. **One owner of `/dev/ttyAMA0` before the ear walks.** Why: the gait and the IMU reader share the board object, the reader closes the port on a bad read, and TonyPi, the joystick service, the lab relay, and the web head already have the device open. A decoded `go` will keep dying mid-step until those other readers are gone and the gait write cannot observe a closed port. The body side of this is [roadmap-body.md](roadmap-body.md).

## Later

- Let `remote` name the model tags as well as the URLs, still without copying a host into the public config. Why: the file the service prefers is private, and the public example should stay a shape, not a machine.
- A duplex path that is actually llama-server, if its realtime socket can hear and speak without the zerollama pipeline. Why it is not the local default now: the ear's duplex client speaks zerollama's query string and event stream, and pointing that at llama-server would fail open.
- Teach the reply model that a greeting is not a walk. Why: once `{ "go": ... }` is a real call, an invented plan moves the robot. The parser is doing what it was asked. The model is not.

## Not this

Do not put the remote host in a tracked file. Why: it was committed once, and the public branch had to be rewritten to get it back out. The private file stays gitignored.

Do not send scan, detect, or the eye through TinyLlama just because the ear is local. Why: those calls need a vision model. The ear's stack flag is the spoken turn only.
