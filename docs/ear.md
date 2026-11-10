# The ear: hear, answer, and sometimes move

The ear is the wake-word loop on the TonyPi. WonderEcho sits on `/dev/ttyUSB0`. The microphone and speaker are the USB card named `Device`. This page is how a turn is routed. Why each choice was made is in [findings-ear.md](findings-ear.md). What is still unfinished is in [roadmap-ear.md](roadmap-ear.md).

## Install

One command tells you what is still missing:

```bash
contrib/check-ear-local.sh
```

Python is the small part. Why only pyserial: the wake word is a serial frame, and hearing, the reply, and the voice are programs outside this package.

```bash
pip install -e ".[ear]"
```

The harness ear also opens the camera when `[camera]` is enabled, so on the robot install the vision extra too:

```bash
pip install -e ".[ear,vision]"
```

`alsa-utils` supplies `arecord` and `aplay`. Why: the ear records a wav and Piper plays one. The package name on Debian and Raspberry Pi OS is `alsa-utils`.

The local stack is four files, two folders, and one service. About 2 GB. Why the hearing model and its projector share a folder: llama-server only attaches the audio projector when it sits next to the GGUF, and the model id the ear sends is the folder name.

| Piece | Path |
| --- | --- |
| Hearing model | `/var/lib/bruhos/models/qwen3-asr/Qwen3-ASR-0.6B.Q4_K_M.gguf` |
| Audio projector | `/var/lib/bruhos/models/qwen3-asr/Qwen3-ASR-0.6B.mmproj-Q8_0.gguf` |
| Reply model | `/var/lib/bruhos/models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` |
| Voice | `/var/lib/bruhos/voices/en_US-lessac-medium.onnx` and the `.onnx.json` beside it |
| Server | `/usr/local/bin/llama-server` |
| Piper | `/usr/local/bin/piper` |

```bash
mkdir -p /var/lib/bruhos/models/qwen3-asr /var/lib/bruhos/models/tinyllama /var/lib/bruhos/voices

curl -fL -o /var/lib/bruhos/models/qwen3-asr/Qwen3-ASR-0.6B.Q4_K_M.gguf \
  https://huggingface.co/mradermacher/Qwen3-ASR-0.6B-GGUF/resolve/main/Qwen3-ASR-0.6B.Q4_K_M.gguf
curl -fL -o /var/lib/bruhos/models/qwen3-asr/Qwen3-ASR-0.6B.mmproj-Q8_0.gguf \
  https://huggingface.co/mradermacher/Qwen3-ASR-0.6B-GGUF/resolve/main/Qwen3-ASR-0.6B.mmproj-Q8_0.gguf
curl -fL -o /var/lib/bruhos/models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

curl -fL -o /var/lib/bruhos/voices/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -fL -o /var/lib/bruhos/voices/en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

Piper is a venv so it does not fight the robot's system Python:

```bash
python3 -m venv /opt/piper
/opt/piper/bin/pip install piper-tts
ln -sf /opt/piper/bin/piper /usr/local/bin/piper
```

llama-server on this Pi is a source build of tag `b12242`. Why not the Ubuntu arm64 tarball: that build needs glibc 2.38, and this image has 2.36. The binary keeps its runpath into the build directory, so leave the tree at `/opt/llama.cpp/src`.

```bash
sudo apt-get install -y build-essential cmake git
git clone --depth 1 --branch b12242 https://github.com/ggml-org/llama.cpp.git /opt/llama.cpp/src
cmake -S /opt/llama.cpp/src -B /opt/llama.cpp/src/build \
  -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF
cmake --build /opt/llama.cpp/src/build -j 3
ln -sf /opt/llama.cpp/src/build/bin/llama-server /usr/local/bin/llama-server
```

`-j 3` leaves a core for the camera. CUDA and Vulkan are off because this board has neither.

Then the service, and the ear unit that is already in this repo:

```bash
sudo cp contrib/zerollama-llama.service /etc/systemd/system/
sudo systemctl enable --now zerollama-llama.service
sudo cp contrib/bruhos-ear.service /etc/systemd/system/
sudo systemctl enable --now bruhos-ear.service
```

`remote` needs none of the model files. It needs the speech and brain URLs in the config the service opens, and a network path to that zerollama.

## Two stacks

`[ear] stack` picks who hears and who answers. The default is `local`. Omit the key and the ear still stays on the Pi.

| Stack | Who hears | Who answers | Who speaks |
| --- | --- | --- | --- |
| `local` | llama-server, model `qwen3-asr` | llama-server, model `tinyllama` | Piper on this Pi |
| `remote` | whatever `asr_model` the config names, on that file's speech URL | whatever `brain.model` the config names, on that file's brain URL | that server's speech endpoint |

`remote` is the upgrade. Zerollama on another machine already has a larger model loaded, so the reply comes back faster than TinyLlama on this CPU. The eye, the scan, and detect stay on the URLs in the config either way. Why: those jobs are not the spoken turn.

```bash
bruhos ear --config configs/zerollama.toml                 # local
bruhos ear --config configs/zerollama.toml --stack remote  # zerollama URLs in the file
```

The service reads `configs/zerollama.local.toml` when that file exists, and `configs/zerollama.toml` otherwise. A `stack` line in the file the service actually opens is the one that counts.

## What local runs

llama-server listens on `127.0.0.1:11435` and loads one model at a time from `/var/lib/bruhos/models`. The router id is the folder name: `qwen3-asr`, `tinyllama`. `contrib/llama-models.ini` is passed as `--models-preset` so TinyLlama's prompt ends with the newline its GGUF dropped. Why that file and not one template for the whole server: the hearing model uses a different template, and replacing both would break transcription.

Qwen3-ASR's transcript includes a `language …<asr_text>` tag in front of the words. The ear keeps the words. Why the tag is parsed instead of prevented: that tag is the model's answer format, and the transcription call cannot prefill it with the configured language.

A local turn records a wav, posts it to `/v1/audio/transcriptions`, then asks `/v1/chat/completions`, then speaks with Piper. Why it does not open the duplex socket: that socket is zerollama's pipeline (hear, think, and speak on one connection). llama-server does not speak it.

Piper is `/usr/local/bin/piper` with `/var/lib/bruhos/voices/en_US-lessac-medium.onnx`. Why it is not llama-server's speech route: the voice is ONNX. That route expects a GGUF voice, and this Pi does not have one.

There is also an Ollama-shaped front on `127.0.0.1:11434` (`/opt/zerollama/ollama_front.py`). The ear does not use it. Why: the ear already speaks `/v1`, and llama-server is the process that has the models.

## What a reply log means

After the spoken text, the ear logs the generation rate of the completion that produced that text:

```
[ear] REPLY +45.72s '...'
[ear] GEN 13.6 t/s 130 tok
```

That number is llama-server's `predicted_per_second`. It is not Piper, and it is not the earlier plan request. The wall clock of the turn is longer, because the server holds only one model and has to unload the hearing model before the reply model can load.

## When a sentence becomes a step

The harness reads skill calls out of the reply and runs them on the TonyPi body. A call can look like `go { "direction": "forward", "steps": 3 }` or like `{ "go": { "direction": "forward", "steps": 3 } }`. The second shape puts the skill name inside the object. Both are dispatched. The braces are not read aloud. After the skills run, a second completion asks for one spoken sentence about what happened.

The duplex path does not run skills. It only streams the microphone and plays audio, and it leaves the servo board closed.
