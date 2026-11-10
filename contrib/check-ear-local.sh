#!/bin/sh
# Report the local ear's requirements. Exit 0 when every piece is present.
# Why a check instead of an installer: llama-server is a source build, and
# the model files are large. This only says what is still missing.

set -u

root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
if [ -x "$root/.venv/bin/python" ]; then
  py="$root/.venv/bin/python"
else
  py=python3
fi

missing=0

ok() { printf 'ok    %s\n' "$1"; }
need() { printf 'NEED  %s\n' "$1"; missing=1; }

file_ok() {
  if [ -s "$2" ]; then ok "$1"; else need "$1 — $2"; fi
}

cmd_ok() {
  if command -v "$2" >/dev/null 2>&1; then ok "$1"; else need "$1 — $2"; fi
}

cmd_ok "arecord (alsa-utils)" arecord
cmd_ok "aplay (alsa-utils)" aplay
cmd_ok "piper" piper
cmd_ok "llama-server" llama-server

file_ok "qwen3-asr weights" /var/lib/bruhos/models/qwen3-asr/Qwen3-ASR-0.6B.Q4_K_M.gguf
file_ok "qwen3-asr projector (same folder as the weights)" \
  /var/lib/bruhos/models/qwen3-asr/Qwen3-ASR-0.6B.mmproj-Q8_0.gguf
file_ok "tinyllama weights" /var/lib/bruhos/models/tinyllama/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
file_ok "tinyllama chat template (newline the GGUF dropped)" \
  "$root/contrib/templates/tinyllama.jinja"
file_ok "llama-server model preset" "$root/contrib/llama-models.ini"
file_ok "piper voice" /var/lib/bruhos/voices/en_US-lessac-medium.onnx
file_ok "piper voice config" /var/lib/bruhos/voices/en_US-lessac-medium.onnx.json

if "$py" -c 'import serial' >/dev/null 2>&1; then
  ok "pyserial ($py)"
else
  need "pyserial — $py -m pip install -e \".[ear]\""
fi

if [ "$missing" -eq 0 ]; then
  printf 'local ear requirements are present\n'
else
  printf 'see docs/ear.md for the install commands\n'
fi
exit "$missing"
