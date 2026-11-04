#!/bin/bash
set -euo pipefail
ROOT=/root/Sites/bruhOS
cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
exec "$ROOT/.venv/bin/bruhos" web --config "$ROOT/configs/zerollama.toml" --tls
