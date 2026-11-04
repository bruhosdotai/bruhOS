#!/usr/bin/env bash
# Benchmark all cozmic VL models on a directory of stills (uses curl for HTTP).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
IMAGES="${1:-/tmp/scan}"
OUT="${2:-/tmp/vl_bench.json}"
BASE="${BASE_URL:-http://127.0.0.1:2083/v1}"
TIMEOUT="${TIMEOUT:-300}"
ROWS=/tmp/vl_bench_rows.jsonl
rm -f "$ROWS"

MODELS=$(curl -sS --max-time 30 -H 'User-Agent: bruhos-vl-bench/0.1' "$BASE/models" \
  | .venv/bin/python -c '
import json,sys
try:
    d=json.load(sys.stdin)
except Exception:
    sys.exit(0)
for m in d.get("data") or []:
    mods=m.get("input_modalities") or []
    caps=m.get("capabilities") or []
    mid=m.get("id") or ""
    if "image" in mods or "vision" in [str(c).lower() for c in caps]:
        print(mid)
')
if [[ -z "${MODELS}" ]]; then
  MODELS="gemma4:26b-a4b-it-qat
gemma4:31b
ministral-3:3b
qwen3.6-64k:latest
qwen3.6:27b
qwen3.6:35b
qwen3.8:27b"
fi

echo "Building payload from $IMAGES ..."
.venv/bin/python - "$IMAGES" <<'PY'
import base64, json, sys
from pathlib import Path
from bruhos.scan.focus import select_for_vl
image_dir = Path(sys.argv[1])
raw = [(p.stem, p.read_bytes()) for p in sorted(image_dir.glob("*.jpg"))]
images, dropped = select_for_vl(raw, min_lap_var=25, min_mean=40, min_score=30, max_images=6)
text = (
    f"These are {len(images)} quality-filtered head-camera views. "
    "List distinct notable objects. Reply JSON only with key objects. Prefer sharper views."
)
parts = [{"type": "text", "text": text}]
for name, jpeg in images:
    parts.append({"type": "text", "text": f"pose={name}"})
    parts.append({
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()},
    })
Path("/tmp/vl_bench_msgs.json").write_text(json.dumps(parts))
Path("/tmp/vl_bench_meta.json").write_text(json.dumps({
    "vl_labels": [n for n, _ in images],
    "dropped": dropped,
    "n_input": len(raw),
}))
print("frames:", [n for n, _ in images])
print("dropped:", dropped)
PY

echo "Models:"
echo "$MODELS"
echo "benchmarking..."

while IFS= read -r M; do
  [[ -z "$M" ]] && continue
  SAFE=${M//\//_}
  SAFE=${SAFE//:/_}
  .venv/bin/python - "$M" <<'PY'
import json, sys
from pathlib import Path
model = sys.argv[1]
parts = json.loads(Path("/tmp/vl_bench_msgs.json").read_text())
payload = {
    "model": model,
    "temperature": 0,
    "messages": [
        {
            "role": "system",
            "content": (
                "You inventory objects. Reply JSON only: {\"objects\":[\"...\"]}. "
                "Short lowercase names. Merge duplicates. Ignore wall/floor/ceiling/robot."
            ),
        },
        {"role": "user", "content": parts},
    ],
}
Path("/tmp/vl_bench_payload.json").write_text(json.dumps(payload))
PY
  START=$(date +%s.%N)
  set +e
  curl -sS --max-time "$TIMEOUT" \
    -H 'Content-Type: application/json' \
    -H 'User-Agent: bruhos-vl-bench/0.1' \
    -d @/tmp/vl_bench_payload.json \
    "$BASE/chat/completions" > "/tmp/vl_resp_${SAFE}.json"
  RC=$?
  set -e
  END=$(date +%s.%N)
  SEC=$(.venv/bin/python -c "print(round(float('$END')-float('$START'), 2))")
  .venv/bin/python - "$M" "$SEC" "$RC" "/tmp/vl_resp_${SAFE}.json" "$ROWS" <<'PY'
import json, sys
from pathlib import Path
from bruhos.scan.vision import parse_objects_payload
model, sec, rc, resp_path, rows_path = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
raw = Path(resp_path).read_text() if Path(resp_path).exists() else ""
err = None
objs = []
if rc != 0:
    err = f"curl exit {rc}: {raw[:200]}"
elif raw.startswith("Blocked") or not raw.strip().startswith("{"):
    err = raw[:200] or "empty response"
else:
    try:
        content = json.loads(raw)["choices"][0]["message"]["content"]
        objs = parse_objects_payload(content)
    except Exception as e:
        err = f"{type(e).__name__}: {e}; body={raw[:160]}"
row = {"model": model, "seconds": sec, "n_objects": len(objs), "objects": objs, "error": err, "ok": err is None}
with open(rows_path, "a") as f:
    f.write(json.dumps(row) + "\n")
status = "ERR" if err else f"{len(objs)} objs"
print(f"  {sec:7.2f}s  {status:10}  {model}", flush=True)
if err:
    print(f"    {err[:160]}", flush=True)
else:
    print(f"    → {', '.join(objs) or '(none)'}", flush=True)
PY
done <<< "$MODELS"

.venv/bin/python - "$OUT" "$BASE" "$IMAGES" <<'PY'
import json, sys
from pathlib import Path
from collections import Counter
out, base, images = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in Path("/tmp/vl_bench_rows.jsonl").read_text().splitlines() if l.strip()]
meta = json.loads(Path("/tmp/vl_bench_meta.json").read_text())
ok = [r for r in rows if r["ok"]]
counts: Counter[str] = Counter()
for r in ok:
    counts.update(r["objects"])
consensus = sorted(o for o, n in counts.items() if ok and n >= max(1, (len(ok) + 1) // 2))
fastest = min(ok, key=lambda r: r["seconds"])["model"] if ok else None
richest = max(ok, key=lambda r: r["n_objects"])["model"] if ok else None
report = {
    "base_url": base,
    "image_dir": images,
    "n_input_images": meta["n_input"],
    "n_vl_images": len(meta["vl_labels"]),
    "vl_labels": meta["vl_labels"],
    "dropped": meta["dropped"],
    "models": rows,
    "consensus_objects": consensus,
    "fastest": fastest,
    "richest": richest,
}
Path(out).write_text(json.dumps(report, indent=2) + "\n")
print()
print("======== SUMMARY ========")
print(f"{'seconds':>8}  {'objs':>4}  model")
for r in sorted(rows, key=lambda x: (not x["ok"], x["seconds"])):
    if r["error"]:
        print(f"{r['seconds']:8.2f}   ERR  {r['model']}")
    else:
        print(f"{r['seconds']:8.2f}  {r['n_objects']:4d}  {r['model']}")
        print(f"          → {', '.join(r['objects']) or '(none)'}")
print(f"fastest: {fastest}")
print(f"richest: {richest}")
print(f"consensus (≥half): {', '.join(consensus) or '(none)'}")
print(f"wrote {out}")
PY
