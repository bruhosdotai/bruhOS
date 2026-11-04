"""Benchmark all vision models on a directory of stills (speed + object lists)."""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from bruhos.scan.focus import select_for_vl
from bruhos.scan.vision import VisionLister

# Fallback vision-capable model ids (image in input_modalities) when /models fails.
_COZMIC_VL_FALLBACK = (
    "gemma4:26b-a4b-it-qat",
    "gemma4:31b",
    "ministral-3:3b",
    "qwen3.6-64k:latest",
    "qwen3.6:27b",
    "qwen3.6:35b",
    "qwen3.8:27b",
)


def curl_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
    """POST via curl (works on hosts where urllib is sandbox-blocked)."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        f.write(body)
        path = f.name
    cmd = [
        "curl", "-sS", "--fail-with-body",
        "--max-time", str(max(1, int(timeout))),
        "-X", "POST", url,
        "-H", "Content-Type: application/json",
        "-H", "User-Agent: bruhos-vl-bench/0.1",
        "--data-binary", f"@{path}",
    ]
    for k, v in headers.items():
        if k.lower() in ("content-type", "user-agent"):
            continue
        cmd.extend(["-H", f"{k}: {v}"])
    try:
        r = subprocess.run(cmd, capture_output=True, check=False)
    finally:
        Path(path).unlink(missing_ok=True)
    if r.returncode != 0:
        err = (r.stderr or r.stdout or b"").decode("utf-8", "replace")[:300]
        raise RuntimeError(f"curl failed ({r.returncode}): {err}")
    return r.stdout


def list_vision_models(base_url: str, timeout: float = 30.0) -> list[str]:
    url = base_url.rstrip("/") + "/models"
    r = subprocess.run(
        ["curl", "-sS", "--max-time", str(int(timeout)), "-H", "User-Agent: bruhos-vl-bench/0.1", url],
        capture_output=True, check=False,
    )
    if r.returncode != 0 or not r.stdout or r.stdout.startswith(b"Blocked"):
        return list(_COZMIC_VL_FALLBACK)
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return list(_COZMIC_VL_FALLBACK)
    out: list[str] = []
    for m in data.get("data") or []:
        mods = m.get("input_modalities") or (m.get("meta") or {}).get("input_modalities") or []
        caps = m.get("capabilities") or []
        mid = m.get("id") or ""
        if not mid:
            continue
        if "image" in mods or "vision" in [str(c).lower() for c in caps]:
            out.append(mid)
    return sorted(out, key=str.lower) if out else list(_COZMIC_VL_FALLBACK)


def load_images(image_dir: Path) -> list[tuple[str, bytes]]:
    paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg"))
    paths += sorted(image_dir.glob("*.png"))
    # Prefer top-level stills; ignore retune/ subdir unless empty
    paths = [p for p in paths if "retune" not in p.parts]
    if not paths:
        paths = sorted(image_dir.rglob("best.jpg"))
    return [(p.stem, p.read_bytes()) for p in paths]


def bench_models(
    *,
    base_url: str,
    image_dir: Path,
    models: list[str] | None = None,
    timeout: float = 180.0,
    quality_gate: bool = True,
    api_key: str = "",
) -> dict[str, Any]:
    raw = load_images(image_dir)
    if not raw:
        raise FileNotFoundError(f"no images in {image_dir}")
    if quality_gate:
        images, dropped = select_for_vl(raw, min_lap_var=25, min_mean=40, min_score=30, max_images=6)
    else:
        images, dropped = raw, []
    if models is None:
        models = list_vision_models(base_url, timeout=min(30.0, timeout))
    if not models:
        raise RuntimeError(f"no vision models at {base_url}")

    rows: list[dict[str, Any]] = []
    for model in models:
        vl = VisionLister(
            base_url=base_url, model=model, api_key=api_key, timeout=timeout,
            transport=curl_transport,
        )
        t0 = time.perf_counter()
        err = None
        objects: list[str] = []
        try:
            objects = vl.list_objects(images)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        elapsed = round(time.perf_counter() - t0, 2)
        rows.append({
            "model": model,
            "seconds": elapsed,
            "n_objects": len(objects),
            "objects": objects,
            "error": err,
            "ok": err is None,
        })
        status = "ERR" if err else f"{len(objects)} objs"
        print(f"  {elapsed:7.2f}s  {status:10}  {model}", flush=True)

    ok_rows = [r for r in rows if r["ok"]]
    # Agreement: objects appearing in ≥ half of successful models
    from collections import Counter
    counts: Counter[str] = Counter()
    for r in ok_rows:
        counts.update(r["objects"])
    consensus = sorted(o for o, n in counts.items() if ok_rows and n >= max(1, (len(ok_rows) + 1) // 2))
    fastest = min(ok_rows, key=lambda r: r["seconds"]) if ok_rows else None
    richest = max(ok_rows, key=lambda r: r["n_objects"]) if ok_rows else None

    return {
        "base_url": base_url,
        "image_dir": str(image_dir),
        "n_input_images": len(raw),
        "n_vl_images": len(images),
        "vl_labels": [n for n, _ in images],
        "dropped": dropped,
        "models": rows,
        "consensus_objects": consensus,
        "fastest": fastest["model"] if fastest else None,
        "richest": richest["model"] if richest else None,
    }


def format_bench_table(report: dict[str, Any]) -> str:
    lines = [
        f"VL BENCH  {report['base_url']}  images={report['n_vl_images']}/{report['n_input_images']}",
        f"frames: {', '.join(report.get('vl_labels') or [])}",
        "-" * 72,
        f"{'seconds':>8}  {'objs':>4}  model",
    ]
    for r in sorted(report["models"], key=lambda x: (not x["ok"], x["seconds"])):
        if r["error"]:
            lines.append(f"{r['seconds']:8.2f}  {'ERR':>4}  {r['model']}  ({r['error'][:60]})")
        else:
            lines.append(f"{r['seconds']:8.2f}  {r['n_objects']:4d}  {r['model']}")
            lines.append(f"{'':8}  {''}  → {', '.join(r['objects']) or '(none)'}")
    lines.append("-" * 72)
    lines.append(f"fastest: {report.get('fastest')}")
    lines.append(f"richest: {report.get('richest')}")
    lines.append(f"consensus (≥half): {', '.join(report.get('consensus_objects') or []) or '(none)'}")
    return "\n".join(lines)
