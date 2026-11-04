"""Grounded detection experiments: name + where (bbox → bearing/elevation).

Backends:
  - vl-boxes: remote VL prompted for JSON boxes (cozmic / zerollama)
  - yolo-world: ultralytics YOLO-World open-vocab (optional dep)
  - florence: Microsoft Florence-2 grounding tasks (optional dep)
"""

from __future__ import annotations

import base64
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from bruhos.body.perception import CameraModel
from bruhos.scan.bench import curl_transport, load_images
from bruhos.scan.focus import select_for_vl
from bruhos.scan.vision import _FENCE, normalize_objects

Transport = Callable[[str, dict[str, str], bytes, float], bytes]

# Consensus nouns from the VL bench — good default open-vocab prompt.
DEFAULT_CLASSES = (
    "cabinet", "shelf", "poster", "door", "tv", "book", "plant",
    "chair", "bed", "lamp", "pillow", "box", "picture", "bottle",
)

VL_BOX_SYSTEM = (
    "You locate objects in a single camera image. "
    "Reply with JSON only: "
    '{"objects":[{"name":"shelf","bbox":[x1,y1,x2,y2],"conf":0.8}]}. '
    "bbox is normalized 0..1 as [left, top, right, bottom] relative to image width/height. "
    "Use short lowercase names. Ignore wall, floor, ceiling, sky, and the robot body. "
    "If nothing notable, return {\"objects\":[]}."
)


@dataclass
class GroundedDet:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0
    bearing_deg: float = 0.0
    elev_deg: float = 0.0
    normalized: bool = False  # True if xyxy are 0..1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrameResult:
    image: str
    backend: str
    seconds: float
    width: int
    height: int
    detections: list[GroundedDet] = field(default_factory=list)
    error: str | None = None
    raw_preview: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "backend": self.backend,
            "seconds": round(self.seconds, 3),
            "width": self.width,
            "height": self.height,
            "n_detections": len(self.detections),
            "detections": [d.to_dict() for d in self.detections],
            "error": self.error,
            "raw_preview": self.raw_preview,
            "ok": self.ok,
        }


def vfov_deg(cam: CameraModel) -> float:
    """Vertical FOV from horizontal FOV and aspect ratio."""
    return 2.0 * math.degrees(
        math.atan(math.tan(math.radians(cam.hfov_deg / 2.0)) * (cam.height / cam.width))
    )


def elevation(cam: CameraModel, cy: float) -> float:
    """+ up / - down, degrees from optical axis."""
    fy = (cam.height / 2.0) / math.tan(math.radians(vfov_deg(cam) / 2.0))
    return round(math.degrees(math.atan((cam.height / 2.0 - cy) / fy)), 2)


def enrich(dets: list[GroundedDet], cam: CameraModel) -> list[GroundedDet]:
    """Convert normalized or pixel boxes → pixel + bearing/elev."""
    out: list[GroundedDet] = []
    for d in dets:
        x1, y1, x2, y2 = d.x1, d.y1, d.x2, d.y2
        if d.normalized or (0.0 <= x1 <= 1.5 and 0.0 <= x2 <= 1.5 and max(x1, x2) <= 1.5):
            # treat as normalized if values look like 0..1
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
                x1, x2 = x1 * cam.width, x2 * cam.width
                y1, y2 = y1 * cam.height, y2 * cam.height
        x1, x2 = sorted((float(x1), float(x2)))
        y1, y2 = sorted((float(y1), float(y2)))
        x1 = max(0.0, min(float(cam.width), x1))
        x2 = max(0.0, min(float(cam.width), x2))
        y1 = max(0.0, min(float(cam.height), y1))
        y2 = max(0.0, min(float(cam.height), y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        out.append(GroundedDet(
            label=d.label.casefold().strip(),
            x1=round(x1, 1), y1=round(y1, 1), x2=round(x2, 1), y2=round(y2, 1),
            conf=float(d.conf),
            bearing_deg=round(cam.bearing(cx), 2),
            elev_deg=elevation(cam, cy),
            normalized=False,
        ))
    return out


def parse_grounded_payload(content: str, *, default_norm: bool = True) -> list[GroundedDet]:
    """Parse VL / Florence-style JSON into GroundedDet list (coords not yet enriched)."""
    text = content.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return []
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []
    items: list[Any]
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("objects") or data.get("detections") or data.get("items") or []
        if not isinstance(items, list):
            return []
    else:
        return []
    out: list[GroundedDet] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        label = raw.get("name") or raw.get("label") or raw.get("object") or ""
        if not isinstance(label, str) or not label.strip():
            continue
        bbox = raw.get("bbox") or raw.get("box") or raw.get("xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        try:
            x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        except (TypeError, ValueError):
            continue
        conf = raw.get("conf", raw.get("confidence", raw.get("score", 1.0)))
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            conf_f = 1.0
        out.append(GroundedDet(
            label=label, x1=x1, y1=y1, x2=x2, y2=y2, conf=conf_f, normalized=default_norm,
        ))
    return out


def _draw_label(img, text: str, x: int, y: int, *, scale: float = 0.55) -> None:
    """High-contrast label: dark filled pill + white text (OpenCV BGR)."""
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad_x, pad_y = 6, 4
    # Prefer above the box; flip below if clipped at top.
    text_y = y - pad_y
    if text_y - th - pad_y < 0:
        text_y = y + th + pad_y + 2
    x1 = max(0, x)
    y1 = max(0, text_y - th - pad_y)
    x2 = min(img.shape[1] - 1, x1 + tw + pad_x * 2)
    y2 = min(img.shape[0] - 1, text_y + pad_y + baseline)
    cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 80), 1)
    cv2.putText(
        img, text, (x1 + pad_x, text_y), font, scale,
        (255, 255, 255), thickness, cv2.LINE_AA,
    )


def draw_overlay(jpeg: bytes, dets: list[GroundedDet], title: str = "") -> bytes:
    import cv2
    import numpy as np

    arr = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("bad jpeg")
    for d in dets:
        p1 = (int(d.x1), int(d.y1))
        p2 = (int(d.x2), int(d.y2))
        cv2.rectangle(img, p1, p2, (0, 220, 80), 2)
        tag = f"{d.label}  {d.bearing_deg:+.0f}°/{d.elev_deg:+.0f}°"
        _draw_label(img, tag, p1[0], p1[1], scale=0.55)
    if title:
        _draw_label(img, title, 8, 22, scale=0.6)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("encode failed")
    return bytes(buf)


def _image_size(jpeg: bytes) -> tuple[int, int]:
    import cv2
    import numpy as np

    arr = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("bad jpeg")
    h, w = img.shape[:2]
    return w, h


# ----- backends -----


def run_gpu_detect(
    *,
    label: str,
    jpeg: bytes,
    client: Any,
    queries: list[str],
    cam: CameraModel | None = None,
    threshold: float | None = None,
    max_detections: int | None = None,
    model: str | None = None,
    coordinate_space: str = "normalized",
    frame_id: str | None = None,
) -> FrameResult:
    """GPU ``POST /v1/detect`` → enriched ``GroundedDet`` list.

    ``status=wire_only`` is returned as ``error`` so callers can fall back to VL
    (stub empty must not be treated as a real miss).
    """
    t0 = time.perf_counter()
    w0, h0 = _image_size(jpeg)
    cam = cam or CameraModel(width=w0, height=h0)
    q = [str(x).strip() for x in queries if str(x).strip()]
    if not q:
        q = list(DEFAULT_CLASSES)
    try:
        resp = client.detect_jpeg(
            jpeg,
            queries=q,
            threshold=threshold,
            max_detections=max_detections,
            model=model,
            coordinate_space=coordinate_space,
            frame_id=frame_id or label,
        )
    except Exception as e:  # noqa: BLE001
        return FrameResult(
            image=label, backend="detect", seconds=time.perf_counter() - t0,
            width=w0, height=h0, error=str(e)[:400],
        )
    backend = f"detect:{resp.backend or resp.model or 'remote'}"
    if resp.wire_only:
        return FrameResult(
            image=label, backend=backend, seconds=time.perf_counter() - t0,
            width=w0, height=h0,
            error="detect status=wire_only (stub — not a real miss)",
            raw_preview=json.dumps({"status": resp.status, "backend": resp.backend})[:240],
        )
    space = (resp.coordinate_space or coordinate_space or "normalized").casefold()
    norm = space == "normalized"
    raw_dets = [
        GroundedDet(
            label=b.label, x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2,
            conf=b.confidence, normalized=norm or b.normalized,
        )
        for b in resp.detections
    ]
    cam_full = CameraModel(
        width=int(resp.width or w0),
        height=int(resp.height or h0),
        hfov_deg=cam.hfov_deg,
    )
    dets = enrich(raw_dets, cam_full)
    return FrameResult(
        image=label, backend=backend,
        seconds=resp.seconds if resp.seconds is not None else time.perf_counter() - t0,
        width=cam_full.width, height=cam_full.height, detections=dets,
        raw_preview=json.dumps({
            "status": resp.status, "backend": resp.backend,
            "n": len(dets), "queries": q[:12],
        })[:240],
    )


def ground_image(
    *,
    label: str,
    jpeg: bytes,
    cam: CameraModel,
    cfg: dict[str, Any],
    hint_objects: list[str] | None = None,
    transport: Transport | None = None,
    detect_client: Any | None = None,
    vl_fn: Callable[..., FrameResult] | None = None,
) -> FrameResult:
    """See/scan grounder: prefer GPU ``/v1/detect``, fall back to VL boxes.

    Fallback when: no detect URL, detect error, ``wire_only``, or real empty
    boxes while inventory still named objects (quality over speed).
    """
    from bruhos.detect import make_remote_client

    scan_cfg = cfg.get("scan", {})
    det_cfg = cfg.get("detect", {})
    brain = cfg.get("brain", {})
    hints = [h for h in (hint_objects or []) if isinstance(h, str) and h.strip()]
    mode = str(det_cfg.get("ground") or det_cfg.get("ground_backend") or "auto").casefold()
    # auto | detect | vl
    client = detect_client
    if client is None and mode != "vl":
        client = make_remote_client(cfg)

    fr: FrameResult | None = None
    if client is not None and mode != "vl":
        thr = det_cfg.get("threshold", det_cfg.get("confidence"))
        fr = run_gpu_detect(
            label=label, jpeg=jpeg, client=client,
            queries=hints or list(DEFAULT_CLASSES),
            cam=cam,
            threshold=float(thr) if thr is not None else None,
            max_detections=det_cfg.get("max_detections"),
            model=det_cfg.get("remote_model") or det_cfg.get("model") or "detect",
            coordinate_space=str(det_cfg.get("coordinate_space", "normalized")),
            frame_id=label,
        )
        if fr.ok and fr.detections:
            return fr
        if fr.ok and not fr.detections and not hints and mode == "detect":
            return fr
        # else fall through to VL (wire_only / error / empty+hints / auto)

    if mode == "detect" and fr is not None and not fr.ok:
        return fr  # caller asked detect-only

    base_url = scan_cfg.get("base_url") or brain.get("base_url")
    if not base_url:
        return fr or FrameResult(
            image=label, backend="none", seconds=0.0,
            width=cam.width, height=cam.height,
            error="no detect client and no [scan].base_url for VL fallback",
        )
    ground_model = str(scan_cfg.get("ground_model") or scan_cfg.get("model") or "ministral-3:3b")
    vl = vl_fn or run_vl_boxes
    vl_fr = vl(
        label=label, jpeg=jpeg, base_url=base_url, model=ground_model,
        timeout=float(scan_cfg.get("timeout", 180.0)),
        api_key=scan_cfg.get("api_key") or brain.get("api_key") or "",
        cam=cam, transport=transport, hint_objects=hints,
    )
    if fr is not None and not vl_fr.ok and fr.ok:
        return fr  # keep empty detect miss over VL failure
    if fr is not None and fr.error and vl_fr.ok:
        vl_fr.raw_preview = (
            f"(detect fallback: {fr.error[:80]}) " + (vl_fr.raw_preview or "")
        )[:240]
    return vl_fr


def run_vl_boxes(
    *,
    label: str,
    jpeg: bytes,
    base_url: str,
    model: str,
    timeout: float = 180.0,
    api_key: str = "",
    transport: Transport | None = None,
    cam: CameraModel | None = None,
    max_side: int = 640,
    hint_objects: list[str] | None = None,
) -> FrameResult:
    """Ask a remote VL model for normalized bboxes."""
    import cv2
    import numpy as np

    t0 = time.perf_counter()
    transport = transport or curl_transport
    w0, h0 = _image_size(jpeg)
    # shrink large stills for faster VL
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert img is not None
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        jpeg_send = bytes(buf) if ok else jpeg
        w, h = img.shape[1], img.shape[0]
    else:
        jpeg_send = jpeg
    cam = cam or CameraModel(width=w0, height=h0)
    # enrich in original pixel space: scale boxes back after parse
    b64 = base64.b64encode(jpeg_send).decode("ascii")
    hints = [h for h in (hint_objects or []) if isinstance(h, str) and h.strip()]
    if hints:
        hint_txt = (
            f"Image {label} ({w}x{h} px). Draw a bbox for each of these if visible: "
            + ", ".join(hints)
            + ". Also include any other notable objects. "
            "bbox must be normalized 0..1 [left, top, right, bottom] "
            f"(or pixel coords in 0..{w-1} / 0..{h-1}). JSON only."
        )
    else:
        hint_txt = (
            f"Image {label} ({w}x{h} px). List notable objects with bboxes. "
            "bbox must be normalized 0..1 [left, top, right, bottom] "
            f"(or pixel coords in 0..{w-1} / 0..{h-1}). JSON only."
        )
    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": VL_BOX_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": hint_txt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            },
        ],
    }
    headers = {"Content-Type": "application/json", "User-Agent": "bruhos-ground-exp/0.1"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = base_url.rstrip("/") + "/chat/completions"
    try:
        raw = transport(url, headers, json.dumps(payload).encode(), timeout)
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
        dets = parse_grounded_payload(content, default_norm=True)
        # boxes were on (possibly) resized image; map to original cam size
        if w != w0 or h != h0:
            sx, sy = w0 / w, h0 / h
            mapped = []
            for d in dets:
                if max(abs(d.x1), abs(d.y1), abs(d.x2), abs(d.y2)) <= 1.5:
                    mapped.append(d)  # already normalized — enrich uses cam size
                else:
                    mapped.append(GroundedDet(
                        d.label, d.x1 * sx, d.y1 * sy, d.x2 * sx, d.y2 * sy, d.conf, normalized=False,
                    ))
            dets = mapped
        cam_full = CameraModel(width=w0, height=h0, hfov_deg=cam.hfov_deg)
        dets = enrich(dets, cam_full)
        return FrameResult(
            image=label, backend=f"vl-boxes:{model}", seconds=time.perf_counter() - t0,
            width=w0, height=h0, detections=dets,
            raw_preview=(content or "")[:240],
        )
    except Exception as e:  # noqa: BLE001 — experiment harness
        return FrameResult(
            image=label, backend=f"vl-boxes:{model}", seconds=time.perf_counter() - t0,
            width=w0, height=h0, error=str(e)[:400],
        )


def run_yolo_world(
    *,
    label: str,
    jpeg: bytes,
    classes: list[str] | None = None,
    weights: str = "yolov8s-world.pt",
    conf: float = 0.15,
    cam: CameraModel | None = None,
) -> FrameResult:
    t0 = time.perf_counter()
    w0, h0 = _image_size(jpeg)
    cam = cam or CameraModel(width=w0, height=h0)
    classes = classes or list(DEFAULT_CLASSES)
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLOWorld
    except ImportError as e:
        return FrameResult(
            image=label, backend="yolo-world", seconds=time.perf_counter() - t0,
            width=w0, height=h0, error=f"ultralytics not installed: {e}",
        )
    try:
        model = YOLOWorld(weights)
        model.set_classes(classes)
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        results = model.predict(img, conf=conf, verbose=False)
        dets: list[GroundedDet] = []
        if results:
            r0 = results[0]
            names = r0.names or {}
            if r0.boxes is not None:
                for b in r0.boxes:
                    xyxy = b.xyxy[0].tolist()
                    cls_id = int(b.cls[0].item()) if b.cls is not None else 0
                    name = names.get(cls_id, classes[cls_id] if cls_id < len(classes) else str(cls_id))
                    score = float(b.conf[0].item()) if b.conf is not None else 1.0
                    dets.append(GroundedDet(
                        label=str(name), x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3],
                        conf=score, normalized=False,
                    ))
        cam_full = CameraModel(width=w0, height=h0, hfov_deg=cam.hfov_deg)
        dets = enrich(dets, cam_full)
        return FrameResult(
            image=label, backend=f"yolo-world:{weights}", seconds=time.perf_counter() - t0,
            width=w0, height=h0, detections=dets,
        )
    except Exception as e:  # noqa: BLE001
        # fallback: plain YOLO if World class missing
        try:
            from ultralytics import YOLO as Y
            import cv2
            import numpy as np

            model = Y(weights)
            if hasattr(model, "set_classes"):
                model.set_classes(classes)
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            results = model.predict(img, conf=conf, verbose=False)
            dets = []
            if results and results[0].boxes is not None:
                r0 = results[0]
                names = r0.names or {}
                for b in r0.boxes:
                    xyxy = b.xyxy[0].tolist()
                    cls_id = int(b.cls[0].item()) if b.cls is not None else 0
                    name = names.get(cls_id, str(cls_id))
                    score = float(b.conf[0].item()) if b.conf is not None else 1.0
                    dets.append(GroundedDet(
                        label=str(name), x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3],
                        conf=score, normalized=False,
                    ))
            dets = enrich(dets, CameraModel(width=w0, height=h0, hfov_deg=cam.hfov_deg))
            return FrameResult(
                image=label, backend=f"yolo:{weights}", seconds=time.perf_counter() - t0,
                width=w0, height=h0, detections=dets,
            )
        except Exception as e2:  # noqa: BLE001
            return FrameResult(
                image=label, backend="yolo-world", seconds=time.perf_counter() - t0,
                width=w0, height=h0, error=f"{e}; fallback: {e2}"[:400],
            )


_FLORENCE_TASKS = {
    "od": "<OD>",
    "dense": "<DENSE_REGION_CAPTION>",
    "ground": "<CAPTION_TO_PHRASE_GROUNDING>",
    "open": "<OPEN_VOCABULARY_DETECTION>",
}


def run_florence(
    *,
    label: str,
    jpeg: bytes,
    task: str = "od",
    model_id: str = "microsoft/Florence-2-base",
    text_input: str | None = None,
    classes: list[str] | None = None,
    cam: CameraModel | None = None,
) -> FrameResult:
    t0 = time.perf_counter()
    w0, h0 = _image_size(jpeg)
    cam = cam or CameraModel(width=w0, height=h0)
    task_token = _FLORENCE_TASKS.get(task, task)
    try:
        import io

        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as e:
        return FrameResult(
            image=label, backend="florence", seconds=time.perf_counter() - t0,
            width=w0, height=h0, error=f"transformers/torch/PIL not installed: {e}",
        )
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True,
        ).to(device)
        image = Image.open(io.BytesIO(jpeg)).convert("RGB")
        if task_token in ("<CAPTION_TO_PHRASE_GROUNDING>", "<OPEN_VOCABULARY_DETECTION>"):
            text_input = text_input or ". ".join(classes or DEFAULT_CLASSES)
            prompt = task_token + text_input
        else:
            prompt = task_token
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
        if device == "cuda":
            inputs = {k: v.to(dtype=dtype) if v.dtype.is_floating_point else v for k, v in inputs.items()}
        generated = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
        text = processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            text, task=task_token, image_size=(image.width, image.height),
        )
        block = parsed.get(task_token) or {}
        dets: list[GroundedDet] = []
        bboxes = block.get("bboxes") or block.get("bboxes_labels") and block.get("bboxes")
        labels = block.get("labels") or block.get("bboxes_labels") or []
        if isinstance(block.get("bboxes"), list):
            bboxes = block["bboxes"]
            labels = block.get("labels") or [""] * len(bboxes)
            for bb, lab in zip(bboxes, labels):
                if not isinstance(bb, (list, tuple)) or len(bb) < 4:
                    continue
                dets.append(GroundedDet(
                    label=str(lab or "object"), x1=float(bb[0]), y1=float(bb[1]),
                    x2=float(bb[2]), y2=float(bb[3]), conf=1.0, normalized=False,
                ))
        dets = enrich(dets, CameraModel(width=w0, height=h0, hfov_deg=cam.hfov_deg))
        return FrameResult(
            image=label, backend=f"florence:{model_id}:{task}", seconds=time.perf_counter() - t0,
            width=w0, height=h0, detections=dets, raw_preview=str(block)[:240],
        )
    except Exception as e:  # noqa: BLE001
        return FrameResult(
            image=label, backend="florence", seconds=time.perf_counter() - t0,
            width=w0, height=h0, error=str(e)[:400],
        )


def run_experiments(
    *,
    image_dir: Path,
    out_dir: Path,
    backends: list[str],
    base_url: str | None = None,
    vl_model: str = "ministral-3:3b",
    classes: list[str] | None = None,
    quality_gate: bool = True,
    hfov_deg: float = 62.0,
    timeout: float = 180.0,
    api_key: str = "",
    yolo_weights: str = "yolov8s-world.pt",
    florence_model: str = "microsoft/Florence-2-base",
    florence_task: str = "od",
) -> dict[str, Any]:
    """Run selected backends on quality-gated (or all) stills; write overlays + JSON."""
    raw = load_images(image_dir)
    if not raw:
        raise FileNotFoundError(f"no images in {image_dir}")
    if quality_gate and len(raw) > 1:
        kept, _dropped = select_for_vl(
            list(raw),
            max_images=3,
            min_lap_var=25.0,
            min_mean=40.0,
            min_score=30.0,
        )
        # select_for_vl annotates labels ("L2 (score=…)"); keep bare pose stem for files.
        frames = []
        for ann, jpeg in kept:
            stem = ann.split()[0] if ann else "frame"
            frames.append((stem, jpeg))
        if not frames:
            frames = raw[:3]
    else:
        frames = raw

    out_dir.mkdir(parents=True, exist_ok=True)
    classes = classes or list(DEFAULT_CLASSES)
    all_rows: list[dict[str, Any]] = []

    for backend in backends:
        bdir = out_dir / re.sub(r"[^\w.\-+]+", "_", backend)
        bdir.mkdir(parents=True, exist_ok=True)
        for name, jpeg in frames:
            w, h = _image_size(jpeg)
            cam = CameraModel(width=w, height=h, hfov_deg=hfov_deg)
            if backend == "vl-boxes":
                if not base_url:
                    fr = FrameResult(
                        image=name, backend="vl-boxes", seconds=0, width=w, height=h,
                        error="base_url required for vl-boxes",
                    )
                else:
                    fr = run_vl_boxes(
                        label=name, jpeg=jpeg, base_url=base_url, model=vl_model,
                        timeout=timeout, api_key=api_key, cam=cam,
                    )
            elif backend == "yolo-world":
                fr = run_yolo_world(
                    label=name, jpeg=jpeg, classes=classes, weights=yolo_weights, cam=cam,
                )
            elif backend == "florence":
                fr = run_florence(
                    label=name, jpeg=jpeg, task=florence_task, model_id=florence_model,
                    classes=classes, cam=cam,
                )
            else:
                fr = FrameResult(
                    image=name, backend=backend, seconds=0, width=w, height=h,
                    error=f"unknown backend {backend!r}",
                )
            row = fr.to_dict()
            all_rows.append(row)
            if fr.ok:
                overlay = draw_overlay(jpeg, fr.detections, title=f"{backend} {name}")
                (bdir / f"{name}.jpg").write_bytes(overlay)
            (bdir / f"{name}.json").write_text(json.dumps(row, indent=2) + "\n")

    report = {
        "image_dir": str(image_dir),
        "out_dir": str(out_dir),
        "backends": backends,
        "n_frames": len(frames),
        "frame_labels": [n for n, _ in frames],
        "classes": classes,
        "rows": all_rows,
        "summary": _summarize(all_rows),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from collections import defaultdict

    by_b: dict[str, list] = defaultdict(list)
    for r in rows:
        by_b[r["backend"]].append(r)
    out = []
    for backend, rs in sorted(by_b.items()):
        ok = [r for r in rs if r.get("ok")]
        labels: list[str] = []
        for r in ok:
            labels.extend(d["label"] for d in r.get("detections") or [])
        uniq = normalize_objects(labels)
        out.append({
            "backend": backend,
            "ok_frames": len(ok),
            "frames": len(rs),
            "mean_seconds": round(sum(r["seconds"] for r in rs) / max(1, len(rs)), 3),
            "total_dets": sum(r.get("n_detections", 0) for r in ok),
            "unique_labels": uniq,
            "errors": [r["error"] for r in rs if r.get("error")],
        })
    return out


def format_report_table(report: dict[str, Any]) -> str:
    lines = [
        f"frames: {', '.join(report.get('frame_labels') or [])}",
        f"{'backend':<40} {'ok':>5} {'sec':>7} {'dets':>5}  labels",
        "-" * 80,
    ]
    for s in report.get("summary") or []:
        labs = ", ".join(s.get("unique_labels") or [])[:48]
        lines.append(
            f"{s['backend']:<40} {s['ok_frames']:>2}/{s['frames']:<2} "
            f"{s['mean_seconds']:>7.2f} {s['total_dets']:>5}  {labs}"
        )
    return "\n".join(lines)
