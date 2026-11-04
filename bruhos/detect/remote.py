"""Thin HTTP client for GPU-box ``POST {detect_url}/v1/detect``.

Does **not** run zerollama on the Pi — only a JSON POST when the see/scan
pipeline needs open-vocab boxes.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Callable

from bruhos.detect.types import DetectRequest, DetectResponse

Transport = Callable[[str, dict[str, str], bytes, float], bytes]


def _urllib_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> bytes:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


class RemoteDetectClient:
    """``ZEROLLAMA_DETECT_URL``-style client (base URL without trailing /v1)."""

    def __init__(
        self,
        detect_url: str | None = None,
        *,
        model: str = "detect",
        timeout: float = 60.0,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        transport: Transport | None = None,
        coordinate_space: str = "normalized",
    ):
        base = (
            detect_url
            or os.environ.get("ZEROLLAMA_DETECT_URL")
            or os.environ.get("BRUHOS_DETECT_URL")
            or ""
        ).rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3].rstrip("/")
        if not base:
            raise ValueError("detect_url or ZEROLLAMA_DETECT_URL required for remote detect")
        self.base_url = base
        self.model = model or "detect"
        self.timeout = float(timeout)
        self.api_key = api_key if api_key is not None else os.environ.get(api_key_env, "")
        self.transport = transport or _urllib_transport
        self.coordinate_space = coordinate_space or "normalized"

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/v1/detect"

    def detect(self, req: DetectRequest) -> DetectResponse:
        payload = req.to_json_dict()
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "bruhos-detect/0.1",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        t0 = time.perf_counter()
        try:
            raw = self.transport(
                self.endpoint, headers, json.dumps(payload).encode(), self.timeout,
            )
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"detect HTTP {e.code}: {body}") from e
        data = json.loads(raw)
        resp = DetectResponse.from_dict(data)
        resp.backend = resp.backend or f"remote:{self.base_url}"
        resp.seconds = round(time.perf_counter() - t0, 3)
        if resp.model is None:
            resp.model = payload.get("model") or self.model
        return resp

    def detect_jpeg(
        self,
        jpeg: bytes,
        *,
        queries: list[str],
        threshold: float | None = None,
        max_detections: int | None = None,
        model: str | None = None,
        coordinate_space: str | None = None,
        frame_id: str | None = None,
    ) -> DetectResponse:
        return self.detect(DetectRequest(
            image_jpeg=jpeg,
            queries=list(queries),
            model=model or self.model,
            threshold=threshold,
            max_detections=max_detections,
            coordinate_space=coordinate_space or self.coordinate_space,
            frame_id=frame_id,
        ))
