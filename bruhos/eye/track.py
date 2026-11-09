"""Decide how the head should turn so a moving thing stays in view.

The eye owns this choice. It does not open the servo board. The web process
already has ``/dev/ttyAMA0`` and applies the nudge.
"""

from __future__ import annotations

import json
import ssl
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from bruhos.eye.state import DEFAULT_DIR, eye_dir

DEADZONE = 0.18
MIN_MOTION = 0.04
QUIET_S = 0.8
HOLD_NAME = "head-hold"


def look_nudge(
    flow: dict[str, Any],
    motion: float,
    *,
    deadzone: float = DEADZONE,
    min_motion: float = MIN_MOTION,
) -> str | None:
    """``left|right|up|down`` toward the busiest off-center patch, or None."""
    if motion < min_motion:
        return None
    movers = flow.get("movers") or []
    if not movers:
        return None
    best = max(movers, key=lambda m: abs(float(m.get("x", 0.5)) - 0.5) + abs(float(m.get("y", 0.5)) - 0.5))
    ex = float(best.get("x", 0.5)) - 0.5
    ey = float(best.get("y", 0.5)) - 0.5
    if abs(ex) < deadzone and abs(ey) < deadzone:
        return None
    if abs(ex) >= abs(ey):
        return "right" if ex > 0 else "left"
    return "down" if ey > 0 else "up"


def head_held(root: str | Path | None = None, *, now: float | None = None) -> bool:
    path = eye_dir(root) / HOLD_NAME
    try:
        until = float(path.read_text().strip())
    except (OSError, ValueError):
        return False
    return (now if now is not None else time.time()) < until


def hold_head(seconds: float = 5.0, root: str | Path | None = None) -> None:
    path = eye_dir(root) / HOLD_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(time.time() + seconds))


class HeadFollow:
    """Ask the existing head API to nudge. One command at a time."""

    def __init__(
        self,
        url: str,
        *,
        root: str | Path | None = None,
        send: Callable[[str], None] | None = None,
        quiet_s: float = QUIET_S,
        min_mean: float = 12.0,
    ):
        self.url = url
        self.root = Path(root) if root else DEFAULT_DIR
        self.quiet_s = quiet_s
        self.min_mean = min_mean
        self._send = send or (lambda nudge: _post_nudge(self.url, nudge))
        self._quiet_until = 0.0
        self._lock = threading.Lock()
        self._busy = False

    def consider(self, flow: dict[str, Any], motion: float, mean: float = 255.0) -> str | None:
        if mean < self.min_mean:
            return None
        nudge = look_nudge(flow, motion)
        if not nudge or head_held(self.root):
            return None
        now = time.monotonic()
        with self._lock:
            if self._busy or now < self._quiet_until:
                return None
            self._busy = True
            self._quiet_until = now + self.quiet_s
        print(f"[eye] LOOK {nudge}", flush=True)

        def _go() -> None:
            try:
                self._send(nudge)
            except Exception as e:  # noqa: BLE001
                print(f"[eye] LOOK fail {e}", flush=True)
            finally:
                with self._lock:
                    self._busy = False

        threading.Thread(target=_go, name="bruhos-eye-look", daemon=True).start()
        return nudge


def _post_nudge(url: str, nudge: str) -> None:
    body = json.dumps({"nudge": nudge, "preview": False, "source": "eye"}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=2.0, context=ctx) as resp:
        resp.read()


def follow_from_config(cfg: dict[str, Any]) -> HeadFollow | None:
    eye = cfg.get("eye") if isinstance(cfg.get("eye"), dict) else {}
    if eye.get("follow", True) is False:
        return None
    if not cfg.get("head"):
        return None
    web = cfg.get("web") if isinstance(cfg.get("web"), dict) else {}
    port = int(web.get("port", 8090))
    tls = bool(web.get("tls", False))
    scheme = "https" if tls else "http"
    url = str(eye.get("head_url") or f"{scheme}://127.0.0.1:{port}/api/head")
    return HeadFollow(url, root=eye.get("dir") or DEFAULT_DIR, min_mean=12.0)
