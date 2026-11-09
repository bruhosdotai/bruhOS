"""Read the published eye files. Never opens the camera."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from bruhos.eye.state import (
    DEFAULT_DIR,
    STALE_CAPTION_S,
    STALE_FRAME_S,
    frame_age_s,
    read_frame_bytes,
    read_state,
)


class EyeCaption:
    """Latest fresh caption from ``state.json``. Duplex uses ``.text`` and ``subscribe``."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        max_age_s: float = STALE_CAPTION_S,
        poll_s: float = 0.5,
    ):
        self.root = Path(root) if root else DEFAULT_DIR
        self.max_age_s = max_age_s
        self.poll_s = poll_s
        self.text = ""
        self._lock = threading.Lock()
        self._listeners: list[Callable[[str], None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="bruhos-eye-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def subscribe(self, fn: Callable[[str], None]) -> Callable[[], None]:
        with self._lock:
            self._listeners.append(fn)

        def unsubscribe() -> None:
            with self._lock:
                if fn in self._listeners:
                    self._listeners.remove(fn)

        return unsubscribe

    def refresh(self) -> str:
        state = read_state(self.root)
        if state is None:
            return self.text
        caption = state.caption_fresh(max_age_s=self.max_age_s)
        if not caption or caption == self.text:
            if not caption and self.text:
                with self._lock:
                    self.text = ""
            return self.text
        with self._lock:
            self.text = caption
            listeners = list(self._listeners)
        print(f"[ear] SEE {caption}", flush=True)
        for fn in listeners:
            try:
                fn(caption)
            except Exception:
                continue
        return self.text

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.refresh()
            self._stop.wait(self.poll_s)


def decode_frame(root: str | Path | None = None, *, max_age_s: float = STALE_FRAME_S):
    """BGR ndarray from a fresh ``frame.jpg``, or None."""
    age = frame_age_s(root)
    if age is None or age > max_age_s:
        return None
    data = read_frame_bytes(root)
    if not data:
        return None
    import cv2
    import numpy as np

    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame
