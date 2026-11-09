"""Ear-side vision: read the eye publisher's caption. Never opens the camera."""

from __future__ import annotations

from typing import Any

from bruhos.eye.publisher import motion_fraction
from bruhos.eye.reader import EyeCaption
from bruhos.eye.state import DEFAULT_DIR, EyeState, read_state


def vision_context_item(text: str) -> dict[str, Any]:
    """A system note. No ``response.create``, so it is context for the next reply."""
    return {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "system",
            "content": [{"type": "input_text", "text": "In front of you: " + text}],
        },
    }


def start_caption_reader(cfg: dict[str, Any]) -> EyeCaption | None:
    """Watch ``state.json`` when the camera is enabled. None when it is off."""
    cam = cfg.get("camera")
    if not isinstance(cam, dict) or not cam.get("enabled", True):
        return None
    eye = cfg.get("eye") if isinstance(cfg.get("eye"), dict) else {}
    reader = EyeCaption(root=eye.get("dir") or DEFAULT_DIR)
    reader.start()
    print(f"[ear] SEE reading {reader.root}", flush=True)
    return reader


# Kept for tests that imported motion_fraction from here.
__all__ = [
    "EyeCaption",
    "EyeState",
    "motion_fraction",
    "read_state",
    "start_caption_reader",
    "vision_context_item",
]
