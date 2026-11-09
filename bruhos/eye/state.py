"""Published eye files: frame.jpg + state.json on a RAM disk."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

def _movers(raw: Any) -> list[dict[str, float]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        out.append({
            "x": float(item.get("x") or 0.0),
            "y": float(item.get("y") or 0.0),
            "dx": float(item.get("dx") or 0.0),
            "dy": float(item.get("dy") or 0.0),
        })
    return out


DEFAULT_DIR = Path("/dev/shm/bruhos-eye")
FRAME_NAME = "frame.jpg"
STATE_NAME = "state.json"
STALE_CAPTION_S = 20.0
STALE_FRAME_S = 1.0


@dataclass
class EyeState:
    frame_seq: int = 0
    understood_seq: int = 0
    caption: str = ""
    objects: list[str] = field(default_factory=list)
    motion: float = 0.0
    mean: float = 0.0
    direction: str = "still"
    flow_x: float = 0.0
    flow_y: float = 0.0
    movers: list[dict[str, float]] = field(default_factory=list)
    updated: float = 0.0
    model: str = ""
    error: str = ""
    focus: str = ""
    survey_changed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EyeState:
        objs = data.get("objects") or []
        if not isinstance(objs, list):
            objs = []
        return cls(
            frame_seq=int(data.get("frame_seq") or 0),
            understood_seq=int(data.get("understood_seq") or 0),
            caption=str(data.get("caption") or ""),
            objects=[str(x) for x in objs if str(x).strip()],
            motion=float(data.get("motion") or 0.0),
            mean=float(data.get("mean") or 0.0),
            direction=str(data.get("direction") or "still"),
            flow_x=float(data.get("flow_x") or 0.0),
            flow_y=float(data.get("flow_y") or 0.0),
            movers=_movers(data.get("movers")),
            updated=float(data.get("updated") or 0.0),
            model=str(data.get("model") or ""),
            error=str(data.get("error") or ""),
            focus=str(data.get("focus") or ""),
            survey_changed=[str(x) for x in (data.get("survey_changed") or []) if str(x).strip()],
        )

    def caption_fresh(self, *, now: float | None = None, max_age_s: float = STALE_CAPTION_S) -> str:
        text = " ".join(self.caption.split())
        if not text:
            return ""
        stamp = float(self.updated or 0.0)
        if stamp <= 0:
            return ""
        age = (now if now is not None else time.time()) - stamp
        if age > max_age_s:
            return ""
        return text


def eye_dir(path: str | Path | None = None) -> Path:
    return Path(path) if path else DEFAULT_DIR


def frame_path(root: str | Path | None = None) -> Path:
    return eye_dir(root) / FRAME_NAME


def state_path(root: str | Path | None = None) -> Path:
    return eye_dir(root) / STATE_NAME


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    atomic_write_bytes(path, (json.dumps(obj, indent=2) + "\n").encode())


def write_frame(jpeg: bytes, root: str | Path | None = None) -> Path:
    path = frame_path(root)
    atomic_write_bytes(path, jpeg)
    return path


def write_state(state: EyeState, root: str | Path | None = None) -> Path:
    path = state_path(root)
    atomic_write_json(path, state.to_dict())
    return path


def read_state(root: str | Path | None = None) -> EyeState | None:
    path = state_path(root)
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return None
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return EyeState.from_dict(data)


def read_frame_bytes(root: str | Path | None = None) -> bytes | None:
    path = frame_path(root)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError:
        return None
    return data or None


def frame_age_s(root: str | Path | None = None, *, now: float | None = None) -> float | None:
    path = frame_path(root)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return (now if now is not None else time.time()) - mtime
