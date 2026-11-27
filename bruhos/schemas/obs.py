"""Compressed observation: what the Brain sees instead of raw video."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

# Any run of non-alphanumerics separates words: `red_soccer_ball`, `red-soccer-ball`
# and `red soccer ball` all tokenise the same way.
_WORD_SPLIT = re.compile(r"[^a-z0-9]+")
_FILLER_WORDS = frozenset({"the", "one"})


def _words(text: str) -> set[str]:
    """Lowercase word set of `text`, ignoring separators and filler words."""
    return {w for w in _WORD_SPLIT.split(text.lower()) if w and w not in _FILLER_WORDS}


@dataclass
class Detection:
    label: str                      # "block", "box", "ball", "face", ...
    color: str | None = None        # "red", "blue", ...
    bearing_deg: float = 0.0        # + left, - right, relative to heading
    distance_m: float | None = None
    confidence: float = 1.0

    @property
    def name(self) -> str:
        return f"{self.color}_{self.label}" if self.color else self.label

    def matches(self, target: str) -> bool:
        """`red_soccer_ball`, `soccer_ball`, `red`, `ball` all resolve against label/color.

        Both sides are tokenised the same way, so a multi-word label (`soccer_ball`,
        `trash_can`) is reachable by its own reported name and by any part of it,
        while an unknown word or a different color still rejects.
        """
        tokens = _words(target)
        return bool(tokens) and tokens <= _words(self.label) | _words(self.color or "")


@dataclass
class Imu:
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    fallen: bool = False
    fell_forward: bool | None = None   # None when upright


@dataclass
class LineReading:
    visible: bool = False
    offset: float = 0.0                # -1 (far left) .. +1 (far right)


@dataclass
class Observation:
    t: float
    frame: int
    objects: list[Detection] = field(default_factory=list)
    imu: Imu = field(default_factory=Imu)
    line: LineReading = field(default_factory=LineReading)
    holding: str | None = None

    def find(self, target: str) -> Detection | None:
        hits = [d for d in self.objects if d.matches(target)]
        if not hits:
            return None
        return min(hits, key=lambda d: (d.distance_m if d.distance_m is not None else 1e9, abs(d.bearing_deg)))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for o in d["objects"]:
            o["name"] = f"{o['color']}_{o['label']}" if o["color"] else o["label"]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Observation":
        return cls(
            t=d["t"], frame=d["frame"],
            objects=[Detection(**{k: v for k, v in o.items() if k != "name"}) for o in d.get("objects", [])],
            imu=Imu(**d.get("imu", {})),
            line=LineReading(**d.get("line", {})),
            holding=d.get("holding"),
        )
