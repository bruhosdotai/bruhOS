"""Compressed observation: what the Brain sees instead of raw video."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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
        """`red_block`, `red`, `block`, `box` all resolve against label/color."""
        tokens = {t for t in target.lower().replace("-", "_").split("_") if t and t not in {"the", "one"}}
        have = {self.label.lower()} | ({self.color.lower()} if self.color else set())
        return bool(tokens) and tokens <= have


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
