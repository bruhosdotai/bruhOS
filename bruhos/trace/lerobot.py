"""LeRobot-aligned frame export.

v0.1 aligns field names and shapes with the LeRobot dataset convention
(``observation.state``, ``action``, ``timestamp``, ``frame_index``, ``episode_index``,
``index``, ``task``, ``next.done``, ``next.success``). Writing an actual ``LeRobotDataset``
(parquet + meta + videos) is v0.2 and will live behind the optional ``lerobot`` dependency.
"""

from __future__ import annotations

from typing import Any

from bruhos.schemas import Episode
from bruhos.skills import SKILLS

SKILL_INDEX = {name: i for i, name in enumerate(sorted(SKILLS))}

STATE_NAMES = ["imu.roll_deg", "imu.pitch_deg", "imu.fallen", "holding", "n_objects", "line.visible", "line.offset"]
ACTION_NAMES = ["skill_id", "status_ok"]

LEROBOT_FEATURES: dict[str, dict[str, Any]] = {
    "observation.state": {"dtype": "float32", "shape": [len(STATE_NAMES)], "names": STATE_NAMES},
    "action": {"dtype": "float32", "shape": [len(ACTION_NAMES)], "names": ACTION_NAMES},
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "next.done": {"dtype": "bool", "shape": [1], "names": None},
    "next.success": {"dtype": "bool", "shape": [1], "names": None},
}


def _state(obs: dict[str, Any]) -> list[float]:
    imu, line = obs.get("imu", {}), obs.get("line", {})
    return [
        float(imu.get("roll_deg", 0.0)), float(imu.get("pitch_deg", 0.0)), float(bool(imu.get("fallen"))),
        float(obs.get("holding") is not None), float(len(obs.get("objects", []))),
        float(bool(line.get("visible"))), float(line.get("offset", 0.0)),
    ]


def to_lerobot_frames(ep: Episode, episode_index: int = 0, start_index: int = 0) -> list[dict[str, Any]]:
    steps = [s for s in ep.steps if s.kind in {"skill", "recovery", "stop"}]
    t0 = steps[0].t if steps else 0.0
    frames = []
    for i, s in enumerate(steps):
        last = i == len(steps) - 1
        frames.append({
            "observation.state": _state(s.obs_before),
            "action": [float(SKILL_INDEX.get(s.call["skill"], -1)), float(s.outcome["status"] == "ok")],
            "timestamp": round(s.t - t0, 4),
            "frame_index": i,
            "episode_index": episode_index,
            "index": start_index + i,
            "task": ep.instruction,
            "next.done": last,
            "next.success": bool(last and ep.success),
            "bruhos.call": s.call,
            "bruhos.step_hash": s.hash,
        })
    return frames
