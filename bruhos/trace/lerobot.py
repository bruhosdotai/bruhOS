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


def _num(value: Any, default: float = 0.0) -> float:
    """A numeric field, or ``default`` when it holds the wrong JSON type.

    Trace files are operator-supplied input: a step field may be a string
    (or any other type) where a number is expected. Coercing only real
    numbers keeps a wrong-typed field from crashing the export; bools are
    not numbers here (``float(True)`` would silently become 1.0).
    """
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _state(obs: dict[str, Any]) -> list[float]:
    obs = _mapping(obs)
    imu, line = _mapping(obs.get("imu")), _mapping(obs.get("line"))
    objects = obs.get("objects")
    return [
        _num(imu.get("roll_deg")), _num(imu.get("pitch_deg")), float(bool(imu.get("fallen"))),
        float(obs.get("holding") is not None), float(len(objects) if isinstance(objects, list) else 0),
        float(bool(line.get("visible"))), _num(line.get("offset")),
    ]


def to_lerobot_frames(ep: Episode, episode_index: int = 0, start_index: int = 0) -> list[dict[str, Any]]:
    steps = [s for s in ep.steps if s.kind in {"skill", "recovery", "stop"}]
    t0 = _num(steps[0].t) if steps else 0.0
    frames = []
    for i, s in enumerate(steps):
        last = i == len(steps) - 1
        call, outcome = _mapping(s.call), _mapping(s.outcome)
        frames.append({
            "observation.state": _state(s.obs_before),
            "action": [float(SKILL_INDEX.get(call.get("skill"), -1)), float(outcome.get("status") == "ok")],
            "timestamp": round(_num(s.t, t0) - t0, 4),
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
