"""Local trace layout: ``<root>/<episode_id>/episode.json`` + ``steps.jsonl``."""

from __future__ import annotations

import json
from pathlib import Path

from bruhos.schemas import Episode, Step


class TraceWriter:
    def __init__(self, root: str | Path = "traces"):
        self.root = Path(root)

    def write(self, ep: Episode) -> Path:
        d = self.root / ep.episode_id
        d.mkdir(parents=True, exist_ok=True)
        with (d / "steps.jsonl").open("w") as f:
            for s in ep.steps:
                f.write(json.dumps(s.to_dict(), sort_keys=True) + "\n")
        (d / "episode.json").write_text(json.dumps(ep.header(), indent=2, sort_keys=True) + "\n")
        return d


def load_episode(path: str | Path) -> Episode:
    d = Path(path)
    h = json.loads((d / "episode.json").read_text())
    steps = []
    with (d / "steps.jsonl").open() as f:
        for line in f:
            s = json.loads(line)
            steps.append(Step(index=s["index"], t=s["t"], call=s["call"], outcome=s["outcome"],
                              obs_before=s["obs_before"], obs_after=s["obs_after"],
                              kind=s["kind"], hash=s["hash"]))
    return Episode(
        episode_id=h["episode_id"], instruction=h["instruction"], brain=h["brain"], body=h["body"],
        started_at=h["started_at"], plan=h["plan"], steps=steps, success=h["success"],
        recoveries=h["recoveries"], falls=h["falls"], ended_at=h["ended_at"], end_reason=h["end_reason"],
        skillpack_hash=h["skillpack_hash"], brain_version=h["brain_version"], trace_root=h["trace_root"],
    )
