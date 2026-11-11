from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody
from bruhos.trace import (ChainSink, LEROBOT_FEATURES, LocalSink, load_episode, merkle_root,
                          to_lerobot_frames, verify_episode)


def _episode():
    return Runtime(SimBody(), RulesPlanner()).run("Put the red one in the box.")


def test_every_episode_has_a_root_and_verifies():
    ep = _episode()
    assert len(ep.trace_root) == 64 and all(s.hash for s in ep.steps)
    assert verify_episode(ep)


def test_roundtrip_and_tamper_detection(tmp_path):
    ep = _episode()
    d = LocalSink(tmp_path).emit(ep)["path"]
    loaded = load_episode(d)
    assert verify_episode(loaded) and loaded.trace_root == ep.trace_root
    loaded.steps[2].outcome["status"] = "failed"
    assert not verify_episode(loaded)


def test_reorder_changes_root():
    ep = _episode()
    leaves = [s.hash for s in ep.steps]
    assert merkle_root(leaves) != merkle_root(leaves[::-1])


def test_chain_payload_has_hashes_not_video():
    ep = _episode()
    p = ChainSink.anchor_payload(ep)
    assert p["trace_root"] == "0x" + ep.trace_root
    assert p["skillpack_hash"].startswith("sha256:")
    assert set(p) == {"episode_id", "trace_root", "skillpack_hash", "brain_version", "success", "n_steps"}


def test_lerobot_frames_match_features():
    ep = _episode()
    frames = to_lerobot_frames(ep, episode_index=3)
    assert len(frames) == len(ep.steps)
    for i, f in enumerate(frames):
        assert len(f["observation.state"]) == LEROBOT_FEATURES["observation.state"]["shape"][0]
        assert len(f["action"]) == LEROBOT_FEATURES["action"]["shape"][0]
        assert f["frame_index"] == i and f["episode_index"] == 3 and f["task"] == ep.instruction
    assert frames[-1]["next.done"] and frames[-1]["next.success"]


def test_trace_roundtrip_with_unicode(tmp_path, monkeypatch):
    from pathlib import Path
    from bruhos.schemas import Episode, Step
    from bruhos.trace.hashing import seal

    orig_open = Path.open

    def strict_locale_open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        if "b" not in mode and encoding is None:
            encoding = "ascii"
        return orig_open(self, mode=mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(Path, "open", strict_locale_open)

    ep = Episode(
        episode_id="unicode-test-1",
        instruction="Robot \U0001f916 wave \u62db\u624b",
        brain="rules",
        body="sim",
        started_at=1.0,
        steps=[
            Step(
                index=0,
                t=1.0,
                call={"skill": "wave", "args": {}},
                outcome={"status": "ok", "detail": "waved \u270c\ufe0f (\u62db\u624b)"},
                obs_before={"t": 1.0, "frame": 1, "objects": [], "imu": {}, "line": {}},
            )
        ],
    )
    seal(ep)
    d = Path(LocalSink(tmp_path).emit(ep)["path"])
    assert "Robot \U0001f916 wave \u62db\u624b" in (d / "episode.json").read_text(encoding="utf-8")
    assert "waved \u270c\ufe0f (\u62db\u624b)" in (d / "steps.jsonl").read_text(encoding="utf-8")
    loaded = load_episode(d)
    assert loaded.instruction == "Robot \U0001f916 wave \u62db\u624b"
    assert loaded.steps[0].outcome["detail"] == "waved \u270c\ufe0f (\u62db\u624b)"
    assert verify_episode(loaded)


def test_load_episode_utf8_bytes_under_ascii_locale(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from bruhos.schemas import Episode, Step
    from bruhos.trace.hashing import seal

    ep = Episode(
        episode_id="unicode-load-1",
        instruction="Robot \U0001f916 wave \u62db\u624b",
        brain="rules",
        body="sim",
        started_at=1.0,
        steps=[
            Step(
                index=0,
                t=1.0,
                call={"skill": "wave", "args": {}},
                outcome={"status": "ok", "detail": "waved \u270c\ufe0f (\u62db\u624b)"},
                obs_before={"t": 1.0, "frame": 1, "objects": [], "imu": {}, "line": {}},
            )
        ],
    )
    seal(ep)
    d = tmp_path / ep.episode_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "episode.json").write_bytes(
        (json.dumps(ep.header(), indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    )
    (d / "steps.jsonl").write_bytes(
        (json.dumps(ep.steps[0].to_dict(), sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    )

    orig_open = Path.open

    def strict_locale_open(self, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
        if "b" not in mode and encoding is None:
            encoding = "ascii"
        return orig_open(self, mode=mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(Path, "open", strict_locale_open)

    loaded = load_episode(d)
    assert loaded.instruction == "Robot \U0001f916 wave \u62db\u624b"
    assert verify_episode(loaded)

