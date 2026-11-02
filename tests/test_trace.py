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


class _ModelReply(RulesPlanner):
    """A chat brain whose JSON reply ends in a lone surrogate escape (an emoji cut in half)."""

    def plan(self, instruction, obs, catalog):
        from bruhos.brain import parse_plan
        return parse_plan('{"plan": [{"skill": "wave", "args": {}}, '
                          '{"skill": "detect", "args": {"target": "box \\ud83e"}}]}')


def test_lone_surrogates_are_sealed_and_saved(tmp_path):
    # argv carries undecodable bytes as lone surrogates (surrogateescape); JSON can decode them too.
    for brain, instruction in [(RulesPlanner(), "Wave hello \udcff"), (_ModelReply(), "Wave hello.")]:
        ep = Runtime(SimBody(), brain).run(instruction)
        assert len(ep.trace_root) == 64 and verify_episode(ep)
        loaded = load_episode(LocalSink(tmp_path).emit(ep)["path"])
        assert verify_episode(loaded) and loaded.trace_root == ep.trace_root


def test_canonical_bytes_unchanged_for_valid_text():
    from bruhos.trace.hashing import canonical

    assert canonical({"b": "招手 🤖", "a": 1}) == '{"a":1,"b":"招手 🤖"}'.encode()
    # A split pair hashes the way a JSON reader will load it back: as the joined character.
    assert canonical([chr(0xD83E) + chr(0xDD16)]) == canonical(["🤖"])
