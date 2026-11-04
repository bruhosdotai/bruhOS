from pathlib import Path

import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.cli import main
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody
from bruhos.trace import LocalSink, load_episode, verify_episode


@pytest.fixture
def saved_trace(tmp_path):
    ep = Runtime(SimBody(), RulesPlanner()).run("Put the red one in the box.")
    d = LocalSink(tmp_path).emit(ep)["path"]
    return ep, Path(d)


def test_load_episode_from_directory_path(saved_trace):
    orig, directory = saved_trace
    loaded = load_episode(directory)
    assert loaded.episode_id == orig.episode_id
    assert loaded.trace_root == orig.trace_root
    assert verify_episode(loaded)


@pytest.mark.parametrize("as_string", [False, True])
def test_load_episode_from_episode_json_path(saved_trace, as_string):
    orig, directory = saved_trace
    file_path = directory / "episode.json"
    arg = str(file_path) if as_string else file_path
    loaded = load_episode(arg)
    assert loaded.episode_id == orig.episode_id
    assert loaded.trace_root == orig.trace_root
    assert len(loaded.steps) == len(orig.steps)
    assert verify_episode(loaded)


@pytest.mark.parametrize("as_string", [False, True])
def test_load_episode_from_steps_jsonl_path(saved_trace, as_string):
    orig, directory = saved_trace
    file_path = directory / "steps.jsonl"
    arg = str(file_path) if as_string else file_path
    loaded = load_episode(arg)
    assert loaded.episode_id == orig.episode_id
    assert loaded.trace_root == orig.trace_root
    assert len(loaded.steps) == len(orig.steps)
    assert verify_episode(loaded)


def test_cli_verify_accepts_episode_json_file(saved_trace, capsys):
    orig, directory = saved_trace
    code = main(["verify", str(directory / "episode.json")])
    assert code == 0
    out = capsys.readouterr().out
    assert "OK" in out
    assert orig.trace_root in out


def test_cli_lerobot_accepts_episode_json_file(saved_trace, capsys):
    _, directory = saved_trace
    code = main(["lerobot", str(directory / "episode.json")])
    assert code == 0
    out = capsys.readouterr().out
    assert "features" in out
    assert "frames" in out


def test_load_episode_nonexistent_file_points_to_file():
    with pytest.raises(FileNotFoundError) as excinfo:
        load_episode("traces/nonexistent/episode.json")
    err_str = str(excinfo.value)
    assert "episode.json" in err_str
    assert "episode.json/episode.json" not in err_str.replace("\\", "/")
