"""`bruhos lerobot` prints LEROBOT_FEATURES as the schema for the frames printed beside it.

`task` is listed in this module's own alignment docstring, `tests/test_trace.py` already
asserts `frame["task"] == ep.instruction`, and every frame carries it -- but the feature map
has no entry for it, so the published schema omits the one field that ties a dataset to the
instruction the robot was given.
"""

import json

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.cli import main
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody, World
from bruhos.trace.lerobot import LEROBOT_FEATURES, to_lerobot_frames
from bruhos.trace.writer import TraceWriter

# bruhos side channels are namespaced and deliberately outside the LeRobot schema.
_EXTENSIONS = "bruhos."


def _episode():
    return Runtime(SimBody(World.tabletop()), RulesPlanner()).run("Put the red one in the box.")


def _schema_fields(frame: dict) -> set[str]:
    return {k for k in frame if not k.startswith(_EXTENSIONS)}


def test_every_frame_field_is_declared_by_the_feature_map():
    frames = to_lerobot_frames(_episode())
    assert frames
    for frame in frames:
        undeclared = _schema_fields(frame) - set(LEROBOT_FEATURES)
        assert not undeclared, f"frame carries {sorted(undeclared)}, which LEROBOT_FEATURES does not describe"


def test_task_is_declared_like_the_other_scalar_columns():
    ep = _episode()
    frames = to_lerobot_frames(ep)
    assert all(f["task"] == ep.instruction for f in frames)
    spec = LEROBOT_FEATURES["task"]
    assert spec["dtype"] == "string"
    assert spec["shape"] == [1]
    assert spec["names"] is None


def test_cli_lerobot_prints_a_schema_that_covers_its_frames(tmp_path, capsys):
    directory = TraceWriter(tmp_path).write(_episode())
    assert main(["lerobot", str(directory)]) == 0
    printed = json.loads(capsys.readouterr().out)
    for frame in printed["frames"]:
        assert _schema_fields(frame) <= set(printed["features"])
