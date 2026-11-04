"""An interrupted run must not save a trace with a blank end_reason.

``Runtime.run`` seals and submits the episode in its ``finally`` block, so an
episode cut short by Ctrl-C is persisted even though ``KeyboardInterrupt``
propagates and ``cmd_run`` (which maps it to exit 130) never sees the episode.
Every normal exit path in ``_run`` sets ``ep.end_reason``; the interrupt path
set nothing, so the saved header — and the dashboard "Result" column built
from it — showed an empty reason for an episode that in fact was interrupted.
"""

import pytest

from bruhos.brain.base import Planner
from bruhos.dashboard import status_data
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody
from bruhos.trace import LocalSink, SinkDispatcher, load_episode, verify_episode


class CtrlCPlanner(Planner):
    name = "ctrlc"

    def plan(self, instruction, obs, catalog):
        raise KeyboardInterrupt


def _interrupted_episode_dir(tmp_path):
    dispatcher = SinkDispatcher([LocalSink(tmp_path)])
    try:
        with pytest.raises(KeyboardInterrupt):
            Runtime(SimBody(), CtrlCPlanner(), dispatcher).run("Wave hello.")
        assert dispatcher.flush(2)
    finally:
        dispatcher.close()
    directories = list(tmp_path.iterdir())
    assert len(directories) == 1
    return directories[0]


def test_interrupt_is_recorded_in_the_saved_episode(tmp_path):
    ep = load_episode(_interrupted_episode_dir(tmp_path))
    assert not ep.success
    assert ep.end_reason == "interrupted"
    assert ep.ended_at is not None
    assert verify_episode(ep)


def test_dashboard_shows_interrupted_instead_of_a_blank_result(tmp_path):
    _interrupted_episode_dir(tmp_path)
    data = status_data(tmp_path)
    assert len(data["episodes"]) == 1
    assert data["episodes"][0]["end_reason"] == "interrupted"
    assert data["episodes"][0]["verified"]
