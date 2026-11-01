import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody
from bruhos.trace import LocalSink, SinkDispatcher, load_episode, verify_episode


def test_body_error_is_recorded_while_exception_still_propagates(tmp_path):
    class BrokenBody(SimBody):
        def act(self, primitive):
            raise OSError("synthetic body failure")

    dispatcher = SinkDispatcher([LocalSink(tmp_path)])
    try:
        with pytest.raises(OSError, match="synthetic body failure"):
            Runtime(BrokenBody(), RulesPlanner(), dispatcher).run("Wave hello.")
        assert dispatcher.flush(2)
        directories = list(tmp_path.iterdir())
        assert len(directories) == 1
        episode = load_episode(directories[0])
        assert episode.end_reason == "runtime error: OSError"
        assert not episode.success
        assert episode.ended_at is not None
        assert verify_episode(episode)
    finally:
        dispatcher.close()
        dispatcher._t.join(1)
