from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody
from bruhos.trace import LocalSink, SinkDispatcher, load_episode


def test_close_drains_and_flush_can_finish(tmp_path):
    dispatcher = SinkDispatcher([LocalSink(tmp_path)])
    episode = Runtime(SimBody(), RulesPlanner(), dispatcher).run("Wave hello.")
    dispatcher.close()
    dispatcher._t.join(2)
    assert not dispatcher._t.is_alive()
    assert dispatcher.flush(0.2)
    assert dispatcher._q.unfinished_tasks == 0
    saved = load_episode(tmp_path / episode.episode_id)
    assert saved.trace_root == episode.trace_root
    assert saved.success
