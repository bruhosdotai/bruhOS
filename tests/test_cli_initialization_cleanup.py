import pytest

from bruhos.cli import main
from bruhos.sim.kinematic import SimBody
from bruhos.trace import SinkDispatcher


def test_bad_brain_configuration_closes_body_and_sink_worker(tmp_path, monkeypatch):
    import bruhos.cli as cli
    import bruhos.trace as trace

    class Body(SimBody):
        closed = False

        def close(self):
            self.closed = True

    workers = []

    class Dispatcher(SinkDispatcher):
        def __init__(self, sinks):
            super().__init__(sinks)
            workers.append(self)

    body = Body()
    monkeypatch.setattr(cli, "_body", lambda *args: body)
    monkeypatch.setattr(trace, "SinkDispatcher", Dispatcher)
    try:
        with pytest.raises(ValueError, match="unknown brain"):
            main(["run", "Wave hello.", "--brain", "invalid", "--traces", str(tmp_path)])
        assert body.closed
        workers[0]._t.join(1)
        assert not workers[0]._t.is_alive()
    finally:
        for worker in workers:
            if worker._t.is_alive():
                worker.close()
                worker._t.join(1)
