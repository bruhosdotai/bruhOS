import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import FORWARD_M, SimBody
from bruhos.trace import LocalSink, SinkDispatcher, load_episode, verify_episode


@pytest.mark.parametrize("phase, halt_fails", [("act", False), ("observe", False), ("act", True)])
def test_body_error_halts_before_finalizing_and_requires_reset(tmp_path, phase, halt_fails):
    fault = OSError("synthetic body communication failure")

    class FaultingSimBody(SimBody):
        faulted = False
        observation_fault_pending = False
        halt_attempts = 0

        def act(self, primitive):
            result = super().act(primitive)
            if primitive == "forward" and not self.faulted:
                self.faulted = True
                if phase == "act":
                    raise fault
                self.observation_fault_pending = True
            return result

        def observe(self):
            if self.observation_fault_pending:
                self.observation_fault_pending = False
                raise fault
            return super().observe()

        def halt(self):
            self.halt_attempts += 1
            if halt_fails:
                raise RuntimeError("synthetic halt failure")
            super().halt()

    body = FaultingSimBody()
    halted_when_saved = []

    class InspectingLocalSink(LocalSink):
        def emit(self, ep):
            halted_when_saved.append(body.halt_attempts)
            return super().emit(ep)

    dispatcher = SinkDispatcher([InspectingLocalSink(tmp_path)])
    rt = Runtime(body, RulesPlanner(), dispatcher)
    try:
        with pytest.raises(OSError) as raised:
            rt.run("Walk to the box.")
        assert raised.value is fault
        assert body.world.x == pytest.approx(FORWARD_M)
        assert body.halt_attempts == 1
        assert rt.ctx.should_stop()
        assert body._halted.is_set() == (not halt_fails)

        assert dispatcher.flush(2)
        assert halted_when_saved == [1]
        saved = load_episode(dispatcher.receipts[0][2]["path"])
        assert not saved.success and saved.ended_at is not None
        assert verify_episode(saved)

        stopped = rt.run("Wave hello.")
        assert not stopped.success and stopped.end_reason == "stopped"
        assert "wave" not in body.world.events
        rt.reset_stop()
        resumed = rt.run("Wave hello.")
        assert resumed.success and "wave" in body.world.events
        assert dispatcher.flush(2)
    finally:
        dispatcher.close()
        dispatcher._t.join(2)
