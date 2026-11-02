from bruhos.body.base import Body
from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Imu, Observation, Status
from bruhos.skills import SkillContext
from bruhos.skills.builtin import stand_up


class MockFallenBody(Body):
    name = "mock"

    def __init__(self, can_recover: bool = True):
        self.fallen = True
        self.fell_forward = True
        self.can_recover = can_recover
        self.acts: list[str] = []

    def observe(self) -> Observation:
        return Observation(
            t=0.0,
            frame=1,
            imu=Imu(fallen=self.fallen, fell_forward=self.fell_forward if self.fallen else None),
        )

    def act(self, primitive: str) -> bool:
        self.acts.append(primitive)
        if self.can_recover:
            self.fallen = False
            self.fell_forward = None
        return True

    def halt(self) -> None:
        pass


def test_stand_up_halts_immediately_if_stopped():
    body = MockFallenBody()
    ctx = SkillContext(body=body, should_stop=lambda: True)
    outcome = stand_up(ctx, {})
    assert outcome.status == Status.STOPPED
    assert outcome.detail == "stop requested"
    assert body.acts == []


def test_stand_up_stops_between_recovery_primitives():
    body = MockFallenBody(can_recover=False)
    stopped = False

    def should_stop():
        return stopped

    ctx = SkillContext(body=body, should_stop=should_stop)

    orig_act = body.act

    def act_with_stop(prim: str) -> bool:
        nonlocal stopped
        stopped = True
        return orig_act(prim)

    body.act = act_with_stop

    outcome = stand_up(ctx, {})
    assert outcome.status == Status.STOPPED
    assert outcome.detail == "stop requested"
    assert body.acts == ["stand_up_front"]


def test_runtime_stops_during_recovery():
    class StopOnActBody(MockFallenBody):
        def __init__(self, runtime_holder):
            super().__init__(can_recover=False)
            self.runtime_holder = runtime_holder

        def act(self, primitive: str) -> bool:
            if "rt" in self.runtime_holder:
                self.runtime_holder["rt"].stop()
            return super().act(primitive)

    holder = {}
    body = StopOnActBody(holder)
    rt = Runtime(body, RulesPlanner())
    holder["rt"] = rt

    ep = rt.run("Wave hello.")
    assert ep.end_reason == "stopped"
    assert not ep.success
    assert any(s.kind == "recovery" and s.outcome["status"] == "stopped" for s in ep.steps)
