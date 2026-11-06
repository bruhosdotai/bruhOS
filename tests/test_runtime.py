import threading
import time

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.brain.base import Planner
from bruhos.runtime import Runtime
from bruhos.schemas import SkillCall, Status
from bruhos.sim.kinematic import SimBody, World
from bruhos.trace import Sink, SinkDispatcher


class FixedPlanner(Planner):
    name = "fixed"

    def __init__(self, plan):
        self._plan = plan

    def plan(self, instruction, obs, catalog):
        return list(self._plan)


def test_put_red_in_box():
    body = SimBody()
    ep = Runtime(body, RulesPlanner()).run("Put the red one in the box.")
    assert ep.success, ep.end_reason
    assert "in_box:red_block" in body.world.events
    assert [c["skill"] for c in ep.plan] == ["detect", "walk_toward", "grab", "walk_toward", "release"]


def test_fall_mid_skill_recovers_and_resumes():
    body = SimBody(falls_at={4})
    ep = Runtime(body, RulesPlanner()).run("Put the red one in the box.")
    assert ep.falls == 1 and ep.recoveries == 1
    kinds = [s.kind for s in ep.steps]
    assert "recovery" in kinds
    resumed = ep.steps[kinds.index("recovery") + 1]
    assert resumed.call == ep.steps[kinds.index("recovery") - 1].call
    assert ep.success and "in_box:red_block" in body.world.events


def test_unregistered_skill_from_brain_never_reaches_body():
    body = SimBody()
    ep = Runtime(body, FixedPlanner([SkillCall("wave"), SkillCall("moonwalk")])).run("Dance.")
    assert not ep.success
    assert ep.steps[0].kind == "rejected"
    assert "wave" not in body.world.events


def test_stop_preempts_running_skill():
    class SlowBody(SimBody):
        def act(self, primitive):
            time.sleep(0.01)
            return super().act(primitive)

    body = SlowBody(World(objects=[]))          # nothing to find: detect scans until stopped
    rt = Runtime(body, FixedPlanner([SkillCall("detect", {"target": "red_block"})]))
    threading.Timer(0.05, rt.stop).start()
    t0 = time.monotonic()
    ep = rt.run("Find the red block.")
    assert time.monotonic() - t0 < 1.0
    assert ep.end_reason == "stopped" or ep.steps[-1].outcome["status"] in {"stopped", "failed"}
    assert body._halted.is_set()


def test_slow_sink_does_not_block_control_loop():
    class SlowSink(Sink):
        name = "slow"

        def emit(self, ep):
            time.sleep(2.0)
            return {}

    d = SinkDispatcher([SlowSink()])
    t0 = time.monotonic()
    ep = Runtime(SimBody(), RulesPlanner(), d).run("Wave hello.")
    assert ep.success
    assert time.monotonic() - t0 < 0.5


def test_failed_sink_does_not_crash():
    class BadSink(Sink):
        name = "bad"

        def emit(self, ep):
            raise RuntimeError("chain down")

    d = SinkDispatcher([BadSink()])
    ep = Runtime(SimBody(), RulesPlanner(), d).run("Wave hello.")
    assert d.flush(2)
    assert ep.success and d.receipts[0][2].startswith("error")


def test_out_of_scope_instruction_has_no_plan():
    ep = Runtime(SimBody(), RulesPlanner()).run("Do a backflip.")
    assert not ep.success and ep.end_reason == "no plan for instruction"


def _holding_body(name: str) -> SimBody:
    body = SimBody()
    body.world.holding = body.world.get(name)
    return body


def test_look_at_already_held_target_succeeds():
    # A held object is invisible to observe(), so the scan below could never
    # find it and look_at failed after a pointless full revolution.
    from bruhos.skills import SkillContext, run_skill

    for target in ("red_block", "red", "block"):
        out = run_skill(SkillContext(_holding_body("red_block")), SkillCall("look_at", {"target": target}))
        assert out.status is Status.OK, target
        assert out.data["bearing_deg"] == 0.0, target


def test_detect_already_held_target_succeeds():
    from bruhos.skills import SkillContext, run_skill

    for target in ("red_block", "red", "block"):
        out = run_skill(SkillContext(_holding_body("red_block")), SkillCall("detect", {"target": target}))
        assert out.status is Status.OK, target
        assert out.data == {"found": "red_block", "bearing_deg": 0.0, "distance_m": 0.0}, target


def test_held_target_check_does_not_mask_other_targets():
    from bruhos.skills import SkillContext, run_skill

    body = _holding_body("red_block")
    assert run_skill(SkillContext(body), SkillCall("look_at", {"target": "blue_block"})).status is Status.OK
    assert run_skill(SkillContext(body), SkillCall("detect", {"target": "blue_block"})).status is Status.OK
    assert run_skill(SkillContext(body), SkillCall("detect", {"target": "green_block"})).status is Status.OK


def test_look_at_and_detect_held_target_end_to_end():
    body = _holding_body("red_block")
    ep = Runtime(body, RulesPlanner()).run("Look at the red block.")
    assert ep.success, ep.end_reason
    body = _holding_body("red_block")
    ep = Runtime(body, RulesPlanner()).run("Find the red block.")
    assert ep.success, ep.end_reason
