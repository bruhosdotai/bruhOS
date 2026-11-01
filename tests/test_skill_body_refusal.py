"""A skill must not report success for a primitive the body refused to run.

``Body.act`` returns False when the body could not run a primitive: an action group missing from
``configs/*.toml``, a halted body, a stopped bridge, a robot lying down. The whitelist contract is
that such a step fails, so the runtime can replan or end the episode. Skills that ignore the return
value reported OK for motion that never happened.
"""

from bruhos.body.base import Body
from bruhos.brain.base import Planner
from bruhos.runtime import Runtime
from bruhos.schemas import Detection, Imu, LineReading, Observation, SkillCall, Status
from bruhos.skills import SkillContext, run_skill


class RefusingBody(Body):
    """Records every primitive and refuses the ones in ``refuse``, like an unmapped action group."""

    name = "refusing"
    turn_step_deg = 15.0

    def __init__(self, refuse=(), objects=(), line=True):
        self.refuse = set(refuse)
        self.objects = list(objects)
        self.line = line
        self.acts: list[str] = []
        self.halted = False
        self.stop_after_act = False

    def observe(self):
        return Observation(t=0, frame=0, objects=list(self.objects),
                           imu=Imu(), line=LineReading(visible=self.line, offset=0.0))

    def act(self, primitive):
        self.acts.append(primitive)
        if self.stop_after_act:
            self.halted = True
        return primitive not in self.refuse and not self.halted

    def halt(self):
        self.halted = True


def _run(body, skill, **args):
    return run_skill(SkillContext(body), SkillCall(skill, args))


def test_wave_fails_when_the_body_refuses_it():
    body = RefusingBody(refuse={"wave"})
    out = _run(body, "wave")
    assert out.status is Status.FAILED
    assert "wave" in out.detail and body.acts == ["wave"]


def test_kick_fails_when_the_body_refuses_it():
    body = RefusingBody(refuse={"kick_left"})
    out = _run(body, "kick", foot="left")
    assert out.status is Status.FAILED
    assert "kick_left" in out.detail and body.acts == ["kick_left"]


def test_follow_line_does_not_count_a_refused_forward_step():
    body = RefusingBody(refuse={"forward"})
    out = _run(body, "follow_line", steps=3)
    assert out.status is Status.FAILED
    assert body.acts == ["forward"]


def test_walk_toward_stops_when_the_body_refuses_forward():
    body = RefusingBody(objects=[Detection("block", "red", 0.0, 0.5)], refuse={"forward"})
    out = run_skill(SkillContext(body, max_steps=20), SkillCall("walk_toward", {"target": "red_block"}))
    assert out.status is Status.FAILED
    assert body.acts == ["forward"]              # one refusal, not the whole step budget


def test_look_at_stops_when_the_body_refuses_to_turn():
    body = RefusingBody(objects=[Detection("block", "red", 60.0, 0.5)], refuse={"turn_left"})
    out = run_skill(SkillContext(body, max_steps=20), SkillCall("look_at", {"target": "red_block"}))
    assert out.status is Status.FAILED
    assert body.acts == ["turn_left"]


def test_detect_scan_stops_when_the_body_refuses_to_turn():
    body = RefusingBody(refuse={"turn_left"})
    out = _run(body, "detect", target="red_block")
    assert out.status is Status.FAILED
    assert body.acts == ["turn_left"]


def test_a_refused_primitive_during_a_stop_is_reported_as_stopped():
    body = RefusingBody(refuse={"wave"})
    body.stop_after_act = True
    out = run_skill(SkillContext(body, should_stop=lambda: body.halted), SkillCall("wave"))
    assert out.status is Status.STOPPED


class FixedPlanner(Planner):
    name = "fixed"

    def __init__(self, plan):
        self._plan = plan

    def plan(self, instruction, obs, catalog):
        return list(self._plan)


def test_runtime_does_not_mark_a_refused_skill_successful():
    body = RefusingBody(refuse={"wave"})
    ep = Runtime(body, FixedPlanner([SkillCall("wave")])).run("Wave hello.")
    assert not ep.success
    assert ep.steps[-1].outcome["status"] == Status.FAILED.value
