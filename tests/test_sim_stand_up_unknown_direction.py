"""A fall with unknown direction must still be recoverable.

When ``fell_forward`` is None the robot fell, but the direction is unknown
(e.g. a roll-only fall). The stand_up skill tries both recovery primitives;
the sim used to refuse both, so the runtime burned its whole recovery
budget and ended the episode fallen.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import SkillCall, Status
from bruhos.sim.kinematic import SimBody, World
from bruhos.skills import SkillContext, run_skill


def test_stand_up_succeeds_when_direction_unknown():
    w = World(fallen=True, fell_forward=None)
    body = SimBody(w)
    out = run_skill(SkillContext(body), SkillCall("stand_up"))
    assert out.status is Status.OK
    assert w.fallen is False


def test_stand_up_still_refuses_wrong_direction_when_known():
    w = World(fallen=True, fell_forward=False)
    body = SimBody(w)
    assert body.act("stand_up_front") is False
    assert w.fallen is True
    assert body.act("stand_up_back") is True
    assert w.fallen is False


def test_runtime_recovers_from_unknown_direction_fall():
    w = World(fallen=True, fell_forward=None)
    ep = Runtime(SimBody(w), RulesPlanner()).run("Stand up.")
    assert ep.success is True
    assert ep.end_reason == "completed"
    assert w.fallen is False
