"""A fallen robot with unknown fall direction must still be able to stand up.

Failing-test-first for a reproduced defect: the sim's ``_stand_up`` refused
any recovery attempt unless the attempted direction exactly matched
``fell_forward``, and ``fell_forward`` is ``None`` when the direction is
unknown. So a robot that starts fallen without a recorded direction — the
most natural way to construct a fallen world, ``World(fallen=True)`` — could
never get up: the stand_up skill tried both primitives, both failed, the
runtime burned all three recoveries, and the episode ended
``fallen, recovery budget exhausted`` with the robot permanently down. A
known direction still requires its matching primitive; only the unknown
direction is now accepted by either attempt.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import SkillCall
from bruhos.sim.kinematic import SimBody, World
from bruhos.skills import run_skill
from bruhos.skills.registry import SkillContext


def _fallen_world(fell_forward):
    w = World.tabletop()
    w.fallen = True
    w.fell_forward = fell_forward
    return w


def test_stand_up_unknown_direction_recovers():
    body = SimBody(_fallen_world(None))
    out = run_skill(SkillContext(body), SkillCall("stand_up", {}))
    assert out.status.name == "OK", out.detail
    assert not body.world.fallen


def test_stand_up_unknown_direction_episode_recovers():
    body = SimBody(_fallen_world(None))
    ep = Runtime(body, RulesPlanner()).run("Stand up.")
    assert ep.success, ep.end_reason
    assert ep.end_reason == "completed"
    assert ep.recoveries == 1
    assert not body.world.fallen


def test_stand_up_known_direction_unchanged():
    front = SimBody(_fallen_world(True))
    out = run_skill(SkillContext(front), SkillCall("stand_up", {}))
    assert out.status.name == "OK", out.detail
    assert "stand_up_front" in out.detail
    assert not front.world.fallen
    back = SimBody(_fallen_world(False))
    out = run_skill(SkillContext(back), SkillCall("stand_up", {}))
    assert out.status.name == "OK", out.detail
    assert "stand_up_back" in out.detail
    assert not back.world.fallen


def test_stand_up_wrong_direction_primitive_still_refused():
    # A known direction still requires its matching primitive: the fix
    # only opens the unknown-direction case, it does not accept anything.
    body = SimBody(_fallen_world(True))
    assert body.act("stand_up_back") is False
    assert body.world.fallen
    body = SimBody(_fallen_world(False))
    assert body.act("stand_up_front") is False
    assert body.world.fallen


def test_stand_up_standing_unchanged():
    body = SimBody(World.tabletop())
    out = run_skill(SkillContext(body), SkillCall("stand_up", {}))
    assert out.status.name == "OK"
    assert out.detail == "already standing"
