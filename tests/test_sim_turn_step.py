"""The simulator must rotate by the turn step that skill logic reads."""

from bruhos.schemas import Status
from bruhos.skills import SkillContext, run_skill
from bruhos.sim.kinematic import SimBody, World, WorldObject
from bruhos.schemas import SkillCall


def test_turn_primitive_uses_the_configured_step():
    body = SimBody()
    body.turn_step_deg = 90.0
    body.act("turn_left")
    assert body.world.heading_deg == 90.0
    body.act("turn_right")
    assert body.world.heading_deg == 0.0


def test_scan_reaches_a_target_behind_with_a_custom_turn_step():
    world = World(objects=[WorldObject("block", "red", -1.0, 0.0)])
    body = SimBody(world=world)
    body.turn_step_deg = 90.0
    out = run_skill(SkillContext(body, max_steps=50), SkillCall("detect", {"target": "red_block"}))
    assert out.status is Status.OK, out.detail
