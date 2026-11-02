import math

import pytest

from bruhos.schemas import SkillCall, Status
from bruhos.skills import SkillContext, run_skill
from bruhos.sim.kinematic import SimBody, World, WorldObject


@pytest.mark.parametrize("bearing", [-15, 15])
def test_look_at_recognizes_alignment_on_the_last_allowed_turn(bearing):
    angle = math.radians(bearing)
    world = World(objects=[WorldObject("box", None, 0.3 * math.cos(angle), 0.3 * math.sin(angle))])
    body = SimBody(world)
    result = run_skill(SkillContext(body, max_steps=1), SkillCall("look_at", {"target": "box"}))
    assert result.status is Status.OK
    assert body.n_locomotion == 1
    assert abs(body.observe().find("box").bearing_deg) <= 8


def test_walk_recognizes_reach_on_the_last_allowed_step():
    body = SimBody(World(objects=[WorldObject("box", None, 0.15, 0)]))
    result = run_skill(SkillContext(body, max_steps=1), SkillCall("walk_toward", {"target": "box"}))
    assert result.status is Status.OK
    assert body.n_locomotion == 1
    assert body.observe().find("box").distance_m <= 0.1


@pytest.mark.parametrize("skill,bearing,distance", [("look_at", 30, 0.3), ("walk_toward", 0, 0.3)])
def test_unachieved_goal_still_fails_at_the_budget(skill, bearing, distance):
    angle = math.radians(bearing)
    body = SimBody(World(objects=[WorldObject("box", None, distance * math.cos(angle), distance * math.sin(angle))]))
    result = run_skill(SkillContext(body, max_steps=1), SkillCall(skill, {"target": "box"}))
    assert result.status is Status.FAILED
    assert body.n_locomotion == 1


@pytest.mark.parametrize("skill,bearing,distance", [("look_at", 15, 0.3), ("walk_toward", 0, 0.15)])
def test_stop_requested_after_the_final_primitive_precedes_goal_success(skill, bearing, distance):
    class StopAfterPrimitive(SimBody):
        def act(self, primitive):
            result = super().act(primitive)
            self.halt()
            return result

    angle = math.radians(bearing)
    body = StopAfterPrimitive(World(objects=[
        WorldObject("box", None, distance * math.cos(angle), distance * math.sin(angle)),
    ]))
    context = SkillContext(body, should_stop=body._halted.is_set, max_steps=1)
    result = run_skill(context, SkillCall(skill, {"target": "box"}))
    assert result.status is Status.STOPPED
    assert body.n_locomotion == 1
