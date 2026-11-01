"""A ball in reach must be grabbable in the simulator.

Failing-test-first for a reproduced defect: ``SimBody._p_grab`` excluded
``ball`` (like ``box``) from graspable objects, so a ball standing right in
front of the robot could never be picked up — ``grab`` failed with the
misleading detail "nothing in reach" even at 0.1 m, dead-on bearing. Nothing
documents balls as kick-only: the ``grab`` skill says "Close the hand on the
object in reach", and the project's own model examples (``lfm3_omni`` few-shot
prompts) teach brains to ``detect`` / ``walk_toward`` / ``grab`` a ball to
bring it to the box. A model brain following that pattern therefore fails a
perfectly ordinary "pick up the ball" task. The box stays ungraspable: it is a
container, not a manipulable object.
"""

from bruhos.brain.base import Planner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation, SkillCall
from bruhos.sim.kinematic import SimBody, World
from bruhos.skills import SkillContext, run_skill


def _walk_grab(ctx, target):
    out = run_skill(ctx, SkillCall("walk_toward", {"target": target}))
    assert out.status.value == "ok", out
    return run_skill(ctx, SkillCall("grab", {}))


class _ScriptedPlanner(Planner):
    name = "scripted"
    version = "scripted-0.1"

    def plan(self, instruction, obs, catalog):
        return [SkillCall("walk_toward", {"target": "yellow_ball"}), SkillCall("grab", {}),
                SkillCall("walk_toward", {"target": "box"}), SkillCall("release", {})]


def test_ball_grabbable_when_in_reach():
    body = SimBody(World.tabletop())
    ball = body.world.get("yellow_ball")
    body.world.x, body.world.y = ball.x - 0.1, ball.y
    body.world.heading_deg = 0.0
    out = run_skill(SkillContext(body), SkillCall("grab", {}))
    assert out.status.value == "ok", out
    assert body.world.holding is not None and body.world.holding.name == "yellow_ball"


def test_walk_to_ball_then_grab():
    body = SimBody(World.tabletop())
    out = _walk_grab(SkillContext(body), "yellow_ball")
    assert out.status.value == "ok", out
    assert body.world.holding is not None and body.world.holding.name == "yellow_ball"


def test_ball_can_be_put_in_box():
    body = SimBody(World.tabletop())
    ctx = SkillContext(body)
    out = _walk_grab(ctx, "yellow_ball")
    assert out.status.value == "ok", out
    assert run_skill(ctx, SkillCall("walk_toward", {"target": "box"})).status.value == "ok"
    assert run_skill(ctx, SkillCall("release", {})).status.value == "ok"
    assert "in_box:yellow_ball" in body.world.events


def test_runtime_delivers_ball_to_box():
    body = SimBody(World.tabletop())
    ep = Runtime(body, _ScriptedPlanner()).run("Bring the ball to the box.")
    assert ep.success, ep.end_reason
    assert "in_box:yellow_ball" in body.world.events


def test_box_still_not_grabbable():
    # Guard: the box is a container, not a manipulable object.
    body = SimBody(World.tabletop())
    ctx = SkillContext(body)
    assert run_skill(ctx, SkillCall("walk_toward", {"target": "box"})).status.value == "ok"
    out = run_skill(ctx, SkillCall("grab", {}))
    assert out.status.value == "failed", out


def test_block_grab_unchanged():
    # Guard: ordinary block grasping is unaffected.
    body = SimBody(World.tabletop())
    out = _walk_grab(SkillContext(body), "red_block")
    assert out.status.value == "ok", out
    assert body.world.holding is not None and body.world.holding.name == "red_block"
