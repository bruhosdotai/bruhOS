"""A grab must not certify the wrong object as success.

Failing-test-first for a reproduced defect: the ``grab`` skill took no
target, so it closed the hand on whatever was nearest in reach while the
planner's pickup/delivery plans assumed it grabbed the walked-to object.
In a cluttered scene that assumption breaks: ``Pick up the red block.``
walked to the red block, grabbed the nearer blue block, and reported
``success=True`` — and ``Put the red block in the box.`` delivered the
blue block with event ``in_box:blue_block`` as a certified success. The
grab now accepts an optional ``target`` and fails when the held object
does not match it, so the episode fails honestly instead of certifying
the wrong object. A grab without a target keeps the old behavior.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.brain.base import Planner
from bruhos.runtime import Runtime
from bruhos.schemas import SkillCall
from bruhos.sim.kinematic import SimBody, World, WorldObject


def _clutter_world():
    # Blue sits slightly closer to the robot's stopping point than red,
    # so a target-blind grab takes blue after walking to red.
    return World(objects=[
        WorldObject("block", "red", 0.5, 0.0),
        WorldObject("block", "blue", 0.45, 0.01),
        WorldObject("box", None, 0.9, 0.0),
        WorldObject("ball", "yellow", 0.4, -0.6),
    ])


def _grab_steps(ep):
    return [s for s in ep.steps if s.call["skill"] == "grab"]


class _StubPlanner(Planner):
    name = "stub"

    def __init__(self, plan):
        self._plan = plan

    def plan(self, instruction, obs, catalog):
        return list(self._plan)


def test_pickup_wrong_object_fails():
    body = SimBody(_clutter_world())
    ep = Runtime(body, RulesPlanner()).run("Pick up the red block.")
    assert not ep.success
    grabs = _grab_steps(ep)
    assert grabs[0].outcome["status"] == "failed"
    assert "blue_block" in grabs[0].outcome["detail"]
    assert "expected red_block" in grabs[0].outcome["detail"]
    # Truthful state, not a certified success: the blue block is held.
    assert body.world.holding is not None and body.world.holding.name == "blue_block"


def test_delivery_wrong_object_not_delivered():
    body = SimBody(_clutter_world())
    ep = Runtime(body, RulesPlanner()).run("Put the red block in the box.")
    assert not ep.success
    grabs = _grab_steps(ep)
    assert grabs[0].outcome["status"] == "failed"
    assert "expected red_block" in grabs[0].outcome["detail"]
    assert "in_box:blue_block" not in body.world.events
    assert "in_box:red_block" not in body.world.events


def test_pickup_matching_object_still_works():
    body = SimBody()
    ep = Runtime(body, RulesPlanner()).run("Pick up the red block.")
    assert ep.success, ep.end_reason
    assert body.world.holding is not None and body.world.holding.name == "red_block"


def test_delivery_matching_object_still_works():
    body = SimBody()
    ep = Runtime(body, RulesPlanner()).run("Put the red block in the box.")
    assert ep.success, ep.end_reason
    assert "in_box:red_block" in body.world.events


def test_generic_target_accepts_any_matching_block():
    # "the block" names no color, so the nearest block satisfies it.
    body = SimBody(_clutter_world())
    ep = Runtime(body, RulesPlanner()).run("Pick up the block.")
    assert ep.success, ep.end_reason
    assert body.world.holding is not None and body.world.holding.name == "blue_block"


def test_partial_target_accepts_matching_block():
    body = SimBody()
    ep = Runtime(body, RulesPlanner()).run("Grab the red one.")
    assert ep.success, ep.end_reason
    assert body.world.holding is not None and body.world.holding.name == "red_block"


def test_targetless_grab_unchanged():
    # No target given: the old target-blind behavior is preserved.
    body = SimBody(_clutter_world())
    plan = [SkillCall("walk_toward", {"target": "red_block"}), SkillCall("grab", {})]
    ep = Runtime(body, _StubPlanner(plan)).run("Pick up the red block.")
    assert ep.success, ep.end_reason
    assert body.world.holding is not None and body.world.holding.name == "blue_block"
