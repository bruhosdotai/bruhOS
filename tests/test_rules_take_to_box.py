"""A fetch verb plus a box destination must plan the delivery, not just the pickup.

Failing-test-first for a reproduced defect: the rules planner's ``into_box``
branch only fired for put/place/drop/bring/move/carry/throw, while the
pickup branch claimed take/get. So ``Take the red block to the box.`` and
``Get the red block to the box.`` planned only ``walk_toward`` + ``grab`` and
the episode reported ``success=True`` with the block held and never delivered
— a silently incomplete plan for an explicit delivery instruction. ``Bring``/
``Move`` to the box already produced the full five-step plan; take/get must
too, since they are the same fetch verbs.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody, World


def _obs(holding):
    return Observation(t=0.0, frame=0, objects=[], holding=holding)


def _plan_text(plan):
    return [(c.skill, dict(c.args)) for c in plan]


_FULL = [
    ("detect", {"target": "red_block"}),
    ("walk_toward", {"target": "red_block"}),
    ("grab", {}),
    ("walk_toward", {"target": "box"}),
    ("release", {}),
]


def test_take_block_to_box_plans_delivery():
    plan = RulesPlanner().plan("Take the red block to the box.", _obs(None), [])
    assert _plan_text(plan) == _FULL


def test_get_block_to_box_plans_delivery():
    plan = RulesPlanner().plan("Get the red block to the box.", _obs(None), [])
    assert _plan_text(plan) == _FULL


def test_take_it_to_box_with_holding_skips_pickup():
    plan = RulesPlanner().plan("Take it to the box.", _obs("red_block"), [])
    assert _plan_text(plan) == [
        ("walk_toward", {"target": "box"}),
        ("release", {}),
    ]


def test_take_block_to_box_runtime_delivers():
    body = SimBody(World.tabletop())
    ep = Runtime(body, RulesPlanner()).run("Take the red block to the box.")
    assert ep.success, ep.end_reason
    assert "in_box:red_block" in body.world.events


def test_take_it_to_box_runtime_delivers_held_block():
    world = World.tabletop()
    world.holding = world.get("red_block")
    body = SimBody(world)
    ep = Runtime(body, RulesPlanner()).run("Take it to the box.")
    assert ep.success, ep.end_reason
    assert "in_box:red_block" in body.world.events


def test_take_block_without_destination_unchanged():
    # Guard: bare "Take the red block." requests no delivery, so walk+grab stays.
    plan = RulesPlanner().plan("Take the red block.", _obs(None), [])
    assert _plan_text(plan) == [
        ("walk_toward", {"target": "red_block"}),
        ("grab", {}),
    ]


def test_bring_and_move_to_box_unchanged():
    # Guard: the verbs that already worked must keep producing the full plan.
    for instruction in ("Bring the red block to the box.", "Move the red block to the box."):
        plan = RulesPlanner().plan(instruction, _obs(None), [])
        assert _plan_text(plan) == _FULL
