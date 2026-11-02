"""A box destination with an adjective must still plan the delivery.

Failing-test-first for a reproduced defect: the rules planner's ``into_box``
destination matcher required the destination noun immediately after the
article, so "the big box" / "a box" / "the wooden basket" never matched.
Depending on the verb, the same delivery instruction then failed three
different ways on base: put/bring/carry got no plan at all, move planned a
walk-only trip that reported success without touching the block, and the
destination half of the instruction was silently dropped. The destination
is still the same single box; only the phrasing differs.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody, World


def _obs(holding=None):
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


def test_put_block_in_big_box_plans_delivery():
    plan = RulesPlanner().plan("Put the red block in the big box.", _obs(), [])
    assert _plan_text(plan) == _FULL


def test_bring_block_to_big_box_plans_delivery():
    plan = RulesPlanner().plan("Bring the red block to the big box.", _obs(), [])
    assert _plan_text(plan) == _FULL


def test_move_block_to_big_box_plans_delivery():
    plan = RulesPlanner().plan("Move the red block to the big box.", _obs(), [])
    assert _plan_text(plan) == _FULL


def test_carry_block_to_wooden_basket_plans_delivery():
    plan = RulesPlanner().plan("Carry the red block to the wooden basket.", _obs(), [])
    assert _plan_text(plan) == _FULL


def test_put_block_in_a_box_plans_delivery():
    plan = RulesPlanner().plan("Put the red block in a box.", _obs(), [])
    assert _plan_text(plan) == _FULL


def test_put_block_in_big_box_runtime_delivers():
    body = SimBody(World.tabletop())
    ep = Runtime(body, RulesPlanner()).run("Put the red block in the big box.")
    assert ep.success, ep.end_reason
    assert "in_box:red_block" in body.world.events


def test_put_block_in_box_unchanged():
    # Guard: the bare phrasing that already worked must keep delivering.
    plan = RulesPlanner().plan("Put the red block in the box.", _obs(), [])
    assert _plan_text(plan) == _FULL


def test_move_block_to_box_unchanged():
    # Guard: bare move-to-box already delivered; it must not change.
    plan = RulesPlanner().plan("Move the red block to the box.", _obs(), [])
    assert _plan_text(plan) == _FULL


def test_take_block_to_blue_block_unchanged():
    # Guard: a block is not a box destination, so the adjective slot must not
    # claim it: this stays a pickup plan, exactly as before.
    plan = RulesPlanner().plan("Take the red block to the blue block.", _obs(), [])
    assert _plan_text(plan) == [
        ("walk_toward", {"target": "red_block"}),
        ("grab", {}),
    ]
