"""Walking an object to the box must not certify a walk to the object as success.

Failing-test-first for a reproduced defect: the rules planner's walk branch
claimed any walk/go/come/approach instruction with an object, even when the
instruction also named a box destination for that object. So ``Walk the red
block to the box.`` planned only ``[walk_toward(red_block)]`` and the episode
reported ``success=True`` with zero world events — the block never moved,
yet a to-the-box instruction was certified complete. Walking to an object
cannot transport it; only the into_box delivery plan (or no plan) is honest
for an object-plus-box-destination instruction.
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


def test_walk_block_to_box_has_no_plan():
    plan = RulesPlanner().plan("Walk the red block to the box.", _obs(), [])
    assert _plan_text(plan) == []


def test_go_block_to_box_has_no_plan():
    plan = RulesPlanner().plan("Go the red block to the box.", _obs(), [])
    assert _plan_text(plan) == []


def test_walk_block_to_bin_has_no_plan():
    # "bin" maps to the same box destination, so it must not be dropped either.
    plan = RulesPlanner().plan("Walk the red block to the bin.", _obs(), [])
    assert _plan_text(plan) == []


def test_walk_block_to_box_runtime_does_not_succeed():
    body = SimBody(World.tabletop())
    ep = Runtime(body, RulesPlanner()).run("Walk the red block to the box.")
    assert not ep.success
    assert ep.end_reason == "no plan for instruction"
    assert body.world.events == []


def test_walk_to_box_unchanged():
    # Guard: walking TO the box (destination == object) still works.
    plan = RulesPlanner().plan("Walk to the box.", _obs(), [])
    assert _plan_text(plan) == [("walk_toward", {"target": "box"})]


def test_go_to_box_unchanged():
    plan = RulesPlanner().plan("Go to the box.", _obs(), [])
    assert _plan_text(plan) == [("walk_toward", {"target": "box"})]


def test_walk_block_without_destination_unchanged():
    # Guard: a destination-less walk still walks to the object.
    plan = RulesPlanner().plan("Walk the red block.", _obs(), [])
    assert _plan_text(plan) == [("walk_toward", {"target": "red_block"})]


def test_move_block_to_box_unchanged():
    # Guard: "move" still takes the into_box delivery plan, not this branch.
    plan = RulesPlanner().plan("Move the red block to the box.", _obs(), [])
    assert _plan_text(plan) == _FULL
