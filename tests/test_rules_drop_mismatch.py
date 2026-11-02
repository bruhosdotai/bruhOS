"""Dropping a named object must not release a different held object.

Failing-test-first for a reproduced defect: the rules planner's
``drop``/``release`` branch returned ``[release]`` unconditionally, never
checking what the hand actually holds. So ``Drop the red block.`` while
holding ``blue_block`` released the blue block and reported
``success=True`` with event ``drop:blue_block`` — the wrong object
dropped, certified as success. The planner knows ``obs.holding``; a named
drop that cannot match it is unsatisfiable and must yield no plan, the
same honest outcome as a follow-object or next-to instruction. Dropping
``it`` (no named object) and dropping while holding nothing are
unchanged: the former releases whatever is held, the latter fails
honestly at runtime with "hand empty".
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody, World


def _obs(holding=None):
    return Observation(t=0.0, frame=0, objects=[], holding=holding)


def _plan_text(plan):
    return [(c.skill, dict(c.args)) for c in plan]


_FULL_INTO_BOX = [
    ("detect", {"target": "red_block"}),
    ("walk_toward", {"target": "red_block"}),
    ("grab", {}),
    ("walk_toward", {"target": "box"}),
    ("release", {}),
]


def test_drop_named_matching_block_releases():
    plan = RulesPlanner().plan("Drop the red block.", _obs("red_block"), [])
    assert _plan_text(plan) == [("release", {})]


def test_drop_named_mismatched_block_has_no_plan():
    plan = RulesPlanner().plan("Drop the red block.", _obs("blue_block"), [])
    assert _plan_text(plan) == []


def test_drop_partial_name_matching_block_releases():
    plan = RulesPlanner().plan("Drop the red one.", _obs("red_block"), [])
    assert _plan_text(plan) == [("release", {})]


def test_drop_generic_noun_matching_block_releases():
    plan = RulesPlanner().plan("Drop the block.", _obs("red_block"), [])
    assert _plan_text(plan) == [("release", {})]


def test_drop_it_unchanged():
    # "it" names no object, so whatever is held is the referent.
    plan = RulesPlanner().plan("Drop it.", _obs("blue_block"), [])
    assert _plan_text(plan) == [("release", {})]


def test_drop_named_holding_nothing_unchanged():
    # No holding to mismatch against: the release still fails honestly
    # at runtime with "hand empty", exactly as before.
    plan = RulesPlanner().plan("Drop the red block.", _obs(None), [])
    assert _plan_text(plan) == [("release", {})]


def test_release_named_mismatched_block_has_no_plan():
    plan = RulesPlanner().plan("Release the blue block.", _obs("red_block"), [])
    assert _plan_text(plan) == []


def test_drop_into_box_unchanged():
    # A box destination still goes through the into_box branch, not here.
    plan = RulesPlanner().plan("Drop the red block in the box.", _obs(None), [])
    assert _plan_text(plan) == _FULL_INTO_BOX


def test_drop_mismatch_runtime_drops_nothing():
    world = World.tabletop()
    world.holding = world.get("blue_block")
    body = SimBody(world)
    ep = Runtime(body, RulesPlanner()).run("Drop the red block.")
    assert not ep.success
    assert "drop:blue_block" not in body.world.events
    assert body.world.holding is not None and body.world.holding.name == "blue_block"


def test_drop_match_runtime_releases():
    world = World.tabletop()
    world.holding = world.get("red_block")
    body = SimBody(world)
    ep = Runtime(body, RulesPlanner()).run("Drop the red block.")
    assert ep.success, ep.end_reason
    assert "drop:red_block" in body.world.events
