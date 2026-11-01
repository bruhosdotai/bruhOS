"""A bare "move to the box" is navigation, not a delivery.

Failing-test-first for a reproduced defect: "move" is the only verb in both
the delivery list (put/place/drop/bring/move/carry/throw) and the navigation
list (walk/go/come/approach/move), and the into_box branch fires first. With
no object named, nothing held and no it/that deixis, the branch invented a
"block" target and planned a full fetch-and-deliver: the robot grabbed and
boxed a block the instruction never mentioned, while every parallel phrasing
("Walk to the box.", "Go to the box.", "Come to the box.") just walked.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody, World


def _obs(holding=None):
    return Observation(t=0.0, frame=0, objects=[], holding=holding)


def _plan_text(plan):
    return [(c.skill, dict(c.args)) for c in plan]


_NAV = [("walk_toward", {"target": "box"})]

_FULL = [
    ("detect", {"target": "red_block"}),
    ("walk_toward", {"target": "red_block"}),
    ("grab", {}),
    ("walk_toward", {"target": "box"}),
    ("release", {}),
]

_BLOCK_FALLBACK = [
    ("detect", {"target": "block"}),
    ("walk_toward", {"target": "block"}),
    ("grab", {}),
    ("walk_toward", {"target": "box"}),
    ("release", {}),
]


def test_bare_move_to_box_plans_navigation():
    plan = RulesPlanner().plan("Move to the box.", _obs(), [])
    assert _plan_text(plan) == _NAV


def test_bare_move_to_box_runtime_walks_and_moves_nothing():
    body = SimBody(World.tabletop())
    ep = Runtime(body, RulesPlanner()).run("Move to the box.")
    assert ep.success, ep.end_reason
    assert body.world.events == []


def test_move_named_block_to_box_still_delivers():
    # Guard: a named object is a delivery, exactly as before.
    plan = RulesPlanner().plan("Move the red block to the box.", _obs(), [])
    assert _plan_text(plan) == _FULL


def test_move_it_to_box_without_holding_keeps_block_fallback():
    # Guard: the " it " deixis keeps the existing unnamed-block fallback.
    plan = RulesPlanner().plan("Move it to the box.", _obs(), [])
    assert _plan_text(plan) == _BLOCK_FALLBACK


def test_move_that_to_box_keeps_block_fallback():
    # Guard: "that" is deixis too; the fallback must survive.
    plan = RulesPlanner().plan("Move that to the box.", _obs(), [])
    assert _plan_text(plan) == _BLOCK_FALLBACK


def test_bare_move_to_box_while_holding_keeps_holding_shortcut():
    # Guard: the held-object shortcut is untouched.
    plan = RulesPlanner().plan("Move to the box.", _obs(holding="red_block"), [])
    assert _plan_text(plan) == [("walk_toward", {"target": "box"}), ("release", {})]


def test_move_to_green_block_unchanged():
    # Guard: navigation to a non-box object was never hijacked.
    plan = RulesPlanner().plan("Move to the green block.", _obs(), [])
    assert _plan_text(plan) == [("walk_toward", {"target": "green_block"})]


def test_walk_to_box_unchanged():
    # Guard: the walk branch is untouched.
    plan = RulesPlanner().plan("Walk to the box.", _obs(), [])
    assert _plan_text(plan) == _NAV
