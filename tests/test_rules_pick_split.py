"""A split phrasal "pick X up" must plan the pickup like "pick up X".

Failing-test-first for a reproduced defect: the rules planner only
recognized "pick up" as a contiguous phrase, so "Pick the red block up."
planned nothing at all, and "Pick the red block up to the box." planned
nothing either — even though "pick up" is a claimed fetch verb and the
object and destination are unambiguous. The split form is ordinary
English for the same action; the planner must treat bare "pick" as a
fetch verb in both the pickup branch and the into_box gate (the gate
half matters: without it, a split pickup with a box destination would
fall through to a pickup-only plan that silently drops the delivery,
the same false-success shape PR #63 fixes for take/get).
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody, World


def _obs(holding=None):
    return Observation(t=0.0, frame=0, objects=[], holding=holding)


def _plan_text(plan):
    return [(c.skill, dict(c.args)) for c in plan]


_PICKUP = [
    ("walk_toward", {"target": "red_block"}),
    ("grab", {}),
]

_FULL = [
    ("detect", {"target": "red_block"}),
    ("walk_toward", {"target": "red_block"}),
    ("grab", {}),
    ("walk_toward", {"target": "box"}),
    ("release", {}),
]


def test_pick_split_block_up_plans_pickup():
    plan = RulesPlanner().plan("Pick the red block up.", _obs(), [])
    assert _plan_text(plan) == _PICKUP


def test_pick_split_partial_name_up_plans_pickup():
    plan = RulesPlanner().plan("Pick the red one up.", _obs(), [])
    assert _plan_text(plan) == [
        ("walk_toward", {"target": "red"}),
        ("grab", {}),
    ]


def test_pick_split_up_to_box_plans_delivery():
    plan = RulesPlanner().plan("Pick the red block up to the box.", _obs(), [])
    assert _plan_text(plan) == _FULL


def test_pick_split_up_to_blue_block_stays_pickup():
    # A non-box destination must not claim a delivery, exactly as for take.
    plan = RulesPlanner().plan("Pick the red block up to the blue block.", _obs(), [])
    assert _plan_text(plan) == _PICKUP


def test_pick_split_up_runtime_picks_up():
    body = SimBody(World.tabletop())
    ep = Runtime(body, RulesPlanner()).run("Pick the red block up.")
    assert ep.success, ep.end_reason
    assert body.world.holding is not None and body.world.holding.name == "red_block"


def test_pick_split_up_to_box_runtime_delivers():
    body = SimBody(World.tabletop())
    ep = Runtime(body, RulesPlanner()).run("Pick the red block up to the box.")
    assert ep.success, ep.end_reason
    assert "in_box:red_block" in body.world.events


def test_pick_up_unsplit_unchanged():
    # Guard: the contiguous form that already worked must keep working.
    plan = RulesPlanner().plan("Pick up the red block.", _obs(), [])
    assert _plan_text(plan) == _PICKUP


def test_pick_it_up_stays_no_plan():
    # Guard: "it" with no resolvable object is still no plan, as before.
    plan = RulesPlanner().plan("Pick it up.", _obs(), [])
    assert _plan_text(plan) == []
