"""A "next to the box" destination must not plan a delivery into the box.

Failing-test-first for a reproduced defect: the rules planner's ``into_box``
matcher treated the "to" inside "next to" as a delivery preposition, so
``Put the red block next to the box.`` planned the full delivery plan
(detect / walk / grab / walk to box / release) and reported success with the
block inside the box — the opposite of the spatial relation asked. The skill
set has no place-near primitive, so the honest outcome for a next-to
instruction is no plan, not a delivery. The locative reading ("take the block
that is next to the box") must keep working as a plain pickup.
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


def test_put_block_next_to_box_has_no_plan():
    plan = RulesPlanner().plan("Put the red block next to the box.", _obs(), [])
    assert _plan_text(plan) == []


def test_bring_block_next_to_box_has_no_plan():
    plan = RulesPlanner().plan("Bring the red block next to the box.", _obs(), [])
    assert _plan_text(plan) == []


def test_move_block_next_to_box_does_not_deliver():
    # "move" falls back to the pre-existing walk branch once into_box no
    # longer claims it — exactly the plan for a destination-less
    # "Move the red block." — so it must walk to the block only, with no
    # grab, no release, and no delivery. (Walk-branch semantics for move are
    # pre-existing and out of this fix's scope.)
    plan = RulesPlanner().plan("Move the red block next to the box.", _obs(), [])
    assert _plan_text(plan) == [
        ("walk_toward", {"target": "red_block"}),
    ]


def test_put_block_right_next_to_box_has_no_plan():
    plan = RulesPlanner().plan("Put the red block right next to the box.", _obs(), [])
    assert _plan_text(plan) == []


def test_put_block_next_to_box_runtime_does_not_deliver():
    body = SimBody(World.tabletop())
    ep = Runtime(body, RulesPlanner()).run("Put the red block next to the box.")
    assert not ep.success
    assert "in_box:red_block" not in body.world.events


def test_drop_block_next_to_box_releases_in_place():
    # "drop" likewise falls back to its pre-existing release branch once
    # into_box no longer claims it — the same plan as a destination-less
    # "Drop the red block." — never a delivery into the box.
    plan = RulesPlanner().plan("Drop the red block next to the box.", _obs("red_block"), [])
    assert _plan_text(plan) == [
        ("release", {}),
    ]


def test_move_block_next_to_box_runtime_does_not_deliver():
    body = SimBody(World.tabletop())
    ep = Runtime(body, RulesPlanner()).run("Move the red block next to the box.")
    assert "in_box:red_block" not in body.world.events


def test_put_block_in_box_unchanged():
    # Guard: a real delivery destination must keep delivering.
    plan = RulesPlanner().plan("Put the red block in the box.", _obs(), [])
    assert _plan_text(plan) == _FULL


def test_put_it_in_box_with_holding_unchanged():
    # Guard: the "it" shortcut for a real delivery is untouched.
    plan = RulesPlanner().plan("Put it in the box.", _obs("red_block"), [])
    assert _plan_text(plan) == [
        ("walk_toward", {"target": "box"}),
        ("release", {}),
    ]


def test_take_block_next_to_box_stays_pickup():
    # Guard: "the block next to the box" as a locative (which block to take)
    # must stay a plain pickup, exactly as before.
    plan = RulesPlanner().plan("Take the block next to the box.", _obs(), [])
    assert _plan_text(plan) == [
        ("walk_toward", {"target": "block"}),
        ("grab", {}),
    ]


def test_put_next_block_in_box_unchanged():
    # Guard: "next" elsewhere in the instruction must not break matching.
    plan = RulesPlanner().plan("Put the next block in the box.", _obs(), [])
    assert _plan_text(plan) == [
        ("detect", {"target": "block"}),
        ("walk_toward", {"target": "block"}),
        ("grab", {}),
        ("walk_toward", {"target": "box"}),
        ("release", {}),
    ]
