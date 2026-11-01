"""Putting a referent-less "it" (or a line) in the box must not invent a block.

Failing-test-first for a reproduced defect: the rules planner's into_box
branch fell back to target "block" whenever no object was named (a bare
pronoun "it", or the non-object "line") and the hand was empty. So
``Put it in the box.`` with an empty hand planned a full delivery of an
arbitrary block and reported ``success=True`` with events
``grab:red_block, in_box:red_block`` — even though "it" had no referent at
all: nothing was held and nothing was named. The honest parallel is
``Drop it.`` with an empty hand, which fails at runtime with "hand empty"
instead of inventing a target. A pronoun can only honestly resolve to what
the hand holds; with an empty hand and no named object there is nothing
to deliver, so the instruction must yield no plan. Named deliveries and
the held-object pronoun path are unchanged.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody, World


def _obs(holding=None):
    return Observation(t=0.0, frame=0, objects=[], holding=holding)


def _plan_text(plan):
    return [(c.skill, dict(c.args)) for c in plan]


def _full(target):
    return [
        ("detect", {"target": target}),
        ("walk_toward", {"target": target}),
        ("grab", {}),
        ("walk_toward", {"target": "box"}),
        ("release", {}),
    ]


def test_put_it_empty_hand_has_no_plan():
    plan = RulesPlanner().plan("Put it in the box.", _obs(), [])
    assert _plan_text(plan) == []


def test_bring_it_empty_hand_has_no_plan():
    plan = RulesPlanner().plan("Bring it to the box.", _obs(), [])
    assert _plan_text(plan) == []


def test_put_line_empty_hand_has_no_plan():
    # A line is not a grabbable object; delivering a block does not
    # fulfill "put the line in the box".
    plan = RulesPlanner().plan("Put the line in the box.", _obs(), [])
    assert _plan_text(plan) == []


def test_put_it_empty_hand_runtime_delivers_nothing():
    body = SimBody(World.tabletop())
    ep = Runtime(body, RulesPlanner()).run("Put it in the box.")
    assert not ep.success
    assert ep.end_reason == "no plan for instruction"
    assert body.world.events == []


def test_put_it_holding_unchanged():
    # Guard: "it" with something held resolves to the held object.
    plan = RulesPlanner().plan("Put it in the box.", _obs("red_block"), [])
    assert _plan_text(plan) == [
        ("walk_toward", {"target": "box"}),
        ("release", {}),
    ]


def test_put_named_block_unchanged():
    # Guard: a named block still plans the full delivery.
    plan = RulesPlanner().plan("Put the block in the box.", _obs(), [])
    assert _plan_text(plan) == _full("block")


def test_put_red_one_unchanged():
    # Guard: a named color still plans the full delivery.
    plan = RulesPlanner().plan("Put the red one in the box.", _obs(), [])
    assert _plan_text(plan) == _full("red")


def test_put_line_holding_unchanged():
    # Guard: the pre-existing pronoun-like fallback for a held object
    # is untouched; only the empty-hand invention is fixed.
    plan = RulesPlanner().plan("Put the line in the box.", _obs("blue_block"), [])
    assert _plan_text(plan) == [
        ("walk_toward", {"target": "box"}),
        ("release", {}),
    ]
