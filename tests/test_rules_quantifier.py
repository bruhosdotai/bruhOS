"""A quantifier in a put-in-box instruction must not plan a single delivery.

Failing-test-first for a reproduced defect: the rules planner silently dropped
quantifiers (all/every/both/each) in into_box instructions and planned one
delivery, which the runtime then reported as success even though the
instruction's scope was unmet:

- "Put all blocks in the box." planned a single block delivery and completed
  with only red_block in the box (blue and green left out).
- "Put all red blocks in the box." in a world with two red blocks delivered
  one and completed with the other still out.
- "Put all blocks in the box." while holding blue_block planned
  [walk_toward(box), release] and completed with only blue_block in the box.
- "Put all balls in the box." planned a block delivery (the plural "balls"
  is not a known noun, so the referent-less fallback invented "block") and
  completed with a block in the box and no ball moved.

The planner has no loop primitive, so a multi-object instruction cannot be
expressed as one delivery plan. The honest outcome is no plan, not a
single delivery reported as success. Non-into_box quantifier instructions
already yield no plan ("Pick up all blocks."), so this gate only fixes the
into_box path that overclaimed.

Boundary notes (deliberate, not oversights):
- This is not PR #80 (referent-less "it"/line + empty hand): the quantifier
  gate fires even when a referent exists ("all red blocks") or an object is
  held, cases #80's held shortcut would still deliver-and-succeed.
- Words containing a quantifier as a substring ("small", "ball", "peach")
  must not trip the gate: matching is whole-word via the planner's has().
- Idiomatic "after all" also yields no plan; that is the accepted cost of a
  word-based gate.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody, World, WorldObject


def _obs(holding=None):
    return Observation(t=0.0, frame=0, objects=[], holding=holding)


def _plan_text(plan):
    return [(c.skill, dict(c.args)) for c in plan]


_FULL_RED = [
    ("detect", {"target": "red_block"}),
    ("walk_toward", {"target": "red_block"}),
    ("grab", {}),
    ("walk_toward", {"target": "box"}),
    ("release", {}),
]


def test_quantifier_all_blocks_plans_nothing():
    assert RulesPlanner().plan("Put all blocks in the box.", _obs(), []) == []


def test_quantifier_every_blocks_plans_nothing():
    assert RulesPlanner().plan("Put every block in the box.", _obs(), []) == []


def test_quantifier_both_blocks_plans_nothing():
    assert RulesPlanner().plan("Put both blocks in the box.", _obs(), []) == []


def test_quantifier_each_block_plans_nothing():
    assert RulesPlanner().plan("Put each block in the box.", _obs(), []) == []


def test_quantifier_named_plural_plans_nothing():
    assert RulesPlanner().plan("Put all red blocks in the box.", _obs(), []) == []


def test_quantifier_all_balls_plans_nothing():
    assert RulesPlanner().plan("Put all balls in the box.", _obs(), []) == []


def test_quantifier_move_and_bring_plans_nothing():
    assert RulesPlanner().plan("Move all blocks to the box.", _obs(), []) == []
    assert RulesPlanner().plan("Bring all blocks to the box.", _obs(), []) == []
    assert RulesPlanner().plan("Put all of the blocks in the box.", _obs(), []) == []


def test_quantifier_held_still_plans_nothing():
    # Unlike the #80 held shortcut (deliver what is held for "it"), a
    # quantifier instruction must not deliver the held object and succeed.
    assert RulesPlanner().plan("Put all blocks in the box.", _obs(holding="blue_block"), []) == []


def test_substring_words_do_not_trip_gate():
    # "small" and "ball" contain "all"; whole-word matching must not fire.
    assert _plan_text(RulesPlanner().plan("Put the small block in the box.", _obs(), [])) == [
        ("detect", {"target": "block"}),
        ("walk_toward", {"target": "block"}),
        ("grab", {}),
        ("walk_toward", {"target": "box"}),
        ("release", {}),
    ]
    assert _plan_text(RulesPlanner().plan("Put the ball in the box.", _obs(), [])) == [
        ("detect", {"target": "ball"}),
        ("walk_toward", {"target": "ball"}),
        ("grab", {}),
        ("walk_toward", {"target": "box"}),
        ("release", {}),
    ]


def test_singular_delivery_still_planned():
    assert _plan_text(RulesPlanner().plan("Put the red block in the box.", _obs(), [])) == _FULL_RED
    assert _plan_text(RulesPlanner().plan("Put the block in the box.", _obs(), [])) == [
        ("detect", {"target": "block"}),
        ("walk_toward", {"target": "block"}),
        ("grab", {}),
        ("walk_toward", {"target": "box"}),
        ("release", {}),
    ]


def _run(text, world=None, holding=None):
    w = world or World.tabletop()
    if holding:
        w.holding = w.get(holding)
    ep = Runtime(SimBody(w), RulesPlanner()).run(text)
    return ep, w


def test_runtime_all_blocks_is_no_plan_not_success():
    ep, w = _run("Put all blocks in the box.")
    assert ep.plan == []
    assert ep.success is False
    assert ep.end_reason == "no plan for instruction"
    assert [o.name for o in w.objects if o.in_box] == []


def test_runtime_all_red_blocks_two_reds_is_no_plan():
    w = World(objects=[
        WorldObject("block", "red", 0.6, 0.25),
        WorldObject("block", "red", 0.7, -0.3),
        WorldObject("box", None, 0.9, 0.0),
    ], line_m=0.0)
    ep, w = _run("Put all red blocks in the box.", world=w)
    assert ep.plan == []
    assert ep.success is False
    assert [o.name for o in w.objects if o.in_box] == []


def test_runtime_all_blocks_held_is_no_plan():
    ep, w = _run("Put all blocks in the box.", holding="blue_block")
    assert ep.plan == []
    assert ep.success is False
    assert ep.end_reason == "no plan for instruction"


def test_runtime_singular_delivery_still_succeeds():
    ep, w = _run("Put the red block in the box.")
    assert ep.success is True
    assert [o.name for o in w.objects if o.in_box] == ["red_block"]
