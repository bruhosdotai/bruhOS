"""Putting "it" in the box must not deliver a different held object when "it" has a named antecedent.

Failing-test-first for a reproduced defect: the rules planner's into_box
branch resolves a bare "it" to whatever the hand holds, without checking the
object named earlier in the same sentence. So "Pick up the red block and put
it in the box." while holding ``blue_block`` plans [walk_toward(box), release]
and reports ``success=True`` with event ``in_box:blue_block`` -- the wrong
object delivered, even though "it" refers to the red block, not the held blue
one. The planner knows ``obs.holding``; a named antecedent that cannot match
it is unsatisfiable and must yield no plan, the same honest outcome as the
named put-in-box mismatch (open #232) and the named drop mismatch (#75).
A bare "it" with no named antecedent still resolves to the held object, and a
matching antecedent (including a partial name such as "red") still takes the
shortcut, as does the empty-handed full pickup plan.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody, World


def _obs(holding=None):
    return Observation(t=0.0, frame=0, objects=[], holding=holding)


def _plan_text(plan):
    return [(c.skill, dict(c.args)) for c in plan]


_SHORTCUT = [("walk_toward", {"target": "box"}), ("release", {})]

_FULL = [
    ("detect", {"target": "red_block"}),
    ("walk_toward", {"target": "red_block"}),
    ("grab", {}),
    ("walk_toward", {"target": "box"}),
    ("release", {}),
]


def test_put_it_with_mismatched_antecedent_has_no_plan():
    plan = RulesPlanner().plan("Pick up the red block and put it in the box.", _obs("blue_block"), [])
    assert _plan_text(plan) == []


def test_put_it_with_matching_antecedent_uses_shortcut():
    plan = RulesPlanner().plan("Pick up the red block and put it in the box.", _obs("red_block"), [])
    assert _plan_text(plan) == _SHORTCUT


def test_put_it_with_partial_antecedent_mismatch_has_no_plan():
    plan = RulesPlanner().plan("Pick up the red one and put it in the box.", _obs("blue_block"), [])
    assert _plan_text(plan) == []


def test_put_it_with_partial_antecedent_match_uses_shortcut():
    plan = RulesPlanner().plan("Pick up the red one and put it in the box.", _obs("red_block"), [])
    assert _plan_text(plan) == _SHORTCUT


def test_put_it_with_mismatched_antecedent_delivers_nothing():
    world = World.tabletop()
    world.holding = world.get("blue_block")
    body = SimBody(world)
    ep = Runtime(body, RulesPlanner()).run("Pick up the red block and put it in the box.")
    assert not ep.success
    assert ep.end_reason == "no plan for instruction"
    assert ep.steps == []
    assert "in_box:blue_block" not in body.world.events
    assert body.world.holding is not None and body.world.holding.name == "blue_block"


def test_bare_put_it_still_releases_held_object():
    # No named antecedent: "it" is the held object, unchanged.
    plan = RulesPlanner().plan("Put it in the box.", _obs("blue_block"), [])
    assert _plan_text(plan) == _SHORTCUT


def test_put_it_with_antecedent_empty_hand_unchanged():
    # Nothing held: the full pickup-and-delivery plan is still emitted.
    plan = RulesPlanner().plan("Pick up the red block and put it in the box.", _obs(None), [])
    assert _plan_text(plan) == _FULL
