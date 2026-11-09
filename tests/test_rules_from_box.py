"""Taking a named object out of the box must yield no plan.

Failing-test-first for a reproduced defect: the rules planner's pickup
branch plans a fetch for any named object, including one the instruction
says is inside the box (``Take the red block out of the box.``,
``Get the red block from the box.``). No skill can take an object out of
a container, and in-box objects never appear in observations, so the
planned ``walk_toward`` spins a full 360-degree scan, fails with
``red_block not found after full scan``, replans the same doomed fetch,
and fails again — the episode ends blaming the walk instead of the
planner emitting an unsatisfiable plan. The honest outcome is no plan,
the same as the other ungraspable pickups (#199/#200) and the mismatched
put-in-box (#232).
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody, World


def _obs(holding=None):
    return Observation(t=0.0, frame=0, objects=[], holding=holding)


def _plan_text(plan):
    return [(c.skill, dict(c.args)) for c in plan]


def test_take_out_of_the_box_has_no_plan():
    plan = RulesPlanner().plan("Take the red block out of the box.", _obs(), [])
    assert _plan_text(plan) == []


def test_get_from_the_box_has_no_plan():
    plan = RulesPlanner().plan("Get the red block from the box.", _obs(), [])
    assert _plan_text(plan) == []


def test_get_from_the_bin_has_no_plan():
    plan = RulesPlanner().plan("Get the blue block from the bin.", _obs(), [])
    assert _plan_text(plan) == []


def test_grab_out_of_the_basket_has_no_plan():
    plan = RulesPlanner().plan("Grab the green block out of the basket.", _obs(), [])
    assert _plan_text(plan) == []


def test_take_from_the_table_still_plans_a_pickup():
    # "from" alone is not the trigger: only a container source makes the
    # fetch unsatisfiable; an object on the table is fetchable.
    plan = RulesPlanner().plan("Take the red block from the table.", _obs(), [])
    assert _plan_text(plan) == [
        ("walk_toward", {"target": "red_block"}),
        ("grab", {}),
    ]


def test_take_named_block_still_plans_a_pickup():
    plan = RulesPlanner().plan("Take the red block.", _obs(), [])
    assert _plan_text(plan) == [
        ("walk_toward", {"target": "red_block"}),
        ("grab", {}),
    ]


def test_take_to_the_box_unchanged():
    # "take ... to the box" does not take the delivery branch on head
    # (that upgrade is open PRs #63/#201); the from-box guard must not
    # change it: "to the box" is a destination, not a container source.
    plan = RulesPlanner().plan("Take the red block to the box.", _obs(), [])
    assert _plan_text(plan) == [
        ("walk_toward", {"target": "red_block"}),
        ("grab", {}),
    ]


def test_take_from_box_runtime_plans_nothing():
    world = World.tabletop()
    world.get("red_block").in_box = True
    body = SimBody(world)
    ep = Runtime(body, RulesPlanner()).run("Take the red block out of the box.")
    assert not ep.success
    assert ep.end_reason == "no plan for instruction"
    assert ep.steps == []
    assert body.world.holding is None
    assert "grab:red_block" not in body.world.events


def test_take_from_box_runtime_leaves_boxed_block_untouched():
    world = World.tabletop()
    red = world.get("red_block")
    red.in_box = True
    body = SimBody(world)
    ep = Runtime(body, RulesPlanner()).run("Get the red block from the box.")
    assert not ep.success
    assert red.in_box
    assert body.world.holding is None
    # No wasted scan: the planner refused before any skill ran.
    assert body.frame == 1
    assert ep.steps == []
