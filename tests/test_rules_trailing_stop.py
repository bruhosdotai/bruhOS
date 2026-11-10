"""A trailing stop clause must not hijack a concrete command into a bare stop.

Failing-test-first for a reproduced defect: the rules planner's stop branch
sits first, so any instruction containing a stop word anywhere claims the
whole plan. ``Walk to the box and stop.`` planned only ``[stop]`` and reported
``success=True`` while the robot never moved; ``Pick up the red block and
stop.`` likewise halted and certified success with an empty hand. A stop
word at the end of a compound instruction (``... and stop`` / ``... then
stop``) is a finalizer, not the whole command: the robot must do the work
first and stop last. A bare ``Stop.`` still plans only ``[stop]``.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody, World


def _obs(holding=None):
    return Observation(t=0.0, frame=0, objects=[], holding=holding)


def _plan_text(plan):
    return [(c.skill, dict(c.args)) for c in plan]


def test_trailing_stop_walk_plans_walk_then_stop():
    plan = RulesPlanner().plan("Walk to the box and stop.", _obs(), [])
    assert _plan_text(plan) == [
        ("walk_toward", {"target": "box"}),
        ("stop", {}),
    ]


def test_trailing_stop_pickup_plans_pickup_then_stop():
    plan = RulesPlanner().plan("Pick up the red block and stop.", _obs(), [])
    assert _plan_text(plan) == [
        ("walk_toward", {"target": "red_block"}),
        ("grab", {}),
        ("stop", {}),
    ]


def test_trailing_then_stop_and_halt():
    plan = RulesPlanner().plan("Go to the green block then stop.", _obs(), [])
    assert _plan_text(plan) == [
        ("walk_toward", {"target": "green_block"}),
        ("stop", {}),
    ]
    plan = RulesPlanner().plan("Walk to the box and halt.", _obs(), [])
    assert _plan_text(plan) == [
        ("walk_toward", {"target": "box"}),
        ("stop", {}),
    ]


def test_bare_stop_still_plans_only_stop():
    assert _plan_text(RulesPlanner().plan("Stop.", _obs(), [])) == [("stop", {})]
    assert _plan_text(RulesPlanner().plan("Halt.", _obs(), [])) == [("stop", {})]


def test_trailing_stop_with_no_action_still_stops():
    # Nothing actionable before the finalizer: stopping is the safe answer.
    assert _plan_text(RulesPlanner().plan("Do a backflip and stop.", _obs(), [])) == [("stop", {})]


def test_walk_and_stop_actually_walks():
    body = SimBody()
    ep = Runtime(body, RulesPlanner()).run("Walk to the box and stop.")
    assert ep.success
    box = body.world.get("box")
    dist, _ = body.world.relative(box)
    assert dist <= 0.15, f"robot never walked to the box (dist {dist:.2f} m)"


def test_pickup_and_stop_actually_picks_up():
    body = SimBody(World.tabletop())
    ep = Runtime(body, RulesPlanner()).run("Pick up the red block and stop.")
    assert ep.success
    assert body.world.holding is not None and body.world.holding.name == "red_block"
