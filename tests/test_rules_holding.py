"""The rules planner must not re-fetch an object the robot already holds.

Failing-test-first for a reproduced defect: with ``red_block`` already held,
``Put the red one in the box.`` and ``Put the red block in the box.`` planned a
fresh ``detect`` / ``walk_toward`` / ``grab`` sequence for the held object. Held
objects are excluded from observations, so ``detect`` spun a full scan and failed
with ``red not found after full scan`` (or ``red_block not found ...``), and the
episode failed even though delivering the held object only needed ``walk_toward``
to the box plus ``release``. The ``Put it in the box.`` phrasing already took that
shortcut; the named-object phrasing must too.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody, World


def _obs(holding):
    return Observation(t=0.0, frame=0, objects=[], holding=holding)


def _plan_text(plan):
    return [(c.skill, dict(c.args)) for c in plan]


def test_put_named_held_block_in_box_skips_pickup():
    plan = RulesPlanner().plan("Put the red block in the box.", _obs("red_block"), [])
    assert _plan_text(plan) == [
        ("walk_toward", {"target": "box"}),
        ("release", {}),
    ]


def test_put_partial_name_held_block_in_box_skips_pickup():
    # "the red one" resolves to the partial target "red", which the held
    # "red_block" still satisfies.
    plan = RulesPlanner().plan("Put the red one in the box.", _obs("red_block"), [])
    assert _plan_text(plan) == [
        ("walk_toward", {"target": "box"}),
        ("release", {}),
    ]


def test_put_other_block_in_box_while_holding_does_not_skip():
    # Holding a different object must not change the plan: the robot still has
    # to fetch the named block (and will fail at grab with a full hand, as before).
    plan = RulesPlanner().plan("Put the red block in the box.", _obs("blue_block"), [])
    assert [c.skill for c in plan] == ["detect", "walk_toward", "grab", "walk_toward", "release"]


def test_put_named_block_in_box_while_holding_nothing_unchanged():
    plan = RulesPlanner().plan("Put the red block in the box.", _obs(None), [])
    assert [c.skill for c in plan] == ["detect", "walk_toward", "grab", "walk_toward", "release"]


def test_runtime_delivers_held_block_to_box():
    world = World.tabletop()
    world.holding = world.get("red_block")
    body = SimBody(world)
    ep = Runtime(body, RulesPlanner()).run("Put the red block in the box.")
    assert ep.success
    assert "in_box:red_block" in world.events
