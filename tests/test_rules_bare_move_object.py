"""A bare "move <object>" is a relocation, not navigation: it must plan nothing.

Failing-test-first for a reproduced defect: "move" is a delivery verb in the
into-box branch ("Move the red block to the box." fetches and delivers it), but
with no destination and no "to" the instruction fell through to the walk
branch, which treats every named object as a navigation target. "Move the red
block." therefore planned a lone walk_toward and sealed the episode as
success even though the block never moved -- the same false-success shape as
kicking a named non-ball (open PR #155) and walking an object to the box
(open PR #78). Navigation phrasings ("move to", "move toward(s)", "move
closer/near/next to") still walk to the object; only the bare direct-object
form plans nothing, matching the planner's stance for other underspecified
instructions.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody, World


def _obs(holding=None):
    return Observation(t=0.0, frame=0, objects=[], holding=holding)


def _plan_text(plan):
    return [(c.skill, dict(c.args)) for c in plan]


def test_bare_move_object_plans_nothing():
    assert RulesPlanner().plan("Move the red block.", _obs(), []) == []
    assert RulesPlanner().plan("Move the box.", _obs(), []) == []
    assert RulesPlanner().plan("Move the ball.", _obs(), []) == []


def test_move_to_object_still_navigates():
    assert _plan_text(RulesPlanner().plan("Move to the red block.", _obs(), [])) == [
        ("walk_toward", {"target": "red_block"}),
    ]
    assert _plan_text(RulesPlanner().plan("Move closer to the red block.", _obs(), [])) == [
        ("walk_toward", {"target": "red_block"}),
    ]
    assert _plan_text(RulesPlanner().plan("Move next to the red block.", _obs(), [])) == [
        ("walk_toward", {"target": "red_block"}),
    ]
    assert _plan_text(RulesPlanner().plan("Move towards the box.", _obs(), [])) == [
        ("walk_toward", {"target": "box"}),
    ]


def test_move_object_to_box_still_delivers():
    assert _plan_text(RulesPlanner().plan("Move the red block to the box.", _obs(), [])) == [
        ("detect", {"target": "red_block"}),
        ("walk_toward", {"target": "red_block"}),
        ("grab", {}),
        ("walk_toward", {"target": "box"}),
        ("release", {}),
    ]


def test_move_block_without_destination_is_not_certified_success():
    body = SimBody(World.tabletop())
    block = body.world.get("red_block")
    start = (block.x, block.y)
    ep = Runtime(body, RulesPlanner()).run("Move the red block.")
    assert not ep.success
    assert ep.end_reason == "no plan for instruction"
    assert ep.plan == []
    assert ep.steps == []
    assert (block.x, block.y) == start
    assert body.world.events == []


def test_move_to_block_still_walks_and_succeeds():
    body = SimBody(World.tabletop())
    ep = Runtime(body, RulesPlanner()).run("Move to the red block.")
    assert ep.success, ep.end_reason
    assert [c["skill"] for c in ep.plan] == ["walk_toward"]
    dist, _ = body.world.relative(body.world.get("red_block"))
    assert dist <= 0.15
