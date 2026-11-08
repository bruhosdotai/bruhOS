"""Putting a named object in the box while holding a different one must yield no plan.

Failing-test-first for a reproduced defect: the rules planner's into_box
branch has no guard for a held object that does NOT match the named target
(the bare drop/release branch got one in #75; the matching-held-object
shortcut is #41's open PR). When the instruction names object X but the
hand holds Y != X, it still emits [detect X, walk_toward X, grab,
walk_toward box, release]. On hardware bodies grab reports success even with
a full hand (TonyPi/eliza bridge set holding unconditionally on head), so
the plan walks to the box and releases Y into it — the wrong object
delivered and certified success=True. In the sim the grab fails honestly,
but only after wasted detect/walk cycles, and the episode ends blaming the
grab instead of the planner emitting an unsatisfiable plan. The planner
knows obs.holding; a named put-in-box that cannot match it is unsatisfiable
and must yield no plan, the same honest outcome as the bare-drop mismatch
(#75) and the ungraspable pickups (#199/#200).
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


def test_put_named_mismatched_block_in_box_has_no_plan():
    plan = RulesPlanner().plan("Put the red block in the box.", _obs("blue_block"), [])
    assert _plan_text(plan) == []


def test_put_named_matching_block_in_box_unchanged():
    # #41 owns the already-holding-the-named-object shortcut; on head the
    # full pickup plan is still emitted, and this fix must not change that.
    plan = RulesPlanner().plan("Put the red block in the box.", _obs("red_block"), [])
    assert _plan_text(plan) == _FULL


def test_put_partial_name_matching_block_in_box_unchanged():
    # "the red one" resolves to the partial target "red", which the held
    # "red_block" still satisfies, so the full plan is unchanged on head.
    plan = RulesPlanner().plan("Put the red one in the box.", _obs("red_block"), [])
    assert _plan_text(plan) == [
        ("detect", {"target": "red"}),
        ("walk_toward", {"target": "red"}),
        ("grab", {}),
        ("walk_toward", {"target": "box"}),
        ("release", {}),
    ]


def test_put_named_holding_nothing_unchanged():
    plan = RulesPlanner().plan("Put the red block in the box.", _obs(None), [])
    assert _plan_text(plan) == _FULL


def test_put_it_in_box_holding_other_unchanged():
    # "it" names no object, so whatever is held is the referent.
    plan = RulesPlanner().plan("Put it in the box.", _obs("blue_block"), [])
    assert _plan_text(plan) == [("walk_toward", {"target": "box"}), ("release", {})]


def test_put_named_mismatched_runtime_delivers_nothing():
    world = World.tabletop()
    world.holding = world.get("blue_block")
    body = SimBody(world)
    ep = Runtime(body, RulesPlanner()).run("Put the red block in the box.")
    assert not ep.success
    assert ep.end_reason == "no plan for instruction"
    assert ep.steps == []
    assert "in_box:blue_block" not in body.world.events
    assert body.world.holding is not None and body.world.holding.name == "blue_block"


class _HardwareLikeBody(SimBody):
    """Mimics TonyPi / eliza bridge on head: grab reports success even with
    a full hand (act() sets holding unconditionally, no full-hand guard)."""

    def act(self, primitive: str) -> bool:
        if primitive == "grab":
            self.world.events.append("grab:anyway")
            return True
        return super().act(primitive)


def test_put_named_mismatched_hardware_like_body_delivers_nothing():
    # On hardware-shaped bodies the pre-fix plan walked to the box and
    # released the wrong object with success=True; after the fix there is
    # no plan, so nothing is delivered at all.
    world = World.tabletop()
    world.holding = world.get("blue_block")
    body = _HardwareLikeBody(world)
    ep = Runtime(body, RulesPlanner()).run("Put the red block in the box.")
    assert not ep.success
    assert "in_box:blue_block" not in body.world.events
    assert body.world.holding is not None and body.world.holding.name == "blue_block"
