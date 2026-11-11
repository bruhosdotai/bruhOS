"""The rules planner must deliver the block for "set/store ... to the box".

"Set the red block in the box." and "Store the red block in the box." are
delivery requests, synonymous with "Put the red block in the box." -- but
the planner only routed put/place/drop/bring/move/carry/throw (and, in open
PRs, fetch/deliver/take/get/give/hand/send/pass) into the put-in-box
branch. "set" and "store" fell through every branch, so the episode ended
"no plan for instruction" even though the skill set can do exactly that:
walk to the block, grab it, walk to the box and release it. Both verbs now
join the delivery verbs. "Set ... down" is untouched by this change (open
PR #184 separately teaches set-down to release) because the into-box
branch only fires with a box destination.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody, World


def _run(text, world=None):
    body = SimBody(world or World.tabletop())
    return body, Runtime(body, RulesPlanner()).run(text)


def _plan(text):
    return RulesPlanner().plan(text, Observation(t=0.0, frame=0), [])


def test_set_block_in_box_plans_delivery():
    plan = _plan("Set the red block in the box.")
    assert [c.skill for c in plan] == ["detect", "walk_toward", "grab", "walk_toward", "release"]


def test_store_block_in_box_plans_delivery():
    plan = _plan("Store the red block in the box.")
    assert [c.skill for c in plan] == ["detect", "walk_toward", "grab", "walk_toward", "release"]


def test_set_store_block_in_box_runtime_delivers():
    for text in ("Set the red block in the box.", "Store the red block in the box."):
        body, ep = _run(text)
        assert ep.success, (text, ep.end_reason)
        assert "in_box:red_block" in body.world.events, (text, body.world.events)


def test_set_down_is_not_a_delivery():
    # No box destination: "set down" must not turn into a delivery plan.
    # (Open PR #184 separately teaches set-down to release; on this base it
    # is still no plan, and this fix must not change that.)
    assert _plan("Set the red block down.") == []
