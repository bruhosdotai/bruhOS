"""The rules planner must deliver the block for "fetch/deliver ... to the box".

"Fetch the red block to the box." and "Deliver the red block to the box." are
delivery requests, synonymous with "Bring the red block to the box." -- but
the planner only routed bring (and put/place/drop/move/carry/throw) into the
put-in-box branch. "fetch" and "deliver" fell through: "fetch" reaches only
the pick-up branch (open PR #212) or no plan at all, and "deliver" plans
nothing, so the robot never delivered the block to the box for an explicit
delivery instruction. "fetch" / "deliver" now join the delivery verbs, so the
plan walks to the block, grabs it, walks to the box and releases it.
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


def test_fetch_block_to_box_plans_delivery():
    for text in ("Fetch the red block to the box.", "Fetch the red block into the box."):
        plan = _plan(text)
        assert [c.skill for c in plan] == ["detect", "walk_toward", "grab", "walk_toward", "release"], text


def test_deliver_block_to_box_plans_delivery():
    plan = _plan("Deliver the red block to the box.")
    assert [c.skill for c in plan] == ["detect", "walk_toward", "grab", "walk_toward", "release"]


def test_fetch_block_to_box_runtime_delivers():
    for text in ("Fetch the red block to the box.", "Deliver the red block to the box."):
        body, ep = _run(text)
        assert ep.success, text
        assert ep.end_reason == "completed", text
        assert [c["skill"] for c in ep.plan] == ["detect", "walk_toward", "grab", "walk_toward", "release"], text
        assert "grab:red_block" in body.world.events, text
        assert "in_box:red_block" in body.world.events, text
        assert body.world.holding is None, text


def test_bring_unaffected():
    body, ep = _run("Bring the red block to the box.")
    assert ep.success
    assert "in_box:red_block" in body.world.events
