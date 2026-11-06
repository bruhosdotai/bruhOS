"""The rules planner must deliver the block for "take/get ... to the box".

"Take the red block to the box." and "Get the red block to the box." are
delivery requests, synonymous with "Bring the red block to the box." -- but
the planner only routed "bring" (and put/place/drop/move/carry/throw) into
the put-in-box branch. "take" and "get" fell through to the pick-up branch,
so the robot picked the block up, reported success, and never delivered it
to the box. "Take" / "get" now join the delivery verbs, so the plan walks
to the block, grabs it, walks to the box and releases it. A bare "Take the
red block." (no box) still plans pick-up only.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody, World


def _run(text, world=None):
    body = SimBody(world or World.tabletop())
    return body, Runtime(body, RulesPlanner()).run(text)


def test_take_block_to_box_delivers():
    for text in ("Take the red block to the box.", "Take the red block into the box.",
                 "Get the red block to the box."):
        body, ep = _run(text)
        assert ep.success, text
        assert ep.end_reason == "completed", text
        assert [c["skill"] for c in ep.plan] == ["detect", "walk_toward", "grab", "walk_toward", "release"], text
        assert "grab:red_block" in body.world.events, text
        assert "in_box:red_block" in body.world.events, text
        assert body.world.holding is None, text


def test_take_without_box_still_picks_up_only():
    body, ep = _run("Take the red block.")
    assert ep.success
    assert [c["skill"] for c in ep.plan] == ["walk_toward", "grab"]
    assert body.world.holding is not None and body.world.holding.name == "red_block"
    assert "in_box:red_block" not in body.world.events


def test_bring_unaffected():
    body, ep = _run("Bring the red block to the box.")
    assert ep.success
    assert "in_box:red_block" in body.world.events
