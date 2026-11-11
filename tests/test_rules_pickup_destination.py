"""A pick-up verb with a named non-box destination must plan nothing.

Failing-test-first for a reproduced defect: "Take the red block to the blue
block." plans ``walk_toward(red_block) > grab`` and reports success, while
the destination is silently dropped -- the block is picked up and never
moves toward the blue block. The same silent drop happens for "get", "grab",
"pick up" and "hold" with any destination preposition (to / toward(s) /
onto / on / at / next to) naming a non-box object. The robot has no skill
that places a held object onto or at another object: release drops it at
the robot's feet, and only the box is a real drop target (a box destination
still takes the delivery branch, which runs before the pick-up branch).
An unsatisfiable destination must yield no plan, the same honest outcome
as a bare "move <object>" (open #224) and a kick with a box destination
(open #245). A bare pick-up with no destination is unchanged.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody, World


def _run(text, world=None):
    body = SimBody(world or World.tabletop())
    return body, Runtime(body, RulesPlanner()).run(text)


def test_pickup_to_nonbox_destination_has_no_plan():
    for text in (
        "Take the red block to the blue block.",
        "Get the red block to the blue block.",
        "Grab the red block to the blue block.",
        "Pick up the red block and take it to the blue block.",
        "Take the red block toward the blue block.",
        "Take the red block onto the blue block.",
        "Take the red block next to the blue block.",
        "Take the red block to the yellow ball.",
    ):
        body, ep = _run(text)
        assert not ep.plan, text
        assert ep.end_reason == "no plan for instruction", text
        assert not ep.success, text
        assert "grab:red_block" not in body.world.events, text


def test_pickup_without_destination_still_picks_up():
    body, ep = _run("Take the red block.")
    assert ep.success
    assert [c["skill"] for c in ep.plan] == ["walk_toward", "grab"]
    assert body.world.holding is not None and body.world.holding.name == "red_block"


def test_pickup_to_box_is_unchanged():
    # A box destination is the delivery branch's territory (open #201 owns
    # take/get there); this guard must not swallow it.
    _body, ep = _run("Take the red block to the box.")
    assert [c["skill"] for c in ep.plan] == ["walk_toward", "grab"]
    assert ep.success
