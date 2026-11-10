"""The rules planner must act on send/pass instructions.

"Send the red block to the box." and "Pass the red block to the box." are
delivery requests, synonymous with "Bring the red block to the box." -- but
the planner only routed bring (and put/place/drop/move/carry/throw) into the
put-in-box branch, so send and pass returned no plan and the robot never
delivered the block. Without a box destination ("Send me the red block.",
"Pass me the red block.") they are pickup requests, synonymous with "Give me
the red block." -- but they fell through every branch too. Both verbs now
join the delivery verbs with a box destination and the pickup verbs without
one, so the plan walks to the block, grabs it, and either releases it into
the box or holds it.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation, SkillCall
from bruhos.sim.kinematic import SimBody, World


def _plan(text, holding=None):
    return RulesPlanner().plan(text, Observation(t=0.0, frame=0, holding=holding), [])


def _run(text, world=None):
    body = SimBody(world or World.tabletop())
    return body, Runtime(body, RulesPlanner()).run(text)


DELIVERY = [
    SkillCall("detect", {"target": "red_block"}),
    SkillCall("walk_toward", {"target": "red_block"}),
    SkillCall("grab", {}),
    SkillCall("walk_toward", {"target": "box"}),
    SkillCall("release", {}),
]
PICKUP = [
    SkillCall("walk_toward", {"target": "red_block"}),
    SkillCall("grab", {}),
]


def test_send_pass_to_box_plan_delivery():
    for text in ("Send the red block to the box.", "Pass the red block to the box.",
                 "Send the red block into the box."):
        assert _plan(text) == DELIVERY, text


def test_send_pass_me_plan_pickup():
    for text in ("Send me the red block.", "Pass me the red block."):
        assert _plan(text) == PICKUP, text


def test_send_to_box_runtime_delivers():
    body, ep = _run("Send the red block to the box.")
    assert ep.success, ep.end_reason
    assert [c["skill"] for c in ep.plan] == ["detect", "walk_toward", "grab", "walk_toward", "release"]
    assert "grab:red_block" in body.world.events
    assert "in_box:red_block" in body.world.events
    assert body.world.holding is None


def test_pass_me_runtime_picks_up():
    body, ep = _run("Pass me the red block.")
    assert ep.success, ep.end_reason
    assert [c["skill"] for c in ep.plan] == ["walk_toward", "grab"]
    assert body.world.holding is not None and body.world.holding.name == "red_block"


def test_bring_unaffected():
    body, ep = _run("Bring the red block to the box.")
    assert ep.success
    assert "in_box:red_block" in body.world.events
