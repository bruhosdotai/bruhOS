"""The rules planner must plan a pickup for retrieve/collect instructions.

"Retrieve the red block." and "Collect the red block." are pickup requests,
synonymous with "Fetch the red block." -- but the planner only routed fetch
(and take/get/grab/hold) into the pickup branch, so retrieve and collect
returned no plan and the episode ended "no plan for instruction" even though
the skill set can do exactly that: walk to the object and grab it. Both
verbs now join the pickup verbs. They name a return to the speaker, not a
delivery to the box, so they do not join the delivery verbs.
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


PICKUP = [
    SkillCall("walk_toward", {"target": "red_block"}),
    SkillCall("grab", {}),
]


def test_retrieve_collect_plan_pickup():
    for text in ("Retrieve the red block.", "Collect the red block.",
                 "Retrieve the blue block."):
        target = "blue_block" if "blue" in text else "red_block"
        assert _plan(text) == [
            SkillCall("walk_toward", {"target": target}),
            SkillCall("grab", {}),
        ], text


def test_retrieve_runtime_picks_up():
    body, ep = _run("Retrieve the red block.")
    assert ep.success, ep.end_reason
    assert [c["skill"] for c in ep.plan] == ["walk_toward", "grab"]
    assert body.world.holding is not None and body.world.holding.name == "red_block"


def test_collect_runtime_picks_up():
    body, ep = _run("Collect the blue block.")
    assert ep.success, ep.end_reason
    assert body.world.holding is not None and body.world.holding.name == "blue_block"
