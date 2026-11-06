"""Fetch verbs without a box destination must plan a pickup, not no plan.

``bring`` and ``carry`` are already verbs the planner understands — but only
in the into-box branch (\"Bring the red block to the box.\" is an eval prompt)
— and ``fetch`` is the plain synonym of the pickup verbs the planner does
know (\"take\", \"get\"). With no box destination, though, all three fell
through every branch and returned no plan, so \"Fetch the red block.\" ended
the episode with \"no plan for instruction\" even though the skill set can do
exactly that: walk to the object and grab it. The same verbs with a box
destination still take the delivery branch, which runs before the pickup
branch.
"""

import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation, SkillCall
from bruhos.sim.kinematic import SimBody, World


def plan(text, holding=None):
    return RulesPlanner().plan(text, Observation(t=0.0, frame=0, holding=holding), [])


@pytest.mark.parametrize("text", [
    "Bring me the red block.",
    "Bring the red block.",
    "Fetch the red block.",
    "Carry the red block.",
    "Fetch the blue block.",
])
def test_fetch_verbs_plan_a_pickup(text):
    assert plan(text) == [
        SkillCall("walk_toward", {"target": "red_block" if "red" in text else "blue_block"}),
        SkillCall("grab", {}),
    ]


def test_fetch_end_to_end_holds_the_block():
    body = SimBody(World.tabletop())
    ep = Runtime(body, RulesPlanner()).run("Fetch the red block.")
    assert ep.success
    assert body.world.holding is not None and body.world.holding.name == "red_block"
    assert "grab:red_block" in body.world.events


def test_bring_to_the_box_still_delivers():
    body = SimBody(World.tabletop())
    ep = Runtime(body, RulesPlanner()).run("Bring the red block to the box.")
    assert ep.success
    assert "in_box:red_block" in body.world.events
