"""Fetch verbs in the rules planner.

"Bring me / Fetch / Give me / Hand me / Pass me the <object>" planned
nothing, while the same request phrased as "Pick up the <object>" planned
[walk_toward, grab]. A fetch is a pickup addressed to the speaker, so it
plans the same two skills. A box or line is not grabbable and a bare
"fetch it" names no target, so both still correctly plan nothing.
"""

from bruhos.brain import make_planner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody


def _plan(text):
    return make_planner("rules").plan(text, SimBody().observe(), [])


def _skills(text):
    return [(c.skill, dict(c.args)) for c in _plan(text)]


def test_fetch_verbs_plan_walk_and_grab():
    for text in (
        "Bring me the red block.",
        "Fetch the red block.",
        "Give me the red block.",
        "Hand me the red block.",
        "Pass me the blue ball.",
    ):
        assert _skills(text) == [
            ("walk_toward", {"target": "red_block" if "red" in text else "blue_ball"}),
            ("grab", {}),
        ], text


def test_fetch_without_grabbable_target_plans_nothing():
    assert _plan("Bring me the box.") == []
    assert _plan("Fetch the line.") == []
    assert _plan("Fetch it.") == []


def test_fetch_does_not_change_box_destination():
    assert [c.skill for c in _plan("Bring the red block to the box.")] == [
        "detect", "walk_toward", "grab", "walk_toward", "release",
    ]


def test_fetch_episode_ends_holding_the_block():
    body = SimBody()
    ep = Runtime(body, make_planner("rules")).run("Fetch the red block.")
    assert ep.success, ep.end_reason
    assert body.world.holding is not None and body.world.holding.name == "red_block"
