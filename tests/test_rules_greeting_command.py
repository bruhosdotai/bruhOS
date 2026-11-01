"""A greeting must not hijack a concrete command into a wave.

Failing-test-first for a reproduced defect: the rules planner's wave
branch sat ahead of the drop/pickup/walk/look/find branches, so any
greeting word anywhere in the instruction claimed the whole plan. ``Hi,
walk to the box.`` planned only ``[wave]`` and reported ``success=True``
while the robot never walked; ``Hello, pick up the red block.`` likewise
waved and certified success with an empty hand. A greeting is not a
command: it should only plan a wave when nothing else in the instruction
is actionable. Earlier branches (stop, stand_up, into_box, kick, follow)
already beat the wave branch, so only the later branches were shadowed.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody, World


def _obs(holding=None):
    return Observation(t=0.0, frame=0, objects=[], holding=holding)


def _plan_text(plan):
    return [(c.skill, dict(c.args)) for c in plan]


def test_greeting_walk_to_box_plans_walk():
    plan = RulesPlanner().plan("Hi, walk to the box.", _obs(), [])
    assert _plan_text(plan) == [("walk_toward", {"target": "box"})]


def test_greeting_pickup_plans_pickup():
    plan = RulesPlanner().plan("Hello, pick up the red block.", _obs(), [])
    assert _plan_text(plan) == [
        ("walk_toward", {"target": "red_block"}),
        ("grab", {}),
    ]


def test_greeting_look_plans_look():
    plan = RulesPlanner().plan("Hi, look at the red block.", _obs(), [])
    assert _plan_text(plan) == [("look_at", {"target": "red_block"})]


def test_greeting_drop_plans_release():
    plan = RulesPlanner().plan("Hi, drop the red block.", _obs("red_block"), [])
    assert _plan_text(plan) == [("release", {})]


def test_greeting_find_plans_detect():
    plan = RulesPlanner().plan("Hi, find the red block.", _obs(), [])
    assert _plan_text(plan) == [("detect", {"target": "red_block"})]


def test_greeting_walk_episode_walks():
    body = SimBody(World.tabletop())
    ep = Runtime(body, RulesPlanner()).run("Hi, walk to the box.")
    assert ep.success, ep.end_reason
    assert body.world.x > 0.5
    assert "wave" not in body.world.events


def test_greeting_pickup_episode_picks_up():
    body = SimBody(World.tabletop())
    ep = Runtime(body, RulesPlanner()).run("Hello, pick up the red block.")
    assert ep.success, ep.end_reason
    assert body.world.holding is not None and body.world.holding.name == "red_block"


def test_pure_greetings_still_wave():
    # Guards: a greeting alone still waves.
    for text in ("Hi.", "Wave hello.", "Say hi to me.", "Greet me."):
        assert _plan_text(RulesPlanner().plan(text, _obs(), [])) == [("wave", {})]


def test_greeting_earlier_branches_unchanged():
    # Guards: branches that already beat wave are untouched.
    assert _plan_text(RulesPlanner().plan("Hi, put the red block in the box.", _obs(), [])) == [
        ("detect", {"target": "red_block"}),
        ("walk_toward", {"target": "red_block"}),
        ("grab", {}),
        ("walk_toward", {"target": "box"}),
        ("release", {}),
    ]
    assert _plan_text(RulesPlanner().plan("Hi, kick the ball.", _obs(), [])) == [
        ("walk_toward", {"target": "ball", "stop_distance_m": 0.12}),
        ("kick", {"foot": "right"}),
    ]
    assert _plan_text(RulesPlanner().plan("Hi, stand up.", _obs(), [])) == [("stand_up", {})]
    assert _plan_text(RulesPlanner().plan("Hi, stop.", _obs(), [])) == [("stop", {})]


def test_greeting_undoable_command_falls_back_to_wave():
    # Documented limit: a greeting plus a command with no plan of its own
    # (turn in place has no skill) still waves on the greeting half.
    assert _plan_text(RulesPlanner().plan("Hi, turn left.", _obs(), [])) == [("wave", {})]
