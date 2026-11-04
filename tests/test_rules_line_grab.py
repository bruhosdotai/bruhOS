"""The rules planner must not plan to grab the line.

The line is the floor tape: it is not an object in any shipped body. The
sim's world objects are block/box/ball only (the tape is the scalar
``World.line_m``), and the TonyPi body's ``ColorBlobDetector`` labels every
detection "block" — the line reaches the planner only as ``obs.line``, read
by the ``follow_line`` skill. No ``Detection`` is ever labelled "line", so
``walk_toward(target=line)`` can never find it: on the base the robot burns
two full scans (48 locomotion acts, initial plan plus the one replan) and
the episode fails with "line not found after full scan". A request bruhOS
cannot perform must produce no plan, the same answer the planner gives for
"Do a backflip." (the eval scores that as ``no_plan``). The planner itself
already treats "line" as a non-referent in the put-in-box branch, and the
line stays a valid target for the skills that can act on it: walking to
it, facing it, detecting it and following it are unaffected.
"""

import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody


def plan(text, holding=None):
    return RulesPlanner().plan(text, Observation(t=0.0, frame=0, holding=holding), [])


def plan_text(calls):
    return [(c.skill, dict(c.args)) for c in calls]


@pytest.mark.parametrize("text", [
    "Pick up the line.",
    "Grab the line.",
    "Take the line.",
    "Get the line.",
    "Hold the line.",
])
def test_no_plan_for_grabbing_the_line(text):
    assert plan(text) == []


def test_no_plan_for_grabbing_the_line_end_to_end():
    body = SimBody()
    ep = Runtime(body, RulesPlanner()).run("Pick up the line.")
    assert not ep.success
    assert ep.plan == []
    assert ep.end_reason == "no plan for instruction"
    assert ep.steps == []
    assert body.world.events == []


@pytest.mark.parametrize("text,expected", [
    ("Pick up the red block.",
     [("walk_toward", {"target": "red_block"}), ("grab", {})]),
    ("Follow the line.",
     [("follow_line", {})]),
    ("Walk to the line.",
     [("walk_toward", {"target": "line"})]),
    ("Find the line.",
     [("detect", {"target": "line"})]),
    # A coloured line target is a different (colour, noun) pair, not the
    # bare tape noun; like the coloured-box target in the box guard, it is
    # out of scope here and its plan is unchanged.
    ("Pick up the red line.",
     [("walk_toward", {"target": "red_line"}), ("grab", {})]),
])
def test_neighbouring_plans_unchanged(text, expected):
    assert plan_text(plan(text)) == expected
