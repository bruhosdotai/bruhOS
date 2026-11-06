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

A colour prefix does not make the tape grabbable: ``_target`` synthesises
``red_line``/``blue_line``/... for "Pick up the red line.", and those
forms slipped past the bare-noun guard in BOTH grab-emitting branches —
the pick-up branch planned ``walk_toward(red_line) + grab``, and the
put-in-box branch, which had no line guard at all, planned the full
``detect + walk_toward + grab`` sequence on a target no body can find.
The guard therefore covers every ``*_line`` target in both branches.
The bare put-in-box fallback (an unqualified "line" there resolves to
the block, as the planner already had it) is unchanged.
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


@pytest.mark.parametrize("text", [
    "Pick up the red line.",
    "Grab the blue line.",
    "Take the green line.",
    "Get the yellow line.",
    "Hold the black line.",
])
def test_no_plan_for_grabbing_a_coloured_line(text):
    assert plan(text) == []


@pytest.mark.parametrize("text", [
    "Put the red line in the box.",
    "Put the blue line into the bin.",
    "Place the green line in the basket.",
])
def test_no_plan_for_putting_a_coloured_line_in_the_box(text):
    assert plan(text) == []


def test_no_plan_for_grabbing_the_line_end_to_end():
    body = SimBody()
    ep = Runtime(body, RulesPlanner()).run("Pick up the line.")
    assert not ep.success
    assert ep.plan == []
    assert ep.end_reason == "no plan for instruction"
    assert ep.steps == []
    assert body.world.events == []


@pytest.mark.parametrize("text", [
    "Pick up the red line.",
    "Put the red line in the box.",
])
def test_no_plan_for_a_coloured_line_end_to_end(text):
    body = SimBody()
    ep = Runtime(body, RulesPlanner()).run(text)
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
    ("Put the red block in the box.",
     [("detect", {"target": "red_block"}), ("walk_toward", {"target": "red_block"}),
      ("grab", {}), ("walk_toward", {"target": "box"}), ("release", {})]),
    # The bare put-in-box fallback predates this guard and is unchanged:
    # an unqualified "line" there still resolves to the block.
    ("Put the line in the box.",
     [("detect", {"target": "block"}), ("walk_toward", {"target": "block"}),
      ("grab", {}), ("walk_toward", {"target": "box"}), ("release", {})]),
])
def test_neighbouring_plans_unchanged(text, expected):
    assert plan_text(plan(text)) == expected
