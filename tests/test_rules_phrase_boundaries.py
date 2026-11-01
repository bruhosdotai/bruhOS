"""Multi-word command phrases must match as whole phrases, not substrings.

``RulesPlanner.has`` boundary-checks single words (``" stop " in s``) but matches
multi-word phrases with a bare substring test (``"get up" in s``). A longer word
that merely starts with a phrase therefore fires an unrelated skill: "Get updated"
stands the robot up, "Say high" waves, and "Open your handbook" releases whatever
the hand holds. Only exact phrases (with word boundaries on both sides) may match.
"""

import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody, World
from bruhos.schemas import Observation
from bruhos.skills import CATALOG


def _plan(text):
    return RulesPlanner().plan(text, Observation(t=0.0, frame=0), CATALOG)


def _held_world():
    world = World.tabletop()
    world.holding = world.get("red_block")
    return world


@pytest.mark.parametrize("instruction", [
    "Get updated",       # "get up" + "dated" -> stand_up
    "Get uploads",       # "get up" + "loads" -> stand_up
    "Stand upstairs",    # "stand up" + "stairs" -> stand_up
    "Say high",          # "say hi" + "gh" -> wave
    "Say history",       # "say hi" + "story" -> wave
    "Open your handbook",  # "open your hand" + "book" -> release
    "Open your handout",   # "open your hand" + "out" -> release
])
def test_glued_suffix_is_not_a_command(instruction):
    assert _plan(instruction) == []


def test_pickup_without_space_is_not_a_command():
    # Negative control shared with the whitespace-normalization fix: "Pickup"
    # is one word, not the phrase "pick up".
    assert _plan("Pickup the red block.") == []


def test_handbook_does_not_drop_a_held_object():
    world = _held_world()
    body = SimBody(world)
    ep = Runtime(body, RulesPlanner()).run("Open your handbook")
    assert ep.plan == []
    assert world.holding is not None and world.holding.name == "red_block"
    assert not [e for e in world.events if e.startswith(("drop:", "in_box:"))]


@pytest.mark.parametrize("instruction,skill", [
    ("Get up", "stand_up"),
    ("Get back up", "stand_up"),
    ("Say hi to me", "wave"),
    ("Let go", "release"),
    ("Open your hand", "release"),
])
def test_exact_phrases_still_plan(instruction, skill):
    plan = _plan(instruction)
    assert [c.skill for c in plan] == [skill]


def test_exact_pickup_still_runs():
    body = SimBody()
    ep = Runtime(body, RulesPlanner()).run("Pick up the red block.")
    assert ep.success, ep.end_reason
    assert body.world.holding is not None and body.world.holding.name == "red_block"


def test_exact_delivery_still_runs():
    body = SimBody()
    ep = Runtime(body, RulesPlanner()).run("Put the red block in the box.")
    assert ep.success, ep.end_reason
    assert "in_box:red_block" in body.world.events


def test_it_with_held_object_still_delivers():
    # Guards the " it " branch: stripping the phrase must not break it.
    world = _held_world()
    body = SimBody(world)
    ep = Runtime(body, RulesPlanner()).run("Put it in the box.")
    assert ep.success, ep.end_reason
    assert "in_box:red_block" in body.world.events
