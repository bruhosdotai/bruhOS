"""'Take a look/peek at X' is a look, not a fetch; 'take a picture' is no skill at all."""

import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation
from bruhos.sim.kinematic import SimBody, World
from bruhos.skills import CATALOG


def _plan(text):
    return [str(c) for c in RulesPlanner().plan(text, Observation(t=0.0, frame=0), CATALOG)]


@pytest.mark.parametrize("text", [
    "Take a look at the red block.",
    "Take a peek at the yellow ball.",
    "Get a look at the red block.",
    "Take a look at the box.",
])
def test_look_idiom_plans_look_at_not_grab(text):
    plan = _plan(text)
    assert plan and plan[0].startswith("look_at"), plan
    assert not any(c.startswith("grab") for c in plan)


@pytest.mark.parametrize("text", [
    "Take a picture of the red block.",
    "Take a photo of the box.",
])
def test_take_a_picture_has_no_plan(text):
    assert _plan(text) == []


def test_take_a_look_does_not_grab_end_to_end():
    body = SimBody(World.tabletop())
    ep = Runtime(body, RulesPlanner()).run("Take a look at the red block.")
    assert ep.success
    assert body.world.holding is None
    assert "grab:red_block" not in body.world.events


@pytest.mark.parametrize("text,expected", [
    ("Take a look at the red block.", ["look_at(target=red_block)"]),
    ("Look at the red block.", ["look_at(target=red_block)"]),
    ("Pick up the red block.", ["walk_toward(target=red_block)", "grab()"]),
    ("Take the red block.", ["walk_toward(target=red_block)", "grab()"]),
    ("Take a look around.", ["detect()"]),
])
def test_controls_unchanged(text, expected):
    assert _plan(text) == expected
