"""The rules planner must not plan the action a negated instruction forbids.

``RulesPlanner`` matched command keywords and never looked at negation, so
"Do not kick the ball." planned — and, run end to end, performed — exactly
the kick it forbade; "Don't stop." even planned ``stop``. The planner's
grammar is single imperative commands and cannot represent "do X but not
Y", so a negated instruction gets the house answer for a request that must
not be executed: no plan, the same answer the planner gives for "Do a
backflip." (the eval scores that as ``no_plan``). The markers are the three
negation spellings of this grammar: "do not" / "not", "never", and "don't"
(which the planner's normalisation turns into the tokens "don t").
"""

import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Observation, SkillCall
from bruhos.sim.kinematic import SimBody


def plan(text, holding=None):
    return RulesPlanner().plan(text, Observation(t=0.0, frame=0, holding=holding), [])


@pytest.mark.parametrize("text", [
    "Don't pick up the red block.",
    "Do not pick up the red block.",
    "Do not kick the ball.",
    "Never kick the ball.",
    "Don't put the red block in the box.",
    "Don't stop.",
    "Do not stop.",
    "Never follow the line.",
    "Don't wave.",
    "Do not drop the red block.",
    "Don't walk to the box.",
])
def test_no_plan_for_a_negated_instruction(text):
    assert plan(text) == []


def test_negated_kick_does_not_kick_end_to_end():
    body = SimBody()
    ep = Runtime(body, RulesPlanner()).run("Do not kick the ball.")
    assert not ep.success
    assert ep.plan == []
    assert ep.end_reason == "no plan for instruction"
    assert ep.steps == []
    assert body.world.events == []


@pytest.mark.parametrize("text,expected", [
    ("Pick up the red block.",
     [SkillCall("walk_toward", {"target": "red_block"}), SkillCall("grab", {})]),
    ("Kick the ball.",
     [SkillCall("walk_toward", {"target": "ball", "stop_distance_m": 0.12}),
      SkillCall("kick", {"foot": "right"})]),
    ("Stop.",
     [SkillCall("stop", {})]),
    ("Put the red block in the box.",
     [SkillCall("detect", {"target": "red_block"}), SkillCall("walk_toward", {"target": "red_block"}),
      SkillCall("grab", {}), SkillCall("walk_toward", {"target": "box"}), SkillCall("release", {})]),
    ("What do you see?",
     [SkillCall("detect", {})]),
])
def test_unaffected_instructions_keep_their_plans(text, expected):
    assert plan(text) == expected
