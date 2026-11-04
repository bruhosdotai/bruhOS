"""Tests for routing line-walking instructions to follow_line in RulesPlanner."""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import SkillCall
from bruhos.sim.kinematic import SimBody, World


def test_walk_the_line_plans_follow_line():
    plan = RulesPlanner().plan("Walk the line.", SimBody().observe(), [])
    assert len(plan) == 1
    assert plan[0].skill == "follow_line"


def test_walk_along_the_line_plans_follow_line():
    plan = RulesPlanner().plan("Walk along the line.", SimBody().observe(), [])
    assert len(plan) == 1
    assert plan[0].skill == "follow_line"


def test_walk_on_the_line_plans_follow_line():
    plan = RulesPlanner().plan("Walk on the line.", SimBody().observe(), [])
    assert len(plan) == 1
    assert plan[0].skill == "follow_line"


def test_walk_the_tape_plans_follow_line():
    plan = RulesPlanner().plan("Walk the tape.", SimBody().observe(), [])
    assert len(plan) == 1
    assert plan[0].skill == "follow_line"


def test_walk_along_the_tape_plans_follow_line():
    plan = RulesPlanner().plan("Walk along the tape.", SimBody().observe(), [])
    assert len(plan) == 1
    assert plan[0].skill == "follow_line"


def test_track_and_trace_the_line_plans_follow_line():
    for text in ("Track the line.", "Trace the line."):
        plan = RulesPlanner().plan(text, SimBody().observe(), [])
        assert len(plan) == 1
        assert plan[0].skill == "follow_line"


def test_end_to_end_runtime_walk_the_line_success():
    w = World.tabletop(line_m=1.0)
    body = SimBody(w)
    ep = Runtime(body, RulesPlanner()).run("Walk the line.")
    assert ep.success is True
    assert ep.end_reason == "completed"
    assert [s.call["skill"] for s in ep.steps] == ["follow_line"]
    assert w.line_m == 0.0


def test_end_to_end_runtime_walk_along_the_line_success():
    w = World.tabletop(line_m=1.0)
    body = SimBody(w)
    ep = Runtime(body, RulesPlanner()).run("Walk along the line.")
    assert ep.success is True
    assert ep.end_reason == "completed"
    assert [s.call["skill"] for s in ep.steps] == ["follow_line"]
    assert w.line_m == 0.0


def test_navigation_to_discrete_objects_unaffected():
    for text, target in [
        ("Walk to the box.", "box"),
        ("Walk to the red block.", "red_block"),
        ("Go to the green block.", "green_block"),
    ]:
        plan = RulesPlanner().plan(text, SimBody().observe(), [])
        assert len(plan) == 1
        assert plan[0] == SkillCall("walk_toward", {"target": target})


def test_follow_the_line_unaffected():
    plan = RulesPlanner().plan("Follow the line.", SimBody().observe(), [])
    assert len(plan) == 1
    assert plan[0].skill == "follow_line"
