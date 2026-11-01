import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody, World, WorldObject


def _plan(cmd):
    obs = SimBody().observe()
    return [c.to_dict() for c in RulesPlanner().plan(cmd, obs, [])]


@pytest.mark.parametrize("cmd,expected", [
    ("Scan for the red block", [{"skill": "detect", "args": {"target": "red_block"}}]),
    ("Scan the blue block", [{"skill": "detect", "args": {"target": "blue_block"}}]),
    ("Scan for the ball", [{"skill": "detect", "args": {"target": "ball"}}]),
    ("Scan the room", [{"skill": "detect", "args": {}}]),
    ("Scan", [{"skill": "detect", "args": {}}]),
    ("What do you see", [{"skill": "detect", "args": {}}]),
    ("What's around the red block", [{"skill": "detect", "args": {}}]),
    ("Look around for the red block", [{"skill": "look_at", "args": {"target": "red_block"}}]),
    ("Find the red block", [{"skill": "detect", "args": {"target": "red_block"}}]),
])
def test_scan_branch_plan(cmd, expected):
    assert _plan(cmd) == expected


def _no_red_world():
    return World(objects=[WorldObject("block", "blue", 0.6, 0.25),
                          WorldObject("box", None, 0.9, 0.0)])


def test_scan_for_absent_target_fails_honestly():
    ep = Runtime(SimBody(_no_red_world()), RulesPlanner()).run("Scan for the red block")
    assert ep.plan == [{"skill": "detect", "args": {"target": "red_block"}}] * 2  # initial plan + one replan
    assert not ep.success
    assert "red_block not found after full scan" in ep.end_reason


def test_scan_for_present_target_succeeds_with_target():
    ep = Runtime(SimBody(), RulesPlanner()).run("Scan for the red block")
    assert ep.plan == [{"skill": "detect", "args": {"target": "red_block"}}]
    assert ep.success, ep.end_reason
    assert ep.steps[0].outcome["detail"] == "found red_block"


def test_scan_without_target_still_reports_scene():
    ep = Runtime(SimBody(_no_red_world()), RulesPlanner()).run("Scan the room")
    assert ep.plan == [{"skill": "detect", "args": {}}]
    assert ep.success, ep.end_reason
    assert ep.steps[0].outcome["detail"] == "2 objects"
