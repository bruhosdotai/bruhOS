"""eval.check must judge near/facing at the resolution the body reports to the skill.

SimBody.observe rounds bearings to 0.01 degrees and distances to 1 mm. look_at accepts a reported
bearing of 8.0, so a true bearing of 8.003 degrees is a correctly finished skill, yet the eval
compared the unrounded world geometry against the same 8 degree limit and failed the prompt.
"""

from __future__ import annotations

import math

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.eval.runner import FACING_DEG, NEAR_M, check, run_eval
from bruhos.schemas import Episode
from bruhos.sim.kinematic import World

BALL_DEG = math.degrees(math.atan2(-0.6, 0.4))        # yellow_ball in World.tabletop()


def _episode(success: bool = True) -> Episode:
    return Episode("e", "t", "rules", "sim", 0.0, success=success)


def test_look_at_judged_facing_when_residue_is_below_observation_resolution():
    # start one 15 degree turn before a final true bearing of 8.003 degrees (reported as 8.0)
    heading = round((BALL_DEG - 8.003 - 15.0) % 360, 4)
    report = run_eval(RulesPlanner, prompts=[{"text": "Look at the yellow ball.", "expect": ["facing:yellow_ball"],
                                             "world": {"heading_deg": heading}}])
    r = report.results[0]
    assert r.end_reason == "completed", r
    assert r.passed, r.failed_checks


def test_check_rounds_like_the_body_but_keeps_the_limit():
    world = World.tabletop()
    world.x, world.y = 0.0, 0.0
    ball = world.get("yellow_ball")
    # facing: 8.004 deg true bearing reads 8.0 -> met; 8.006 reads 8.01 -> not met
    world.heading_deg = (BALL_DEG - (FACING_DEG + 0.004)) % 360
    assert check("facing:yellow_ball", _episode(), world)
    world.heading_deg = (BALL_DEG - (FACING_DEG + 0.006)) % 360
    assert not check("facing:yellow_ball", _episode(), world)
    # near: 150.4 mm reads 0.15 -> met; 150.6 mm reads 0.151 -> not met
    world.heading_deg = 0.0
    box = world.get("box")
    world.x, world.y = box.x - (NEAR_M + 0.0004), box.y
    assert check("near:box", _episode(), world)
    world.x = box.x - (NEAR_M + 0.0006)
    assert not check("near:box", _episode(), world)
    # a failed episode never counts as near/facing
    world.x = box.x - 0.05
    assert not check("near:box", _episode(success=False), world)
