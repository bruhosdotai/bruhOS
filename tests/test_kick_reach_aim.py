"""Kick the ball when the ball is already close but off-axis.

walk_toward used to declare "reached" on distance alone, before any aim
check. A ball sitting inside the stop distance but outside the kick cone
(30 deg in the sim) was therefore never touched, while walk_toward, the
kick skill and the whole episode all reported success. The shipped eval
expects "Kick the ball." to produce a kicked:yellow_ball event; these
placements are the same instruction in the same world, just with the
ball starting close and off to the side.
"""

import math

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime, RuntimeConfig
from bruhos.sim import SimBody, World, WorldObject


def run_kick(ball_xy):
    world = World(objects=[WorldObject("ball", "yellow", *ball_xy),
                           WorldObject("box", None, 0.9, 0.0)])
    body = SimBody(world)
    ep = Runtime(body, RulesPlanner(), config=RuntimeConfig()).run("Kick the ball.")
    return world, ep


def ball_at(dist, bearing_deg):
    a = math.radians(bearing_deg)
    return (dist * math.cos(a), dist * math.sin(a))


def test_kick_connects_with_close_ball_at_33_degrees():
    world, ep = run_kick(ball_at(0.11, 33.0))
    assert "kicked:yellow_ball" in world.events, world.events
    assert ep.end_reason == "completed"


def test_kick_connects_with_close_ball_found_by_scan_at_35_degrees():
    # Scan stops turning at the first in-FOV bearing (<=35 deg); with the
    # ball 0.11 m away walk_toward must aim before calling it reached.
    world, ep = run_kick(ball_at(0.11, 95.0))
    assert "kicked:yellow_ball" in world.events, world.events
    assert ep.end_reason == "completed"


def test_kick_connects_with_close_ball_just_outside_the_cone():
    world, ep = run_kick((0.10, 0.06))  # dist 0.117, bearing ~31 deg
    assert "kicked:yellow_ball" in world.events, world.events
    assert ep.end_reason == "completed"


def test_kick_still_connects_dead_ahead():
    world, ep = run_kick((0.117, 0.0))
    assert "kicked:yellow_ball" in world.events, world.events
    assert ep.end_reason == "completed"
