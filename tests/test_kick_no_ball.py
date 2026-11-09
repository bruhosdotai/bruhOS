"""A sim kick that reaches no ball must report that it could not run.

``SimBody._kick`` returns ``True`` even when no ball is within kick reach:
the ball never moves and no ``kicked:`` event is recorded, yet the body
claims the primitive ran. The body contract is "returns False if the body
could not run it", and ``_p_grab``/``_p_release`` already fail the same way
when there is nothing to grab or nothing held. A kick with no ball in reach
is the same case. (The skill-level report of a refused kick is covered by
open PRs #146/#162/#191; this is the sim half they do not touch — none of
them makes ``SimBody`` return ``False`` here.)
"""

from bruhos.sim.kinematic import SimBody


def test_sim_kick_returns_false_with_no_ball_in_reach():
    body = SimBody()  # yellow_ball sits 0.72 m away, past KICK_REACH_M (0.15)
    assert body.act("kick_right") is False
    assert "kick:right" not in body.world.events
    assert "kicked:yellow_ball" not in body.world.events
    assert (body.world.get("yellow_ball").x, body.world.get("yellow_ball").y) == (0.4, -0.6)


def test_sim_kick_still_succeeds_when_ball_in_reach():
    body = SimBody()
    ball = body.world.get("yellow_ball")
    ball.x, ball.y = 0.1, 0.0
    assert body.act("kick_right") is True
    assert "kick:right" in body.world.events
    assert "kicked:yellow_ball" in body.world.events
    assert ball.x > 0.1
