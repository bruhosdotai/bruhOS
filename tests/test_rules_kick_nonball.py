"""Kicking a named non-ball object must plan nothing.

Failing-test-first for a reproduced defect: "Kick the red block." plans a
bare ``kick`` and reports success, while the block never moves -- the kick
skill only acts on a ball in reach, and the sim's kick only moves a ball.
The same silent success happens for "Kick the box." and for a named block
with a foot ("Kick the red block with your left foot."). A kick instruction
that names a non-ball target is unsatisfiable and must yield no plan, the
same honest outcome as picking up the ball (open #199) or the box (open
#200), which the hand cannot close on. A bare "Kick." and kicking the ball
are unchanged.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody, World


def _run(text, world=None):
    body = SimBody(world or World.tabletop())
    return body, Runtime(body, RulesPlanner()).run(text)


def test_kick_nonball_has_no_plan():
    for text in (
        "Kick the red block.",
        "Kick the box.",
        "Kick the red block with your left foot.",
        "Kick the blue block.",
    ):
        body, ep = _run(text)
        assert not ep.plan, text
        assert ep.end_reason == "no plan for instruction", text
        assert not ep.success, text
        assert "kick:right" not in body.world.events and "kick:left" not in body.world.events, text


def test_kick_ball_is_unchanged():
    for text in ("Kick the ball.", "Kick the ball with your left foot."):
        body, ep = _run(text)
        assert ep.success, text
        assert "kicked:yellow_ball" in body.world.events, text


def test_bare_kick_is_unchanged():
    _body, ep = _run("Kick.")
    assert ep.success
    assert [c["skill"] for c in ep.plan] == ["kick"]
