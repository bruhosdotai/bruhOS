"""A fall injected mid-primitive must fail the primitive, not certify it.

``SimBody.act`` injects a fall while a locomotion primitive runs, then returns
``True`` even though the primitive never executed: a ``forward`` moves nowhere,
a ``kick_left`` records no kick event. The body contract is "returns False if
the body could not run it", and the bridge already returns ``False`` when a
walk is cut short (deadman); the sim must do the same for a fall, so callers
that trust the return value never treat a fall as a completed step.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody


def test_fall_during_forward_returns_false_and_moves_nowhere():
    body = SimBody(falls_at={1})
    assert body.act("forward") is False
    assert body.world.fallen is True
    assert (body.world.x, body.world.y) == (0.0, 0.0)
    assert "fell" in body.world.events


def test_fall_during_kick_returns_false_without_kick_event():
    body = SimBody(falls_at={1})
    assert body.act("kick_left") is False
    assert body.world.fallen is True
    assert "kick:left" not in body.world.events
    assert "fell" in body.world.events


def test_no_fall_still_returns_true_and_moves():
    body = SimBody()
    assert body.act("forward") is True
    assert body.world.fallen is False
    assert body.world.x == 0.05


def test_episode_still_recovers_when_the_failed_step_is_a_fall():
    # Skills detect the fall from the observation, not from act()'s return,
    # so the runtime recovery path is unchanged by the return value.
    body = SimBody(falls_at={4})
    ep = Runtime(body, RulesPlanner()).run("Put the red one in the box.")
    assert ep.falls == 1 and ep.recoveries == 1
    assert ep.success and "in_box:red_block" in body.world.events
