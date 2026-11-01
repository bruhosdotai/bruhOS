"""A refused primitive performs no step, so it must not advance simulated time.

On base, ``SimBody.act`` added ``STEP_S`` before checking the halted / fallen
guards, so a halted body (or a fallen body asked for anything but a stand-up)
reported ``False`` yet still moved its clock forward 0.4 s per refused call.
That inflates every later observation timestamp and trace step time for motion
that never happened. Unknown primitives are a separate open fix and are not
covered here.
"""

from bruhos.sim.kinematic import STEP_S, SimBody, World


def test_halted_act_does_not_advance_time():
    body = SimBody()
    body.halt()
    assert body.act("forward") is False
    assert body.t == 0.0
    assert body.observe().t == 0.0
    assert (body.world.x, body.world.y) == (0.0, 0.0)
    body.resume()
    assert body.act("forward") is True
    assert body.t == STEP_S


def test_fallen_act_does_not_advance_time_but_stand_up_does():
    world = World.tabletop()
    world.fallen, world.fell_forward = True, True
    body = SimBody(world)
    assert body.act("forward") is False
    assert body.t == 0.0
    assert body.observe().t == 0.0
    assert body.act("stand_up_front") is True
    assert body.t == STEP_S
    assert not body.world.fallen
