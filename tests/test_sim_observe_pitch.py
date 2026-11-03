from bruhos.sim.kinematic import SimBody, World


def test_sim_observe_pitch_when_fallen_with_unknown_direction():
    w = World(fallen=True, fell_forward=None)
    body = SimBody(w)
    obs = body.observe()
    assert obs.imu.fallen is True
    assert obs.imu.fell_forward is None
    # When fell_forward is None (unknown direction or non-pitch fall),
    # pitch_deg must not falsely claim an extreme backward tilt (-80.0 deg).
    assert obs.imu.pitch_deg == 0.0


def test_sim_observe_pitch_when_fallen_forward():
    w = World(fallen=True, fell_forward=True)
    body = SimBody(w)
    obs = body.observe()
    assert obs.imu.fallen is True
    assert obs.imu.fell_forward is True
    assert obs.imu.pitch_deg == 80.0


def test_sim_observe_pitch_when_fallen_backward():
    w = World(fallen=True, fell_forward=False)
    body = SimBody(w)
    obs = body.observe()
    assert obs.imu.fallen is True
    assert obs.imu.fell_forward is False
    assert obs.imu.pitch_deg == -80.0


def test_sim_observe_pitch_when_upright():
    w = World(fallen=False, fell_forward=None)
    body = SimBody(w)
    obs = body.observe()
    assert obs.imu.fallen is False
    assert obs.imu.fell_forward is None
    assert obs.imu.pitch_deg == 0.0
