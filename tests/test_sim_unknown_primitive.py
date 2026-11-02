from bruhos.sim.kinematic import SimBody


def test_unknown_primitive_is_refused_without_mutating_the_world():
    body = SimBody()
    before = body.observe()
    assert body.act("unknown_primitive") is False
    after = body.observe()
    assert after.t == before.t
    assert after.objects == before.objects
    assert body.world.events == []
    assert body.act("wave") is True
    assert body.world.events == ["wave"]
