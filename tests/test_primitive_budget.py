from bruhos.schemas import SkillCall, Status
from bruhos.skills import SkillContext, run_skill
from bruhos.sim.kinematic import SimBody, World, WorldObject


class CountingBody(SimBody):
    def __init__(self, world):
        super().__init__(world)
        self.commands = []

    def act(self, primitive):
        self.commands.append(primitive)
        return super().act(primitive)


def test_detect_scan_respects_the_primitive_budget():
    body = CountingBody(World(objects=[]))
    result = run_skill(SkillContext(body, max_steps=2), SkillCall("detect", {"target": "box"}))
    assert result.status is Status.FAILED
    assert len(body.commands) == 2


def test_scan_and_walk_share_one_budget_and_next_call_gets_a_new_budget():
    body = CountingBody(World(objects=[WorldObject("box", None, 0, 0.5)]))
    context = SkillContext(body, max_steps=5)
    result = run_skill(context, SkillCall("walk_toward", {"target": "box"}))
    assert result.status is Status.FAILED
    assert len(body.commands) == 5
    result = run_skill(context, SkillCall("wave"))
    assert result.status is Status.OK
    assert body.commands[-1] == "wave"
