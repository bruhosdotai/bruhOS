"""kick and wave must not certify success when the body did not run the primitive.

``Body.act`` is contracted to return False "if the body could not run it"
(a halted body, a fallen body that only accepts stand_up, a failed send).
``grab`` and ``release`` already honor that result; ``kick`` and ``wave``
discarded it and returned OK unconditionally, so an episode could record a
successful kick or wave that never physically happened. On a halted SimBody
the reproduction is exact: no world event is recorded, yet the skill
reported "kicked left" / "waved".
"""

import pytest

from bruhos.schemas import SkillCall, Status
from bruhos.sim.kinematic import SimBody
from bruhos.skills import SkillContext, run_skill


@pytest.mark.parametrize("skill,args,prim", [
    ("kick", {"foot": "left"}, "kick_left"),
    ("kick", {"foot": "right"}, "kick_right"),
    ("wave", {}, "wave"),
])
def test_skill_fails_when_body_refuses(skill, args, prim):
    body = SimBody()
    body.halt()
    out = run_skill(SkillContext(body), SkillCall(skill, args))
    assert out.status is Status.FAILED
    assert prim in out.detail
    assert body.world.events == []  # the primitive really never ran


@pytest.mark.parametrize("skill,args,event", [
    ("kick", {"foot": "left"}, "kick:left"),
    ("wave", {}, "wave"),
])
def test_skill_still_succeeds_on_a_live_body(skill, args, event):
    body = SimBody()
    out = run_skill(SkillContext(body), SkillCall(skill, args))
    assert out.status is Status.OK
    assert event in body.world.events
