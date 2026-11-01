"""Release with an empty hand must fail on every body, not just the sim.

The sim refuses an empty-handed release, so the skill reports FAILED "hand empty".
Bodies that cannot sense holding (TonyPi / the eliza bridge run the release action
group and report success) instead got OK "released None" from the same skill, and
that false success was sealed into the trace. The skill already observes the hand
before acting, so it can enforce the sim's contract itself.
"""

from bruhos.body.base import Body
from bruhos.schemas import Imu, Observation, SkillCall, Status
from bruhos.sim.kinematic import SimBody, World
from bruhos.skills import SkillContext, run_skill


class GroupOnlyBody(Body):
    """Hardware-like body: the release group exists and always runs; holding is untracked."""

    name = "group-only"

    def __init__(self, holding=None, fallen=False):
        self.holding = holding
        self.fallen = fallen
        self.acts = []

    def observe(self):
        return Observation(t=0.0, frame=0, objects=[], imu=Imu(fallen=self.fallen),
                           holding=self.holding)

    def act(self, primitive):
        self.acts.append(primitive)
        if primitive == "release":
            self.holding = None
        return True

    def halt(self):
        pass


def test_release_with_empty_hand_fails_on_group_only_body():
    body = GroupOnlyBody(holding=None)
    out = run_skill(SkillContext(body), SkillCall("release"))
    assert out.status is Status.FAILED
    assert out.detail == "hand empty"
    assert "release" not in body.acts


def test_release_with_empty_hand_fails_on_sim():
    out = run_skill(SkillContext(SimBody()), SkillCall("release"))
    assert out.status is Status.FAILED
    assert out.detail == "hand empty"


def test_fallen_robot_with_empty_hand_still_reports_fallen():
    body = GroupOnlyBody(holding=None, fallen=True)
    out = run_skill(SkillContext(body), SkillCall("release"))
    assert out.status is Status.FALLEN


def test_release_while_holding_still_succeeds():
    world = World.tabletop()
    world.holding = world.get("red_block")
    out = run_skill(SkillContext(SimBody(world)), SkillCall("release"))
    assert out.status is Status.OK
    assert out.detail == "released red_block"
