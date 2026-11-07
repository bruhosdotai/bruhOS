"""release must fail when the hand is already empty.

The skill already treats an empty-hand release as a failure when the body
refuses it (the sim returns False with an empty hand). But bodies that play
a release action group without tracking the hand (TonyPi) return True, so
release reported success and the episode claimed it released None. The skill
observes the hand before it acts, so it can refuse without acting.
"""

from bruhos.body.base import Body
from bruhos.schemas import Imu, Observation, SkillCall, Status
from bruhos.skills import SkillContext, run_skill


class UntrackedHandBody(Body):
    """act() plays the mapped group and always reports success; it never checks the hand."""

    name = "untracked-hand"

    def __init__(self, holding=None):
        self.holding = holding
        self.acts = []

    def observe(self):
        return Observation(t=0, frame=0, objects=[], imu=Imu(), holding=self.holding)

    def act(self, primitive):
        self.acts.append(primitive)
        if primitive == "release":
            self.holding = None
        elif primitive == "grab":
            self.holding = "object"
        return True

    def halt(self):
        pass


def test_release_fails_when_hand_already_empty():
    body = UntrackedHandBody(holding=None)
    out = run_skill(SkillContext(body), SkillCall("release"))
    assert out.status is Status.FAILED
    assert "hand empty" in out.detail
    assert body.acts == []  # refuse without playing the release group


def test_release_still_succeeds_when_holding():
    body = UntrackedHandBody(holding="red_block")
    out = run_skill(SkillContext(body), SkillCall("release"))
    assert out.status is Status.OK
    assert out.data == {"released": "red_block"}
    assert body.acts == ["release"]
