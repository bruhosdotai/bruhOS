"""A fall the skill itself did not see must still be recovered before the next skill.

The runtime takes an observation after every skill (``after``) but only acted on it when the
skill itself returned ``FALLEN``. A skill that finishes its primitive and returns ``OK`` without
reading the IMU again -- ``release`` never does -- hides a fall that happened during that
primitive: no ``stand_up``, ``falls`` stays at 0, and an episode whose plan ends there seals as
``success: true`` with the robot on the floor.
"""

from bruhos.body.base import Body
from bruhos.brain.base import Planner
from bruhos.runtime import Runtime
from bruhos.schemas import Imu, Observation, SkillCall


class DropBody(Body):
    """The release action group knocks the robot over; the primitive itself still returns True."""

    name = "drop"

    def __init__(self):
        self.fallen = False
        self.holding = "red_block"
        self.acts: list[str] = []

    def observe(self):
        return Observation(t=0, frame=0, holding=self.holding,
                           imu=Imu(fallen=self.fallen, fell_forward=True if self.fallen else None))

    def act(self, primitive):
        self.acts.append(primitive)
        if primitive == "release":
            self.holding = None
            self.fallen = True
        elif primitive.startswith("stand_up"):
            self.fallen = False
        return True

    def halt(self):
        pass


class FixedPlanner(Planner):
    name = "fixed"

    def __init__(self, plan):
        self._plan = plan

    def plan(self, instruction, obs, catalog):
        return list(self._plan)


def test_fall_after_a_successful_skill_is_recovered_and_counted():
    body = DropBody()
    ep = Runtime(body, FixedPlanner([SkillCall("release")])).run("Drop it.")
    assert body.acts == ["release", "stand_up_front"]   # the fall is answered, not ignored
    assert ep.falls == 1 and ep.recoveries == 1
    assert ep.success and not body.fallen


def test_next_skill_never_runs_on_a_fallen_robot():
    body = DropBody()
    ep = Runtime(body, FixedPlanner([SkillCall("release"), SkillCall("wave")])).run("Drop it and wave.")
    assert body.acts == ["release", "stand_up_front", "wave"]
    assert ep.falls == 1 and ep.success and not body.fallen

