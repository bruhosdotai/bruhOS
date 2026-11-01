"""A fall the body does not confirm must not re-run the same skill forever.

``Body.act`` returns when a primitive finishes and the skill reads the IMU right after it; the
runtime reads it again once the skill returns. Those are two different samples, so a body can
report ``fallen`` to the skill and ``upright`` to the runtime -- an MPU-6050 saturating during a
fast action group does exactly that for one sample. The runtime resumed the interrupted skill in
that case without anything bounding the retries, so a body that spikes on every primitive never
ended the episode: no end reason, no sealed trace, the same primitive running forever.
"""

from bruhos.body.base import Body
from bruhos.brain.base import Planner
from bruhos.runtime import Runtime
from bruhos.schemas import Imu, Observation, SkillCall


class SpikeBody(Body):
    """IMU reads 'fallen' for the sample right after a primitive, then reads upright again."""

    name = "spike"

    def __init__(self):
        self.spike = False
        self.acts: list[str] = []

    def observe(self):
        fallen, self.spike = self.spike, False
        return Observation(t=0, frame=0, imu=Imu(fallen=fallen))

    def act(self, primitive):
        self.acts.append(primitive)
        self.spike = True
        return True

    def halt(self):
        pass


class FloorBody(Body):
    """Falls on the wave and stays down until a stand_up primitive is run."""

    name = "floor"

    def __init__(self):
        self.fallen = False
        self.acts: list[str] = []

    def observe(self):
        return Observation(t=0, frame=0, imu=Imu(fallen=self.fallen, fell_forward=True if self.fallen else None))

    def act(self, primitive):
        self.acts.append(primitive)
        if primitive == "wave":
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


def test_unconfirmed_fall_ends_the_episode_instead_of_retrying_forever():
    body = SpikeBody()
    ep = Runtime(body, FixedPlanner([SkillCall("wave")])).run("Wave hello.")
    assert body.acts == ["wave", "wave"]          # one run, one bounded replan, then stop
    assert not ep.success
    assert ep.end_reason == "wave failed: wave reported a fall the body does not confirm"
    assert ep.falls == 2
    assert ep.trace_root and all(s.hash for s in ep.steps)


def test_confirmed_fall_still_recovers_and_resumes():
    body = FloorBody()
    ep = Runtime(body, FixedPlanner([SkillCall("wave")])).run("Wave hello.")
    assert body.acts[1] == "stand_up_front"       # the fall was real: it stands up
    assert ep.recoveries == 3                     # ... until the recovery budget is spent
    assert ep.falls == 4
    assert ep.end_reason == "fallen, recovery budget exhausted"

