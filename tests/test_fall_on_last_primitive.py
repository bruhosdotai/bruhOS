"""A fall on the last primitive a skill is allowed to run must surface as FALLEN, not as a timeout.

``walk_toward`` and ``look_at`` check the IMU at the top of every loop iteration, so a fall caused by
the final primitive before the step budget runs out was returned as ``failed`` ("walk timeout" /
"aim timeout"). The runtime then treated it as an ordinary failure: it re-planned from a fallen
observation, or, with no replan budget left, ended the episode with the robot still on the floor and
``falls == recoveries == 0`` in the trace.
"""

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime, RuntimeConfig
from bruhos.schemas import SkillCall, Status
from bruhos.sim.kinematic import SimBody
from bruhos.skills import SkillContext, run_skill


def test_walk_toward_reports_fall_on_last_primitive():
    # The box is straight ahead at 0.9 m: five forward steps never reach it, and the fifth one falls.
    body = SimBody(falls_at={5})
    out = run_skill(SkillContext(body, max_steps=5), SkillCall("walk_toward", {"target": "box"}))
    assert body.world.fallen
    assert out.status is Status.FALLEN, out


def test_look_at_reports_fall_on_last_primitive():
    # The blue block is in view at -23 deg: the first (and only allowed) turn falls.
    body = SimBody(falls_at={1})
    out = run_skill(SkillContext(body, max_steps=1), SkillCall("look_at", {"target": "blue_block"}))
    assert body.world.fallen
    assert out.status is Status.FALLEN, out


def test_runtime_recovers_from_fall_on_last_primitive():
    body = SimBody(falls_at={5})
    cfg = RuntimeConfig(max_skill_steps=5, max_replans=0)
    ep = Runtime(body, RulesPlanner(), config=cfg).run("Walk to the box.")

    # The fall is counted, recovered from, and the robot is back on its feet when the episode ends.
    assert ep.falls == 1 and ep.recoveries == 1
    assert [s.kind for s in ep.steps[:2]] == ["skill", "recovery"]
    assert ep.steps[0].outcome["status"] == "fallen"
    assert not body.world.fallen
    # No skill step may end with the robot on the floor while claiming anything but a fall.
    for s in ep.steps:
        if s.kind == "skill" and s.obs_after["imu"]["fallen"]:
            assert s.outcome["status"] == "fallen", s.to_dict()
