from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Imu, Observation, SkillCall
from bruhos.sim.kinematic import SimBody, World


def test_rules_plan_when_already_holding_named_target():
    planner = RulesPlanner()
    obs = Observation(t=0.0, frame=0, imu=Imu(), holding="red_block")

    # When the robot is already holding the target block, it should walk to the box and release
    # instead of trying to detect and grab what is already in its hand.
    plan = planner.plan("Put the red block in the box.", obs, [])
    assert plan == [SkillCall("walk_toward", {"target": "box"}), SkillCall("release", {})]

    plan_color = planner.plan("Put the red one in the box.", obs, [])
    assert plan_color == [SkillCall("walk_toward", {"target": "box"}), SkillCall("release", {})]

    plan_noun = planner.plan("Place the block in the box.", obs, [])
    assert plan_noun == [SkillCall("walk_toward", {"target": "box"}), SkillCall("release", {})]


def test_runtime_delivers_held_object_to_box():
    world = World.tabletop()
    world.holding = world.get("red_block")
    body = SimBody(world)

    ep = Runtime(body, RulesPlanner()).run("Put the red block in the box.")
    assert ep.success
    assert "in_box:red_block" in world.events
