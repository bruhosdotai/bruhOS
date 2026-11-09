"""The LFM3 few-shot examples must be plans the robot can actually execute.

Each ``EXAMPLES`` pair is a worked example the ~2B model imitates. A plan that
cannot succeed in the sim teaches the model a dead end: "Bring the purple
ball to the box." used to plan ``detect > walk_toward > grab`` for a ball, but
the sim's grab primitive refuses the ball by construction (balls are kicked,
not carried — the rules planner already refuses to plan this, see the
``test_rules_ball_grab`` coverage), so the example episode walked to the ball
and failed the grab, then burned the one replan failing the identical way.
"""

import json

from bruhos.brain.adapters.lfm3_omni import EXAMPLES
from bruhos.brain.base import Planner, parse_plan
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody, World, WorldObject


class _FixedPlanner(Planner):
    name = "fixed"
    version = "0"

    def __init__(self, plan):
        self._plan = plan

    def plan(self, instruction, obs, catalog):
        return list(self._plan)


def _world_for(ask: dict, plan) -> World:
    """World matching the example's observation.

    Objects the ask lists sit straight ahead at their stated distance. A
    target the plan must find but the ask does not list as visible yet (the
    ball in the kick example) sits behind the robot, so the scan has to turn
    to find it, as on the real tabletop.
    """
    objs = []
    for o in ask["objects"]:
        name = o["name"]
        color, label = (name.split("_", 1) + [None])[:2] if "_" in name else (None, name)
        if "_" in name:
            color, label = name.split("_", 1)
        objs.append(WorldObject(label, color, o["distance_m"], 0.0))
    seen = {o["name"] for o in ask["objects"]}
    for call in plan:
        target = call.args.get("target")
        if target and target not in seen:
            color, label = target.split("_", 1) if "_" in target else (None, target)
            objs.append(WorldObject(label, color, -0.5, 0.0))
            seen.add(target)
    return World(objects=objs)


def _ask_and_instruction(pair):
    ask_text, plan_json = pair
    ask = json.loads(ask_text[len("Observation: "):ask_text.index("\nInstruction: ")])
    instruction = ask_text[ask_text.index("\nInstruction: ") + len("\nInstruction: "):]
    return ask, instruction, plan_json


def test_every_lfm3_example_executes_in_sim():
    for pair in EXAMPLES:
        ask, instruction, plan_json = _ask_and_instruction(pair)
        plan = parse_plan(plan_json)
        world = _world_for(ask, plan)
        ep = Runtime(SimBody(world), _FixedPlanner(plan)).run(instruction)
        if not plan:
            assert ep.end_reason == "no plan for instruction", instruction
            continue
        assert ep.success, f"{instruction!r}: {ep.end_reason}"


def test_no_lfm3_example_grabs_a_ball():
    for pair in EXAMPLES:
        _, _, plan_json = _ask_and_instruction(pair)
        plan = parse_plan(plan_json)
        for i, call in enumerate(plan):
            if call.skill == "grab":
                prev = [c.args.get("target") for c in plan[:i] if c.args.get("target")]
                assert not any(t == "ball" or t.endswith("_ball") for t in prev), (
                    f"grab of a ball in example plan {plan_json}"
                )
