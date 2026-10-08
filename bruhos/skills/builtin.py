"""v0.1 whitelist: look_at detect walk_toward follow_line grab release kick wave stand_up stop."""

from __future__ import annotations

from bruhos.schemas import Observation, Outcome, Status

from .registry import Arg, SkillContext, SkillSpec, skill

AIM_DEG = 8.0
WALK_AIM_DEG = 12.0
LINE_DEADBAND = 0.2


def _fallen(obs: Observation) -> Outcome | None:
    if obs.imu.fallen:
        return Outcome(Status.FALLEN, "fall detected", {"fell_forward": obs.imu.fell_forward})
    return None


def _scan(ctx: SkillContext, target: str) -> tuple[Outcome | None, Observation]:
    """Turn in place until ``target`` is in view. One full revolution at most."""
    obs = ctx.body.observe()
    turns = int(360 / ctx.body.turn_step_deg)
    for _ in range(turns):
        if (f := _fallen(obs)):
            return f, obs
        if obs.find(target):
            return None, obs
        if ctx.should_stop():
            return Outcome(Status.STOPPED, "stop requested"), obs
        ctx.body.act("turn_left")
        obs = ctx.body.observe()
    if (f := _fallen(obs)):
        return f, obs
    if obs.find(target):
        return None, obs
    return Outcome(Status.FAILED, f"{target} not found after full scan"), obs


@skill(SkillSpec("detect", "Report visible objects; with `target`, scan until it is in view.",
                 {"target": Arg(str, doc="e.g. red_block, box, ball")}, positional="target"))
def detect(ctx: SkillContext, args: dict) -> Outcome:
    target = args.get("target")
    if not target:
        obs = ctx.body.observe()
        return _fallen(obs) or Outcome(Status.OK, f"{len(obs.objects)} objects",
                                       {"objects": [o.name for o in obs.objects]})
    err, obs = _scan(ctx, target)
    if err:
        return err
    d = obs.find(target)
    return Outcome(Status.OK, f"found {d.name}",
                   {"found": d.name, "bearing_deg": d.bearing_deg, "distance_m": d.distance_m})


@skill(SkillSpec("look_at", "Turn to face a target.",
                 {"target": Arg(str, required=True)}, positional="target"))
def look_at(ctx: SkillContext, args: dict) -> Outcome:
    target = args["target"]
    err, obs = _scan(ctx, target)
    if err:
        return err
    for _ in range(ctx.max_steps):
        if (f := _fallen(obs)):
            return f
        if ctx.should_stop():
            return Outcome(Status.STOPPED, "stop requested")
        d = obs.find(target)
        if d is None:
            return Outcome(Status.FAILED, f"lost {target}")
        if abs(d.bearing_deg) <= AIM_DEG:
            return Outcome(Status.OK, f"facing {d.name}", {"bearing_deg": d.bearing_deg})
        ctx.body.act("turn_left" if d.bearing_deg > 0 else "turn_right")
        obs = ctx.body.observe()
    return Outcome(Status.FAILED, "aim timeout")


@skill(SkillSpec("walk_toward", "Walk until the target is within reach.",
                 {"target": Arg(str, required=True),
                  "stop_distance_m": Arg(float, default=0.1)}, positional="target"))
def walk_toward(ctx: SkillContext, args: dict) -> Outcome:
    target, stop_at = args["target"], args["stop_distance_m"]
    err, obs = _scan(ctx, target)
    if err:
        return err
    for _ in range(ctx.max_steps):
        if (f := _fallen(obs)):
            return f
        if ctx.should_stop():
            return Outcome(Status.STOPPED, "stop requested")
        d = obs.find(target)
        if d is None:
            err, obs = _scan(ctx, target)
            if err:
                return err
            continue
        if d.distance_m is not None and d.distance_m <= stop_at:
            return Outcome(Status.OK, f"reached {d.name}", {"distance_m": d.distance_m})
        if abs(d.bearing_deg) > WALK_AIM_DEG:
            ctx.body.act("turn_left" if d.bearing_deg > 0 else "turn_right")
        else:
            ctx.body.act("forward")
        obs = ctx.body.observe()
    return Outcome(Status.FAILED, "walk timeout")


@skill(SkillSpec("follow_line", "Follow a floor line for up to `steps` gait cycles.",
                 {"steps": Arg(int, default=20)}, positional="steps"))
def follow_line(ctx: SkillContext, args: dict) -> Outcome:
    obs = ctx.body.observe()
    if (f := _fallen(obs)):
        return f
    if not obs.line.visible:
        return Outcome(Status.FAILED, "no line in view")
    done = 0
    while done < args["steps"]:
        if ctx.should_stop():
            return Outcome(Status.STOPPED, "stop requested")
        if obs.line.offset > LINE_DEADBAND:
            ctx.body.act("turn_right")
        elif obs.line.offset < -LINE_DEADBAND:
            ctx.body.act("turn_left")
        else:
            ctx.body.act("forward")
            done += 1
        obs = ctx.body.observe()
        if (f := _fallen(obs)):
            return f
        if not obs.line.visible:
            return Outcome(Status.OK, "end of line", {"steps": done})
    return Outcome(Status.OK, f"followed {done} steps", {"steps": done})


@skill(SkillSpec("grab", "Close the hand on the object in reach."))
def grab(ctx: SkillContext, args: dict) -> Outcome:
    if not ctx.body.act("grab"):
        obs = ctx.body.observe()
        return _fallen(obs) or Outcome(Status.FAILED, "nothing in reach" if not obs.holding else "hand full")
    obs = ctx.body.observe()
    return _fallen(obs) or Outcome(Status.OK, f"holding {obs.holding}", {"holding": obs.holding})


@skill(SkillSpec("release", "Open the hand."))
def release(ctx: SkillContext, args: dict) -> Outcome:
    before = ctx.body.observe().holding
    if not ctx.body.act("release"):
        obs = ctx.body.observe()
        return _fallen(obs) or Outcome(Status.FAILED, "hand empty")
    return Outcome(Status.OK, f"released {before}", {"released": before})


@skill(SkillSpec("kick", "Kick with the given foot.",
                 {"foot": Arg(str, default="right", choices=("left", "right"))}, positional="foot"))
def kick(ctx: SkillContext, args: dict) -> Outcome:
    ctx.body.act(f"kick_{args['foot']}")
    obs = ctx.body.observe()
    return _fallen(obs) or Outcome(Status.OK, f"kicked {args['foot']}")


@skill(SkillSpec("wave", "Wave hello."))
def wave(ctx: SkillContext, args: dict) -> Outcome:
    ctx.body.act("wave")
    obs = ctx.body.observe()
    return _fallen(obs) or Outcome(Status.OK, "waved")


@skill(SkillSpec("stand_up", "Recover from a fall (front or back is read from the IMU)."))
def stand_up(ctx: SkillContext, args: dict) -> Outcome:
    obs = ctx.body.observe()
    if not obs.imu.fallen:
        return Outcome(Status.OK, "already standing")
    front = obs.imu.fell_forward
    order = ["stand_up_front", "stand_up_back"] if front is not False else ["stand_up_back", "stand_up_front"]
    for prim in order:
        ctx.body.act(prim)
        obs = ctx.body.observe()
        if not obs.imu.fallen:
            return Outcome(Status.OK, f"recovered via {prim}", {"via": prim})
    return Outcome(Status.FAILED, "could not stand up")


@skill(SkillSpec("stop", "Halt all motion now."))
def stop(ctx: SkillContext, args: dict) -> Outcome:
    ctx.body.halt()
    return Outcome(Status.STOPPED, "halted")
