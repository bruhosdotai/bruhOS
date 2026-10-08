"""bruhOS Runtime: observe -> plan -> validate -> dispatch -> recover -> trace."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from bruhos.body.base import Body
from bruhos.brain.base import IntegrationError, Planner
from bruhos.schemas import Episode, Observation, Outcome, SkillCall, Status, Step
from bruhos.skills import CATALOG, SkillContext, SkillError, catalog_hash, run_skill, validate
from bruhos.trace.hashing import seal
from bruhos.trace.sinks import SinkDispatcher

log = logging.getLogger(__name__)


@dataclass
class RuntimeConfig:
    max_recoveries: int = 3        # stand_up attempts per episode
    max_replans: int = 1           # brain re-plans after a failed skill
    max_skill_steps: int = 120     # primitive budget inside one skill


class Runtime:
    def __init__(self, body: Body, brain: Planner, sinks: SinkDispatcher | None = None,
                 config: RuntimeConfig | None = None):
        self.body = body
        self.brain = brain
        self.sinks = sinks
        self.cfg = config or RuntimeConfig()
        self._stop = threading.Event()
        self.ctx = SkillContext(body, should_stop=self._stop.is_set, max_steps=self.cfg.max_skill_steps)

    # ---- control ----
    def stop(self) -> None:
        """Preempt everything. Local only: never waits on brain, sinks or chain."""
        self._stop.set()
        self.body.halt()

    def reset_stop(self) -> None:
        self._stop.clear()
        if hasattr(self.body, "resume"):
            self.body.resume()

    # ---- episode ----
    def run(self, instruction: str) -> Episode:
        ep = Episode(
            episode_id=time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6],
            instruction=instruction, brain=self.brain.name, body=self.body.name,
            started_at=time.time(), skillpack_hash=catalog_hash(), brain_version=self.brain.version,
        )
        try:
            self._run(ep)
        finally:
            ep.ended_at = time.time()
            seal(ep)
            if self.sinks:
                self.sinks.submit(ep)
        return ep

    def _record(self, ep: Episode, kind: str, call: SkillCall, outcome: Outcome,
                before: Observation, after: Observation | None) -> None:
        ep.steps.append(Step(index=len(ep.steps), t=before.t, kind=kind, call=call.to_dict(),
                             outcome=outcome.to_dict(), obs_before=before.to_dict(),
                             obs_after=after.to_dict() if after else None))

    def _plan(self, ep: Episode, obs: Observation, history: list[dict[str, Any]] | None) -> list[SkillCall] | None:
        try:
            raw = (self.brain.replan(ep.instruction, obs, CATALOG, history) if history
                   else self.brain.plan(ep.instruction, obs, CATALOG))
            plan = [validate(c) for c in raw]
        except (SkillError, IntegrationError, ValueError) as e:
            self._record(ep, "rejected", SkillCall("plan", {"error": type(e).__name__}),
                         Outcome(Status.REJECTED, str(e)), obs, None)
            ep.end_reason = f"plan rejected: {e}"
            return None
        except Exception as e:
            log.exception("brain failed")
            ep.end_reason = f"brain error: {e}"
            return None
        ep.plan.extend(c.to_dict() for c in plan)
        return plan

    def _run(self, ep: Episode) -> None:
        obs = self.body.observe()
        if obs.imu.fallen:
            obs = self._recover(ep, obs)
            if obs is None:
                return
        plan = self._plan(ep, obs, None)
        if plan is None:
            return
        if not plan:
            ep.end_reason = "no plan for instruction"
            return

        replans, i = 0, 0
        while i < len(plan):
            call = plan[i]
            if self._stop.is_set() and call.skill != "stop":
                ep.end_reason = "stopped"
                return
            before = self.body.observe()
            outcome = run_skill(self.ctx, call)
            after = self.body.observe()
            self._record(ep, "stop" if call.skill == "stop" else "skill", call, outcome, before, after)

            if outcome.status is Status.FALLEN:
                ep.falls += 1
                if self._recover(ep, after) is None:
                    return
                continue                     # resume the interrupted skill
            if outcome.status is Status.STOPPED:
                ep.end_reason = "stopped"
                ep.success = call.skill == "stop" and plan[-1].skill == "stop"
                return
            if outcome.status is Status.FAILED:
                if replans >= self.cfg.max_replans:
                    ep.end_reason = f"{call.skill} failed: {outcome.detail}"
                    return
                replans += 1
                history = [{"call": s.call, "outcome": s.outcome} for s in ep.steps if s.kind != "rejected"]
                new = self._plan(ep, after, history)
                if not new:
                    ep.end_reason = ep.end_reason or f"{call.skill} failed: {outcome.detail}"
                    return
                plan, i = new, 0
                continue
            i += 1

        ep.success = True
        ep.end_reason = "completed"

    def _recover(self, ep: Episode, obs: Observation) -> Observation | None:
        while obs.imu.fallen:
            if ep.recoveries >= self.cfg.max_recoveries or self._stop.is_set():
                ep.end_reason = "fallen, recovery budget exhausted" if not self._stop.is_set() else "stopped"
                return None
            ep.recoveries += 1
            call = SkillCall("stand_up")
            outcome = run_skill(self.ctx, call)
            after = self.body.observe()
            self._record(ep, "recovery", call, outcome, obs, after)
            obs = after
        return obs
