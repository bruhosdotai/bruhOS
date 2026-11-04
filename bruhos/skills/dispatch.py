"""Validate + run one skill without re-planning (voice / HTTP skill API)."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from bruhos.body.base import Body
from bruhos.schemas import Episode, Outcome, SkillCall, Status, Step
from bruhos.skills.registry import SkillContext, SkillError, catalog_hash, run_skill, validate


@dataclass
class DispatchResult:
    call: SkillCall
    outcome: Outcome
    episode_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.call.skill,
            "args": self.call.args,
            "status": self.outcome.status.value,
            "detail": self.outcome.detail,
            "data": self.outcome.data,
            "ok": self.outcome.ok,
            "episode_id": self.episode_id,
        }


@dataclass
class SkillHub:
    """One shared Body for ear + HTTP. Lock covers the whole skill, not just act()."""

    body: Body
    max_skill_steps: int = 120
    should_stop: Callable[[], bool] = field(default=lambda: False)
    _lock: threading.RLock = field(default_factory=threading.RLock)
    last_episode: Episode | None = None

    def dispatch(self, call: SkillCall | dict[str, Any], *, brain: str = "voice") -> DispatchResult:
        try:
            if isinstance(call, dict):
                call = SkillCall.from_dict(call)
            call = validate(call)
        except (SkillError, ValueError, TypeError) as e:
            bad = call if isinstance(call, SkillCall) else SkillCall("?", {})
            return DispatchResult(call=bad, outcome=Outcome(Status.REJECTED, str(e)))
        with self._lock:
            before = self.body.observe()
            ctx = SkillContext(
                self.body,
                should_stop=self.should_stop,
                max_steps=self.max_skill_steps,
            )
            t0 = time.time()
            outcome = run_skill(ctx, call)
            after = self.body.observe()
            ep = Episode(
                episode_id=time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6],
                instruction=f"{call.skill}({call.args})",
                brain=brain,
                body=self.body.name,
                started_at=t0,
                ended_at=time.time(),
                skillpack_hash=catalog_hash(),
                plan=[call.to_dict()],
                steps=[
                    Step(
                        index=0,
                        t=before.t,
                        kind="skill",
                        call=call.to_dict(),
                        outcome=outcome.to_dict(),
                        obs_before=before.to_dict(),
                        obs_after=after.to_dict(),
                    )
                ],
                end_reason=outcome.detail,
                success=outcome.ok or outcome.status is Status.STOPPED,
            )
            if outcome.status is Status.FALLEN:
                ep.falls = 1
            from bruhos.trace.hashing import seal  # local: avoid skills↔trace import cycle

            seal(ep)
            self.last_episode = ep
            return DispatchResult(call=call, outcome=outcome, episode_id=ep.episode_id)
