"""Tests for Runtime replanning and ep.plan consistency.

When a skill in an active plan fails (Status.FAILED), the runtime replans.
The unexecuted tail of the original plan (skills scheduled after the failed skill)
was never run and was abandoned in favor of the replanned sequence.
ep.plan must prune these abandoned steps before appending the replacement plan,
ensuring ep.plan accurately reflects the chronological sequence of skills actually
attempted and planned rather than retaining phantom unexecuted calls.
"""

from __future__ import annotations

from bruhos.brain.base import Planner
from bruhos.runtime import Runtime, RuntimeConfig
from bruhos.schemas import SkillCall
from bruhos.sim.kinematic import SimBody, World


class SequencePlanner(Planner):
    """Planner that returns an initial plan and successive replans."""

    name = "sequence_planner"

    def __init__(self, initial_plan: list[SkillCall], replans: list[list[SkillCall]]):
        self._initial = list(initial_plan)
        self._replans = [list(p) for p in replans]
        self.replan_calls: list[list[dict]] = []

    def plan(self, instruction: str, obs, catalog):
        return list(self._initial)

    def replan(self, instruction: str, obs, catalog, history):
        self.replan_calls.append(history)
        if self._replans:
            return list(self._replans.pop(0))
        return []


def test_replan_prunes_abandoned_suffix_from_ep_plan():
    """Unexecuted steps after a failed skill must be pruned from ep.plan upon replanning."""
    # World with no objects so grab fails immediately
    body = SimBody(World(objects=[]))
    # Initial plan: wave, grab (fails), look_at (unexecuted), release (unexecuted)
    initial = [
        SkillCall("wave"),
        SkillCall("grab"),
        SkillCall("look_at", {"target": "red_block"}),
        SkillCall("release"),
    ]
    # Replan after grab fails: substitute with wave
    replacement = [SkillCall("wave")]

    planner = SequencePlanner(initial, [replacement])
    rt = Runtime(body, planner, config=RuntimeConfig(max_replans=1))
    ep = rt.run("Test replan sequence")

    assert ep.success is True
    assert ep.end_reason == "completed"

    executed_skills = [s.call["skill"] for s in ep.steps]
    assert executed_skills == ["wave", "grab", "wave"]

    # ep.plan must only record the executed attempt up to failure plus the replacement,
    # without the abandoned ['look_at', 'release'] suffix from the initial plan.
    planned_skills = [c["skill"] for c in ep.plan]
    assert planned_skills == ["wave", "grab", "wave"]
    assert len(ep.plan) == len(ep.steps)


def test_multiple_replans_prune_successive_abandoned_tails():
    """Successive replans must each prune the abandoned tail of the superseded plan."""
    body = SimBody(World(objects=[]))
    # Initial: wave, grab (fails at index 1), wave, wave (2 abandoned)
    initial = [
        SkillCall("wave"),
        SkillCall("grab"),
        SkillCall("wave"),
        SkillCall("wave"),
    ]
    # First replan: wave, grab (fails at index 1), wave (1 abandoned)
    replan1 = [
        SkillCall("wave"),
        SkillCall("grab"),
        SkillCall("wave"),
    ]
    # Second replan: final wave
    replan2 = [
        SkillCall("wave"),
    ]

    planner = SequencePlanner(initial, [replan1, replan2])
    rt = Runtime(body, planner, config=RuntimeConfig(max_replans=2))
    ep = rt.run("Test double replan")

    assert ep.success is True
    assert ep.end_reason == "completed"

    executed_skills = [s.call["skill"] for s in ep.steps]
    assert executed_skills == ["wave", "grab", "wave", "grab", "wave"]

    planned_skills = [c["skill"] for c in ep.plan]
    assert planned_skills == ["wave", "grab", "wave", "grab", "wave"]
    assert len(ep.plan) == len(ep.steps)


def test_replan_failure_prunes_abandoned_suffix():
    """When a replan returns empty or fails, abandoned suffix is still pruned."""
    body = SimBody(World(objects=[]))
    initial = [
        SkillCall("wave"),
        SkillCall("grab"),
        SkillCall("wave"),
    ]
    # No replan available (empty list)
    planner = SequencePlanner(initial, [[]])
    rt = Runtime(body, planner, config=RuntimeConfig(max_replans=1))
    ep = rt.run("Test replan failure")

    assert ep.success is False
    assert "grab failed" in ep.end_reason

    executed_skills = [s.call["skill"] for s in ep.steps]
    assert executed_skills == ["wave", "grab"]

    planned_skills = [c["skill"] for c in ep.plan]
    # The abandoned 3rd step 'wave' must not appear in ep.plan
    assert planned_skills == ["wave", "grab"]
