from bruhos.brain.base import Planner
from bruhos.eval import run_eval
from bruhos.schemas import SkillCall


class PartialPlanner(Planner):
    def plan(self, instruction, obs, catalog):
        return [SkillCall("wave"), SkillCall("grab")]


def test_event_before_a_failed_skill_is_not_a_completed_episode():
    report = run_eval(PartialPlanner, prompts=[{"text": "Wave and grab.", "expect": ["event:wave"]}])
    assert "grab failed" in report.results[0].end_reason
    assert not report.results[0].passed
    assert report.completion_rate == 0.0
