import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.eval import run_eval


@pytest.mark.parametrize("count", [1, 2])
def test_report_does_not_construct_an_unused_extra_planner(count):
    created = []

    def factory():
        if len(created) == count:
            raise RuntimeError("no unused planner available")
        brain = RulesPlanner()
        created.append(brain)
        return brain

    prompts = [{"text": "Wave hello.", "expect": ["event:wave"]}] * count
    report = run_eval(factory, prompts=prompts)
    assert len(created) == count
    assert len(report.results) == count
    assert report.completion_rate == 1.0
    assert report.brain == "rules"
