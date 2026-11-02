from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.eval import run_eval


def test_explicit_empty_prompts_do_not_load_or_execute_defaults(monkeypatch):
    import bruhos.eval.runner as runner

    loads = []
    monkeypatch.setattr(runner, "load_prompts", lambda: loads.append(True) or [
        {"text": "Wave hello.", "expect": ["event:wave"]},
    ])
    report = run_eval(RulesPlanner, prompts=[])
    assert loads == []
    assert report.results == []
    assert report.summary()["prompts"] == 0
    assert report.completion_rate == 0.0
    assert report.recovery_rate is None


def test_none_still_selects_default_prompt_set():
    report = run_eval(RulesPlanner, prompts=None)
    assert len(report.results) == 20
    assert report.completion_rate == 1.0
