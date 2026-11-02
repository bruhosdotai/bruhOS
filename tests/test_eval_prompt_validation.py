"""Prompt entries are validated when loaded, not discovered mid-eval.

On ``main`` a misspelled ``world`` key was ``setattr`` onto the World and ignored, so the prompt ran
against the default scene and passed; an unknown ``expect`` kind raised only after every earlier
episode had already run.
"""

from __future__ import annotations

import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.eval.runner import PROMPTS, load_prompts, run_eval


def _no_brain():
    pytest.fail("no episode may run when a prompt entry is invalid")


def test_shipped_prompt_file_loads():
    assert len(load_prompts(PROMPTS)) == 20


def test_unknown_world_key_is_rejected_before_any_episode():
    prompt = {"text": "Look at the yellow ball.", "expect": ["facing:yellow_ball"], "world": {"heading": 170.0}}
    with pytest.raises(ValueError, match=r"prompt 0 .*world.*heading"):
        run_eval(_no_brain, prompts=[prompt])


def test_unknown_expect_kind_is_rejected_before_any_episode():
    prompts = [{"text": "Wave hello.", "expect": ["event:wave"]},
               {"text": "Wave hello.", "expect": ["holds:red_block"]}]
    with pytest.raises(ValueError, match=r"prompt 1 .*holds"):
        run_eval(_no_brain, prompts=prompts)


@pytest.mark.parametrize("expect", ["near", "near:", "event:", "success:yes", "no_plan:x"])
def test_malformed_expect_is_rejected(expect):
    with pytest.raises(ValueError, match="prompt 0 "):
        run_eval(_no_brain, prompts=[{"text": "Wave hello.", "expect": [expect]}])


@pytest.mark.parametrize("entry", [
    {"expect": ["success"]},                                  # no text
    {"text": "Wave hello."},                                  # no expect
    {"text": "Wave hello.", "expect": []},                    # nothing to check
    {"text": "Wave hello.", "expect": "success"},             # not a list
    {"text": "Wave hello.", "expect": ["success"], "world": {"objects": []}},   # not an override
])
def test_structural_errors_are_rejected(entry):
    with pytest.raises(ValueError, match="prompt 0 "):
        run_eval(_no_brain, prompts=[entry])


def test_load_prompts_validates_the_file(tmp_path):
    bad = tmp_path / "p.toml"
    bad.write_text('[[prompt]]\ntext = "Wave hello."\nexpect = ["event:wave"]\n\n'
                   '[[prompt]]\ntext = "Stand up."\nexpect = ["success"]\nworld = { fallen = true, fell_fwd = false }\n')
    with pytest.raises(ValueError, match=r"prompt 1 .*fell_fwd"):
        load_prompts(bad)


def test_valid_overrides_still_run():
    report = run_eval(RulesPlanner, prompts=[
        {"text": "Stand up.", "expect": ["event:stood_up", "success"], "world": {"fallen": True, "fell_forward": False}},
        {"text": "Follow the line.", "expect": ["success"], "world": {"line_m": 0.6}},
    ])
    assert [r.passed for r in report.results] == [True, True]
