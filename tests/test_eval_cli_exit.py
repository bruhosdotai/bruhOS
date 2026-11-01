import json

import pytest

from bruhos.cli import main


@pytest.mark.parametrize("instruction,expected", [("Wave hello.", 0), ("Do a backflip.", 1)])
def test_eval_exit_matches_real_prompt_checks(tmp_path, monkeypatch, capsys, instruction, expected):
    import bruhos.eval.runner as runner

    monkeypatch.setattr(runner, "load_prompts", lambda: [
        {"text": instruction, "expect": ["event:wave"]},
    ])
    code = main(["eval", "--brain", "rules"])
    output = capsys.readouterr().out
    assert ("PASS" if expected == 0 else "FAIL") in output
    summary = json.loads(output[output.index("{"):])
    assert summary["passed"] == (1 if expected == 0 else 0)
    assert code == expected
