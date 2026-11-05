import tomllib
from pathlib import Path

import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.eval import load_prompts, run_eval
from bruhos.speech import SpeechConfig

ROOT = Path(__file__).resolve().parents[1]


def test_prompt_set_is_20_english_prompts():
    prompts = load_prompts()
    assert len(prompts) == 20
    assert all(p["text"].isascii() for p in prompts)


def test_rules_baseline_on_sim():
    r = run_eval(RulesPlanner)
    assert len(r.results) == 20
    assert r.completion_rate == 1.0, [x for x in r.results if not x.passed]


def test_recovery_reported_separately():
    r = run_eval(RulesPlanner, fall_prob=0.05, seed=1)
    s = r.summary()
    assert s["falls"] > 0 and s["recovery_rate"] is not None
    assert 0 <= s["completion_rate"] <= 1


def test_eval_command_fails_when_prompts_fail(monkeypatch):
    """`bruhos eval` must exit non-zero when prompts fail, so CI can gate on it."""
    import argparse

    import bruhos.cli as cli
    from bruhos.brain.base import Planner
    from bruhos.cli import cmd_eval

    class NoPlan(Planner):
        name = "none"

        def plan(self, instruction, obs, catalog):
            return []

    monkeypatch.setattr(cli, "_brain", lambda name, cfg: NoPlan())
    rc = cmd_eval(argparse.Namespace(brain="none", config=None, fall_prob=0.0, seed=0))
    assert rc == 1


def test_eval_command_passes_when_prompts_pass():
    import argparse

    from bruhos.cli import cmd_eval

    rc = cmd_eval(argparse.Namespace(brain="rules", config=None, fall_prob=0.0, seed=0))
    assert rc == 0


def test_english_is_default():
    assert SpeechConfig().language == "en"


def test_chinese_asr_default_is_refused():
    with pytest.raises(ValueError):
        SpeechConfig(language="zh")


@pytest.mark.parametrize("cfg", sorted((ROOT / "configs").glob("*.toml")))
def test_shipped_configs_are_english(cfg):
    with open(cfg, "rb") as f:
        speech = tomllib.load(f)["speech"]
    SpeechConfig(**speech)
    assert speech["language"] == "en"
