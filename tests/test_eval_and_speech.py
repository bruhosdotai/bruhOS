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


def test_english_is_default():
    assert SpeechConfig().language == "en"


def test_supported_languages_normalize():
    assert SpeechConfig(language="english").language == "en"
    assert SpeechConfig(language="chinese").language == "zh"
    assert SpeechConfig(language="malay").language == "ms"
    assert SpeechConfig(language="thai").language == "th"
    assert SpeechConfig(language="vietnam").language == "vi"
    with pytest.raises(ValueError):
        SpeechConfig(language="fr")
    assert SpeechConfig(language="fr", allow_non_english=True).language == "fr"


@pytest.mark.parametrize("cfg", sorted((ROOT / "configs").glob("*.toml")))
def test_shipped_configs_are_english(cfg):
    with open(cfg, "rb") as f:
        speech = tomllib.load(f)["speech"]
    assert SpeechConfig(**speech).language == "en"
