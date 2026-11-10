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


def test_tinyllama_template_ends_with_the_trained_newline():
    from pathlib import Path

    text = Path("contrib/templates/tinyllama.jinja").read_text()
    assert "{{ '<|assistant|>\\n' }}" in text
    assert "chat-template-file = /root/Sites/bruhOS/contrib/templates/tinyllama.jinja" in Path(
        "contrib/llama-models.ini"
    ).read_text()


def test_clean_transcript_keeps_words_after_qwen_tag():
    from bruhos.speech.base import clean_transcript

    assert clean_transcript(
        "language en<asr_text>Who is on the twenty-dollar bill?"
    ) == "Who is on the twenty-dollar bill?"
    assert clean_transcript("language English\n<asr_text>Piper is ready.") == "Piper is ready."
    assert clean_transcript("language None<asr_text>") == ""
    assert clean_transcript("language None<asr_text>still said it") == "still said it"
    assert clean_transcript("Who's on the $20 bill?") == "Who's on the $20 bill?"


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
