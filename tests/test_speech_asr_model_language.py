import pytest

from bruhos.speech import SpeechConfig


def test_non_english_language_with_english_only_asr_model_is_refused():
    # faster-whisper silently forces "en" on an English-only (*.en) checkpoint.
    with pytest.raises(ValueError, match="English-only"):
        SpeechConfig(language="de", allow_non_english=True)


def test_non_english_language_with_multilingual_asr_model_is_accepted():
    cfg = SpeechConfig(language="de", asr_model="base", allow_non_english=True)
    assert cfg.language == "de"


@pytest.mark.parametrize("lang", ["en", "en-US", "EN"])
def test_english_with_english_only_asr_model_is_accepted(lang):
    SpeechConfig(language=lang, asr_model="base.en", allow_non_english=True)
