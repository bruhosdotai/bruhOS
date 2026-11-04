"""Top-level section names in an operator config file are a closed set:
every table the code reads is known (cli: brain/speech/trace, TonyPiBody:
body/actions/camera/imu/perception, ElizaBridgeBody: body/actions/walk).
A misspelled section name used to load silently while every setting under
it was dropped — including validation that would have fired under the
correct name ([speech] language="zh" is refused; [spech] was ignored).
The set below pins that contract; it deliberately does not import any
constant from the implementation, so on the old code the rejection tests
fail as plain DID-NOT-RAISE assertion failures."""

from pathlib import Path

import pytest

from bruhos.cli import _load_config

ROOT = Path(__file__).resolve().parents[1]

KNOWN_SECTIONS = {
    "actions", "body", "brain", "camera", "imu",
    "perception", "speech", "trace", "walk",
}


def _write(tmp_path: Path, text: str) -> str:
    p = tmp_path / "cfg.toml"
    p.write_text(text)
    return str(p)


def test_misspelled_brain_section_is_rejected(tmp_path):
    path = _write(tmp_path, '[brian]\nname = "openai"\n')
    with pytest.raises(ValueError, match="brian"):
        _load_config(path)


def test_misspelled_speech_section_is_rejected(tmp_path):
    # Under the correct name this value is refused by SpeechConfig; under the
    # typo it used to be skipped entirely and the defaults ran instead.
    path = _write(tmp_path, '[spech]\nlanguage = "zh"\n')
    with pytest.raises(ValueError, match="spech"):
        _load_config(path)


def test_all_unknown_sections_are_named(tmp_path):
    path = _write(tmp_path, '[bogus]\nfoo = 1\n[alsonotreal]\nbar = 2\n')
    with pytest.raises(ValueError) as e:
        _load_config(path)
    assert "bogus" in str(e.value) and "alsonotreal" in str(e.value)


def test_every_known_section_is_accepted(tmp_path):
    text = "".join(f"[{name}]\n" for name in sorted(KNOWN_SECTIONS))
    cfg = _load_config(_write(tmp_path, text))
    assert set(cfg) == KNOWN_SECTIONS


@pytest.mark.parametrize("cfg", sorted((ROOT / "configs").glob("*.toml")))
def test_shipped_configs_load(cfg):
    loaded = _load_config(str(cfg))
    assert set(loaded) <= KNOWN_SECTIONS


def test_no_config_path_is_unchanged():
    assert _load_config(None) == {}
