from pathlib import Path

import pytest

from bruhos.listen.stack import apply_ear_stack, resolve_ear_stack
from bruhos.speech import SpeechConfig, make_tts
from bruhos.speech.piper import PiperTTS, piper_model_path


def _cfg():
    return {
        "speech": {
            "asr": "openai",
            "tts": "openai",
            "base_url": "http://remote.example:2083/v1",
            "asr_model": "whisper-base:latest",
            "tts_model": "piper-lessac:latest",
            "realtime": "duplex",
            "chat_model": "qwen2.5:14b",
            "language": "english",
            "playback": "both",
        },
        "brain": {
            "name": "openai",
            "base_url": "http://remote.example:2083/v1",
            "model": "qwen2.5:7b",
        },
        "detect": {"detect_url": "http://remote.example:2083"},
        "ear": {"port": "/dev/ttyUSB0"},
    }


def test_remote_keeps_the_config_urls():
    cfg = _cfg()
    out = apply_ear_stack(cfg, "remote")
    assert out is cfg
    assert out["speech"]["base_url"] == "http://remote.example:2083/v1"
    assert out["brain"]["model"] == "qwen2.5:7b"
    assert out["detect"]["detect_url"] == "http://remote.example:2083"


def test_local_rewrites_only_ear_speech_and_brain():
    out = apply_ear_stack(_cfg(), "local")
    assert out["speech"]["base_url"] == "http://127.0.0.1:11435/v1"
    assert out["speech"]["asr"] == "openai"
    assert out["speech"]["asr_model"] == "qwen3-asr"
    assert out["speech"]["tts"] == "piper"
    assert out["speech"]["tts_model"] == "en_US-lessac-medium"
    assert out["speech"]["realtime"] == "off"
    assert "chat_model" not in out["speech"]
    assert out["speech"]["playback"] == "both"
    assert out["brain"]["base_url"] == "http://127.0.0.1:11435/v1"
    assert out["brain"]["model"] == "tinyllama"
    assert out["detect"]["detect_url"] == "http://remote.example:2083"


def test_cli_override_beats_config_and_omitted_stack_is_local():
    cfg = _cfg()
    cfg["ear"]["stack"] = "remote"
    assert resolve_ear_stack(cfg, None) == "remote"
    assert resolve_ear_stack(cfg, "local") == "local"
    assert resolve_ear_stack(_cfg(), None) == "local"
    out = apply_ear_stack(_cfg(), resolve_ear_stack(_cfg(), None))
    assert out["speech"]["asr_model"] == "qwen3-asr"
    assert out["brain"]["model"] == "tinyllama"


def test_unknown_stack_is_rejected():
    with pytest.raises(SystemExit):
        resolve_ear_stack({}, "cloud")


def test_piper_model_path_names_the_installed_voice():
    assert piper_model_path(None).endswith("/en_US-lessac-medium.onnx")
    assert piper_model_path("/tmp/voice.onnx") == "/tmp/voice.onnx"


def test_piper_tts_writes_the_wav_from_stdin(tmp_path, monkeypatch):
    voice = tmp_path / "en_US-lessac-medium.onnx"
    voice.write_bytes(b"onnx")
    captured = {}

    def fake_run(cmd, input=None, check=False, stdout=None, stderr=None):
        captured["cmd"] = cmd
        captured["input"] = input
        Path(cmd[-1]).write_bytes(b"RIFF")

        class Proc:
            returncode = 0
            stderr = b""

        return Proc()

    monkeypatch.setattr("bruhos.speech.piper.piper_bin", lambda: "/usr/local/bin/piper")
    tts = PiperTTS(
        SpeechConfig(tts="piper", tts_model=str(voice), playback="none"),
        runner=fake_run,
    )
    assert tts.synthesize("Hello.") == b"RIFF"
    assert captured["cmd"][:4] == ["/usr/local/bin/piper", "-m", str(voice), "-f"]
    assert captured["input"] == b"Hello."
    assert isinstance(make_tts(SpeechConfig(tts="piper", tts_model=str(voice))), PiperTTS)
