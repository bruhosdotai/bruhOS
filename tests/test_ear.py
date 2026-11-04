from unittest.mock import MagicMock, patch

from bruhos.listen.ear import EarConfig, run_ear
from bruhos.listen.wonderecho import KNOWN, SLEEP, WAKE
from bruhos.speech import SpeechConfig
from bruhos.speech.openai_compat import OpenAICompatTTS


def test_wake_sleep_frames():
    assert KNOWN[WAKE] == "wakeup"
    assert KNOWN[SLEEP] == "sleep"
    assert len(WAKE) == 5 and WAKE[0] == 0xAA and WAKE[-1] == 0xFB


def test_wav_stats_silent(tmp_path):
    import wave

    from bruhos.listen.audio_debug import wav_stats

    p = tmp_path / "s.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 160)
    st = wav_stats(p)
    assert st["silent"] is True
    assert st["frames"] == 160


def test_tts_aplay_device_flag(tmp_path, monkeypatch):
    captured = {}

    def fake_run(cmd, check=False, stdout=None, stderr=None):
        captured["cmd"] = cmd
        return MagicMock(returncode=0)

    monkeypatch.setattr("bruhos.speech.openai_compat.subprocess.run", fake_run)
    monkeypatch.setattr("bruhos.speech.openai_compat.shutil.which", lambda n: "/usr/bin/aplay" if n == "aplay" else None)

    def transport(url, headers, body, timeout):
        return b"RIFF....WAVEdata"

    tts = OpenAICompatTTS(
        SpeechConfig(
            tts="openai",
            base_url="http://h/v1",
            tts_model="piper",
            playback="aplay",
            aplay_device="plughw:Device,0",
        ),
        transport=transport,
    )
    tts.say("hi")
    assert captured["cmd"][0] == "/usr/bin/aplay"
    assert captured["cmd"][1:3] == ["-D", "plughw:Device,0"]


def test_run_ear_once_wake_cycle(tmp_path):
    speech = SpeechConfig(asr="openai", tts="print", base_url="http://h/v1", playback="none")
    chat = MagicMock()
    chat.reply.return_value = "Hello back."

    class FakeEcho:
        def __init__(self, port, *args, **kwargs):
            self._n = 0

        def read_event(self):
            self._n += 1
            if self._n == 1:
                return None
            return "wakeup"

        def close(self):
            pass

    wav = tmp_path / "rec.wav"
    wav.write_bytes(b"RIFF....WAVE")

    with (
        patch("bruhos.listen.ear.WonderEcho", FakeEcho),
        patch("bruhos.listen.ear._record", return_value=wav),
        patch("bruhos.listen.ear.normalize_file", return_value=wav),
        patch("bruhos.listen.ear.make_asr") as masr,
        patch("bruhos.listen.ear.make_tts") as mtts,
    ):
        asr = MagicMock()
        asr.transcribe.return_value = "hello tony"
        masr.return_value = asr
        tts = MagicMock()
        mtts.return_value = tts
        run_ear(speech, chat, EarConfig(ack=True, seconds=1), once=True)

    assert asr.transcribe.called
    chat.reply.assert_called_once()
    assert chat.reply.call_args[0][0] == "hello tony"
    # ack + reply
    assert tts.say.call_count == 2
    assert tts.say.call_args_list[0][0][0] == "Yes?"
    assert tts.say.call_args_list[1][0][0] == "Hello back."
