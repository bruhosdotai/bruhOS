import json

from bruhos.speech import SpeechConfig, make_asr, make_tts
from bruhos.speech.openai_compat import OpenAICompatASR, OpenAICompatTTS


def test_make_asr_openai():
    asr = make_asr(SpeechConfig(asr="openai", base_url="http://h/v1", asr_model="whisper-base"))
    assert isinstance(asr, OpenAICompatASR)


def test_make_tts_openai():
    tts = make_tts(SpeechConfig(tts="openai", base_url="http://h/v1", tts_model="piper", tts_voice="alloy"))
    assert isinstance(tts, OpenAICompatTTS)


def test_remote_asr_multipart_shape(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF....WAVE")
    captured = {}

    def multipart(url, headers, fields, file_field, filename, file_bytes, content_type, timeout):
        captured.update(url=url, fields=fields, file_field=file_field, filename=filename,
                        n=len(file_bytes), content_type=content_type)
        return json.dumps({"text": "wave hello"}).encode()

    asr = OpenAICompatASR(
        SpeechConfig(asr="openai", base_url="http://h/v1", asr_model="whisper-base:latest", language="en"),
        multipart=multipart,
    )
    assert asr.transcribe(str(wav)) == "wave hello"
    assert captured["url"] == "http://h/v1/audio/transcriptions"
    assert captured["fields"]["model"] == "whisper-base:latest"
    assert captured["file_field"] == "file"


def test_remote_asr_strips_qwen_wrapper(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF....WAVE")

    def multipart(url, headers, fields, file_field, filename, file_bytes, content_type, timeout):
        return json.dumps({
            "text": "language en<asr_text>Who is on the twenty-dollar bill?",
        }).encode()

    asr = OpenAICompatASR(
        SpeechConfig(asr="openai", base_url="http://h/v1", asr_model="qwen3-asr", language="en"),
        multipart=multipart,
    )
    assert asr.transcribe(str(wav)) == "Who is on the twenty-dollar bill?"


def test_remote_tts_request_and_http_playback(tmp_path):
    out = tmp_path / "audio"
    captured = {}

    def transport(url, headers, body, timeout):
        captured.update(url=url, headers=headers, body=json.loads(body))
        return b"RIFF....WAVEdata"

    tts = OpenAICompatTTS(
        SpeechConfig(tts="openai", base_url="http://h/v1", tts_model="piper-lessac:latest",
                     tts_voice="alloy", playback="http", http_dir=str(out)),
        transport=transport,
    )
    tts.say("Hello.")
    assert captured["url"] == "http://h/v1/audio/speech"
    assert captured["body"]["model"] == "piper-lessac:latest"
    assert captured["body"]["voice"] == "alloy"
    assert (out / "tts.wav").read_bytes().startswith(b"RIFF")
