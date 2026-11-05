"""Regression tests for the Nov 2026 security pass."""

import http.client
import threading
from functools import partial
from http.server import ThreadingHTTPServer

import pytest

import bruhos.speech as speech
from bruhos.body.base import Body
from bruhos.brain.adapters.openai import OpenAICompatPlanner
from bruhos.brain.adapters.qwen_vl import QwenVLPlanner
from bruhos.dashboard import Handler
from bruhos.schemas import Imu, LineReading, Observation, SkillCall, Status
from bruhos.skills import SkillContext, SkillError, run_skill, validate


class OffCenterLine(Body):
    """The line is always far right of center, so a naive follower would turn forever."""

    name = "off_center"
    turn_step_deg = 15.0

    def __init__(self):
        self.acts = 0

    def observe(self):
        return Observation(t=0, frame=0, objects=[], imu=Imu(), line=LineReading(visible=True, offset=0.9))

    def act(self, primitive):
        self.acts += 1
        return True

    def halt(self):
        pass


def test_follow_line_is_bounded_by_the_step_budget():
    body = OffCenterLine()
    out = run_skill(SkillContext(body, max_steps=50), SkillCall("follow_line", {"steps": 10**9}))
    assert out.status is Status.FAILED
    assert body.acts == 50


class VanishingLine(Body):
    """The line is visible once, far right of center, then leaves view after the first turn."""

    name = "vanishing_line"
    turn_step_deg = 15.0

    def __init__(self):
        self.looks = 0

    def observe(self):
        self.looks += 1
        visible = self.looks == 1
        return Observation(t=0, frame=self.looks, objects=[], imu=Imu(),
                           line=LineReading(visible=visible, offset=0.9))

    def act(self, primitive):
        return True

    def halt(self):
        pass


def test_follow_line_lost_before_any_step_is_not_success():
    # Same condition as "no line in view" at entry (FAILED), one turn later: the
    # robot followed zero steps, so the run must not certify "end of line".
    out = run_skill(SkillContext(VanishingLine(), max_steps=50), SkillCall("follow_line", {"steps": 5}))
    assert out.status is Status.FAILED
    assert out.data["steps"] == 0


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", float("nan")])
def test_non_finite_numbers_are_rejected(bad):
    with pytest.raises(SkillError):
        validate(SkillCall("walk_toward", {"target": "box", "stop_distance_m": bad}))


def test_huge_int_from_float_is_rejected():
    with pytest.raises(SkillError):
        validate(SkillCall("follow_line", {"steps": float("inf")}))


def test_espeak_text_never_becomes_an_option(monkeypatch):
    calls = []
    monkeypatch.setattr(speech.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)))
    tts = speech.EspeakTTS.__new__(speech.EspeakTTS)
    tts.cfg, tts.bin = speech.SpeechConfig(), "espeak-ng"
    tts.say("-w/home/pi/.bashrc hello")
    argv, kw = calls[0]
    assert argv == ["espeak-ng", "-v", "en-us", "--stdin"]
    assert kw["input"] == b"-w/home/pi/.bashrc hello"


@pytest.mark.parametrize("url", ["http://203.0.113.9/v1", "http://example.com/v1", "http://192.168.1.50:8000/v1"])
def test_real_api_key_is_never_sent_over_plain_http(url):
    with pytest.raises(ValueError, match="cleartext"):
        OpenAICompatPlanner(base_url=url, api_key="sk-real")


@pytest.mark.parametrize("url", ["https://api.openai.com/v1", "http://127.0.0.1:8000/v1", "http://localhost:8080/v1",
                                 "http://[::1]:8080/v1"])
def test_api_key_allowed_over_tls_or_loopback(url):
    OpenAICompatPlanner(base_url=url, api_key="sk-real")


def test_lan_server_without_a_real_key_still_works():
    QwenVLPlanner(base_url="http://192.168.1.50:8000/v1")
    OpenAICompatPlanner(base_url="http://192.168.1.50:8000/v1", api_key="")


def test_insecure_opt_in():
    OpenAICompatPlanner(base_url="http://192.168.1.50:8000/v1", api_key="sk-real", allow_insecure=True)


def test_eliza_token_never_sent_over_plain_ws(monkeypatch):
    from bruhos.body.eliza_bridge import TOKEN_ENV, ElizaBridgeBody

    monkeypatch.setenv(TOKEN_ENV, "secret")
    with pytest.raises(ValueError, match="cleartext"):
        ElizaBridgeBody({"body": {"url": "ws://203.0.113.9:9100"}})


@pytest.fixture
def dashboard(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, root=tmp_path))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_port
    srv.shutdown()


def _status(port, host):
    c = http.client.HTTPConnection("127.0.0.1", port)
    c.request("GET", "/status", headers={"Host": host})
    return c.getresponse().status


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8765", "127.0.0.1.evil.example", ""])
def test_dashboard_refuses_foreign_host_headers(dashboard, host):
    assert _status(dashboard, host) == 421


@pytest.mark.parametrize("host", ["127.0.0.1:{port}", "localhost:{port}", "127.0.0.1:8876", "[::1]:{port}"])
def test_dashboard_answers_loopback_and_ssh_tunnel(dashboard, host):
    assert _status(dashboard, host.format(port=dashboard)) == 200
