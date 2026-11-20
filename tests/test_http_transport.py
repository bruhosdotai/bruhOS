"""The default brain transport never follows redirects, so the API key cannot leave its origin (issue #4)."""

import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from bruhos.brain.adapters.openai import OpenAICompatPlanner, http_post
from bruhos.sim.kinematic import SimBody

TOKEN = "Bearer test-only-not-a-real-key"
CODES = [301, 302, 303, 307, 308]


class Server:
    """Loopback HTTP server. ``location`` set: answer every request with that redirect, else 200 + a plan."""

    def __init__(self, location=None, code=302):
        self.location, self.code, self.requests = location, code, []
        srv = self

        class H(BaseHTTPRequestHandler):
            def _handle(self):
                n = int(self.headers.get("Content-Length") or 0)
                srv.requests.append((self.command, self.path, dict(self.headers), self.rfile.read(n)))
                if srv.location and self.path == "/v1/chat/completions":
                    self.send_response(srv.code)
                    self.send_header("Location", srv.location(self))
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                out = json.dumps({"choices": [{"message": {"content": '{"plan": [{"skill": "wave", "args": {}}]}'}}]})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out.encode())

            do_GET = do_POST = _handle

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, args=(0.02,), daemon=True).start()

    def url(self, path="/v1", host="127.0.0.1", scheme="http"):
        return f"{scheme}://{host}:{self.port}{path}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def servers():
    made = []

    def make(**kw):
        made.append(Server(**kw))
        return made[-1]

    yield make
    for s in made:
        s.close()


def _post(src):
    return http_post(src.url("/v1/chat/completions"), {"Authorization": TOKEN, "Content-Type": "application/json"},
                     b'{"model": "m"}', 5)


@pytest.mark.parametrize("code", CODES)
@pytest.mark.parametrize("change", ["port", "host", "scheme"])
def test_cross_origin_redirect_never_sends_key(servers, code, change):
    other = servers()
    target = {"port": other.url("/steal"),
              "host": other.url("/steal", host="localhost"),
              "scheme": other.url("/steal", scheme="https")}[change]
    src = servers(location=lambda h: target, code=code)
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(src)
    assert e.value.code == code and "refused" in str(e.value)
    assert other.requests == []
    assert len(src.requests) == 1 and src.requests[0][2]["Authorization"] == TOKEN


@pytest.mark.parametrize("code", CODES)
def test_same_origin_redirect_is_refused_too(servers, code):
    # Intended: no redirect is followed, even to the same origin. Configure base_url as the final endpoint.
    src = servers(location=lambda h: f"http://127.0.0.1:{h.server.server_address[1]}/elsewhere", code=code)
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(src)
    assert e.value.code == code
    assert [r[1] for r in src.requests] == ["/v1/chat/completions"]


def test_direct_call_still_works_with_key(servers):
    src = servers()
    raw = _post(src)
    assert json.loads(raw)["choices"][0]["message"]["content"]
    method, path, headers, body = src.requests[0]
    assert (method, path, headers["Authorization"], body) == ("POST", "/v1/chat/completions", TOKEN, b'{"model": "m"}')


def test_planner_with_default_transport_does_not_leak(servers):
    other = servers()
    src = servers(location=lambda h: other.url("/v1/chat/completions"), code=307)
    planner = OpenAICompatPlanner(base_url=src.url(), model="m", api_key="test-only-not-a-real-key")
    with pytest.raises(urllib.error.HTTPError):
        planner.plan("Wave.", SimBody().observe(), [])
    assert other.requests == []
