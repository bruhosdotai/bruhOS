"""SECURITY.md rule 3: "HTTP redirects are never followed."

The brain transport keeps that promise (merged #5, covered by test_http_transport.py). The IPFS
trace sink used `urllib.request.urlopen`, which follows every redirect class, so a node that
answers the Kubo `add` call with a redirect made the sink report a CID that the configured node
never produced.
"""

import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody
from bruhos.trace import IpfsSink

CODES = [301, 302, 303, 307, 308]
ADD_PATH = "/api/v0/add"


class FakeKubo:
    """Loopback stand-in for a Kubo API endpoint; records every request it receives."""

    def __init__(self, response=None, redirect_to=None, code=302):
        self.requests = []
        self.response = response if response is not None else {"Hash": "bafytestcid"}
        self.redirect_to, self.code = redirect_to, code
        srv = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self):
                n = int(self.headers.get("Content-Length") or 0)
                srv.requests.append((self.command, self.path, self.rfile.read(n)))
                if self.path.startswith(ADD_PATH):
                    if srv.redirect_to:
                        self.send_response(srv.code)
                        self.send_header("Location", srv.redirect_to)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    out = json.dumps(srv.response).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_GET = do_POST = _handle

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()

    @property
    def api(self):
        return f"http://127.0.0.1:{self.port}"

    @property
    def paths(self):
        return [path for _, path, _ in self.requests]

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def episode():
    """A real sealed episode: rules brain driving SimBody, exactly what a sink would publish."""
    return Runtime(SimBody(), RulesPlanner()).run("Wave at the red block.")


@pytest.mark.parametrize("code", CODES)
def test_redirect_is_never_followed(code, episode):
    node, other = FakeKubo(), FakeKubo(response={"Hash": "bafyfromanotherorigin"})
    node.redirect_to, node.code = f"{other.api}{ADD_PATH}", code
    try:
        with pytest.raises(urllib.error.HTTPError):
            IpfsSink(api=node.api).emit(episode)
        assert other.requests == [], "the sink reached an origin only a redirect named"
        assert node.paths == [f"{ADD_PATH}?pin=true&cid-version=1"]
    finally:
        node.stop()
        other.stop()


def test_same_origin_redirect_is_refused_too(episode):
    # Intended: no redirect is followed, even to the same origin. Point api at the final endpoint.
    node = FakeKubo(redirect_to="/api/v0/pinned")
    try:
        with pytest.raises(urllib.error.HTTPError):
            IpfsSink(api=node.api).emit(episode)
        assert node.paths == [f"{ADD_PATH}?pin=true&cid-version=1"]
    finally:
        node.stop()


def test_direct_endpoint_still_works(episode):
    node = FakeKubo(response={"Hash": "bafyrealcid", "Name": "episode.json", "Bytes": "1024"})
    try:
        assert IpfsSink(api=node.api).emit(episode) == {"cid": "bafyrealcid"}
        assert node.paths == [f"{ADD_PATH}?pin=true&cid-version=1"]
    finally:
        node.stop()
