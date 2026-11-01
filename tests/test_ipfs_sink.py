"""The IPFS sink publishes a sealed episode through Kubo's add endpoint.

The README's status table lists the IPFS sink as "Implemented, not covered by
tests". These tests pin its wire contract: the request URL and query, a
well-formed multipart body whose only part carries the canonical episode JSON,
and the receipt mapped from Kubo's response. A Kubo error response (200 with
an error body, or a 500) and an unreachable node fail the emit instead of
producing a cid, and the uploaded bytes are exactly the sealed, verifiable
trace.
"""

import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from bruhos.brain.adapters.rules import RulesPlanner
from bruhos.runtime import Runtime
from bruhos.schemas import Episode, Step
from bruhos.sim.kinematic import SimBody
from bruhos.trace import IpfsSink, verify_episode
from bruhos.trace.hashing import canonical


class FakeKubo:
    """Answers ``POST /api/v0/add`` the way Kubo's HTTP API does and records the request."""

    def __init__(self, response=None, status=200):
        self.requests = []
        self.status = status
        self.response = response if response is not None else {
            "Hash": "bafytestcid", "Name": "episode.json", "Bytes": "262158"}
        srv = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n)
                srv.requests.append((self.path, self.headers.get("Content-Type", ""), body))
                out = json.dumps(srv.response).encode()
                self.send_response(srv.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, args=(0.02,), daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _episode():
    return Runtime(SimBody(), RulesPlanner()).run("Wave hello.")


def test_adds_the_sealed_episode_to_kubo_and_maps_the_receipt():
    kubo = FakeKubo()
    try:
        ep = _episode()
        receipt = IpfsSink(api=kubo.url).emit(ep)
        path, content_type, body = kubo.requests[0]

        assert path == "/api/v0/add?pin=true&cid-version=1"
        assert content_type.startswith("multipart/form-data; boundary=")
        boundary = content_type.split("boundary=")[1].encode()

        # The single file part carries the canonical episode JSON.
        payload = canonical({"episode": ep.header(), "steps": [s.to_dict() for s in ep.steps]})
        part_headers = (b"--" + boundary + b"\r\n"
                        b'Content-Disposition: form-data; name="file"; '
                        b'filename="' + ep.episode_id.encode() + b'.json"\r\n'
                        b"Content-Type: application/json\r\n\r\n")
        assert body == part_headers + payload + b"\r\n--" + boundary + b"--\r\n"
        assert receipt == {"cid": "bafytestcid"}
    finally:
        kubo.close()


def test_the_multipart_delimiters_close_the_body():
    kubo = FakeKubo()
    try:
        IpfsSink(api=kubo.url).emit(_episode())
        _, content_type, body = kubo.requests[0]
        boundary = content_type.split("boundary=")[1].encode()

        assert body.startswith(b"--" + boundary + b"\r\n")
        assert body.endswith(b"\r\n--" + boundary + b"--\r\n")
        assert body.count(b"--" + boundary) == 2
    finally:
        kubo.close()


def test_a_kubo_error_response_fails_the_sink_instead_of_faking_a_cid():
    kubo = FakeKubo(response={"Message": "nope"})
    try:
        with pytest.raises(KeyError):
            IpfsSink(api=kubo.url).emit(_episode())
        assert kubo.requests[0][0] == "/api/v0/add?pin=true&cid-version=1"
    finally:
        kubo.close()


def test_a_kubo_http_500_fails_the_sink_instead_of_faking_a_cid():
    kubo = FakeKubo(response={"Message": "context deadline exceeded", "Code": 500}, status=500)
    try:
        with pytest.raises(urllib.error.HTTPError):
            IpfsSink(api=kubo.url).emit(_episode())
    finally:
        kubo.close()


def test_an_unreachable_node_fails_the_sink_instead_of_faking_a_cid():
    # Nothing listens on loopback port 1, so this is the common "Kubo is not
    # running" case rather than a real network dependency.
    with pytest.raises(urllib.error.URLError):
        IpfsSink(api="http://127.0.0.1:1", timeout=2).emit(_episode())


def test_the_uploaded_bytes_are_the_sealed_verifiable_trace():
    kubo = FakeKubo()
    try:
        ep = _episode()
        IpfsSink(api=kubo.url).emit(ep)
        _, content_type, body = kubo.requests[0]
        boundary = content_type.split("boundary=")[1].encode()

        # The file part's body is exactly the canonical JSON document.
        payload = body.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n--" + boundary + b"--", 1)[0]
        doc = json.loads(payload)
        header, steps = doc["episode"], doc["steps"]

        assert header["n_steps"] == len(steps)
        assert header["trace_root"] == ep.trace_root

        # The document rebuilds into an episode whose hash chain verifies.
        uploaded = Episode(
            episode_id=header["episode_id"], instruction=header["instruction"],
            brain=header["brain"], body=header["body"], started_at=header["started_at"],
            plan=header["plan"], steps=[Step(**s) for s in steps],
            success=header["success"], recoveries=header["recoveries"], falls=header["falls"],
            ended_at=header["ended_at"], end_reason=header["end_reason"],
            skillpack_hash=header["skillpack_hash"], brain_version=header["brain_version"],
            trace_root=header["trace_root"])
        assert verify_episode(uploaded)
    finally:
        kubo.close()
