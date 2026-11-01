"""The documented Qwen environment override selects the HTTP endpoint."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from bruhos.brain.adapters.lfm3_omni import LFM3OmniPlanner
from bruhos.brain.adapters.qwen_vl import QwenVLPlanner
from bruhos.cli import _brain
from bruhos.runtime import Runtime
from bruhos.sim.kinematic import SimBody


def test_readme_environment_override_routes_the_real_http_request(monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, payload))
            response = json.dumps({"choices": [{"message": {"content":
                '{"plan": [{"skill": "wave", "args": {}}]}'}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    worker.start()
    endpoint = f"http://127.0.0.1:{server.server_address[1]}/v1"
    monkeypatch.setenv("BRUHOS_LLM_BASE_URL", endpoint)
    try:
        planner = _brain("qwen_vl", {})
        # Fail before sending anything to the incorrect default port on the base revision.
        assert planner.base_url == endpoint
        body = SimBody()
        episode = Runtime(body, planner).run("Wave hello.")
        assert episode.success, episode.end_reason
        assert body.world.events == ["wave"]
        assert len(requests) == 1
        assert requests[0][0] == "/v1/chat/completions"
        assert requests[0][1]["model"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(1)


def test_explicit_qwen_config_keeps_precedence_over_environment(monkeypatch):
    monkeypatch.setenv("BRUHOS_LLM_BASE_URL", "http://localhost:8101/v1")
    planner = _brain("qwen_vl", {"brain": {"base_url": "http://localhost:8102/v1"}})
    assert planner.base_url == "http://localhost:8102/v1"


def test_qwen_without_override_keeps_local_default(monkeypatch):
    monkeypatch.delenv("BRUHOS_LLM_BASE_URL", raising=False)
    assert QwenVLPlanner().base_url == "http://127.0.0.1:8000/v1"


def test_lfm3_fixed_local_default_is_not_changed(monkeypatch):
    monkeypatch.setenv("BRUHOS_LLM_BASE_URL", "http://localhost:8101/v1")
    assert LFM3OmniPlanner().base_url == "http://127.0.0.1:8080/v1"
