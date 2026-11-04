"""Thin LAN web shell: PTT → STT → LLM reply → TTS + single-shot camera see."""

from __future__ import annotations

import json
import mimetypes
import ssl
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

from bruhos.listen import ChatReply, Conversation, NormalizeError, transcribe_audio
from bruhos.speech import SpeechConfig, make_tts
from bruhos.web.tls import ensure_self_signed

STATIC = Path(__file__).resolve().parent / "static"


def _safe_name(name: str) -> str | None:
    name = unquote(name).lstrip("/")
    if not name or name.endswith("/") or ".." in name.split("/") or name.startswith("."):
        return None
    if "/" in name:
        return None
    return name


def _safe_see_file(path: str) -> tuple[str, str] | None:
    """Parse ``/see/<job_id>/<file>`` → (job_id, filename) or None."""
    parts = [p for p in unquote(path).split("/") if p]
    if len(parts) != 3 or parts[0] != "see":
        return None
    job_id, name = parts[1], parts[2]
    if ".." in job_id or ".." in name or "/" in job_id or not name or name.startswith("."):
        return None
    if not all(c.isalnum() or c in "-_" for c in job_id):
        return None
    if not (name.endswith(".jpg") or name.endswith(".json")):
        return None
    return job_id, name


class BruhOSHandler(BaseHTTPRequestHandler):
    server_version = "bruhOSWeb/0.2"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} {fmt % args}")

    def _json(self, code: int, obj: dict[str, Any]) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, *, no_store: bool = False) -> None:
        if not path.is_file():
            self._json(404, {"error": "not found"})
            return
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Accept-Ranges", "bytes")
        if no_store:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._file(STATIC / "index.html")
            return
        if path == "/api/head":
            hc = getattr(self.server, "head", None)
            if hc is None:
                self._json(503, {"error": "head control not configured (need [head] in config)"})
                return
            self._json(200, hc.status())
            return
        if path in ("/v1/skills", "/api/skills"):
            from bruhos.agent.tools import skills_as_eliza_actions
            from bruhos.skills import CATALOG, catalog_hash

            self._json(200, {
                "skillpack_hash": catalog_hash(),
                "skills": CATALOG,
                "eliza_actions": skills_as_eliza_actions(),
            })
            return
        if path == "/api/see" or path == "/api/see/latest":
            hub = getattr(self.server, "see", None)
            if hub is None:
                self._json(503, {"error": "see pipeline not configured (need [camera] in config)"})
                return
            job = hub.latest()
            if job is None:
                self._json(404, {"error": "no see jobs yet"})
                return
            self._json(200, job.to_dict())
            return
        if path.startswith("/api/see/"):
            hub = getattr(self.server, "see", None)
            if hub is None:
                self._json(503, {"error": "see pipeline not configured"})
                return
            job_id = path[len("/api/see/"):].strip("/")
            if not job_id or "/" in job_id:
                self._json(400, {"error": "bad job id"})
                return
            job = hub.get(job_id)
            if job is None:
                self._json(404, {"error": "unknown job"})
                return
            self._json(200, job.to_dict())
            return
        see = _safe_see_file(path)
        if see:
            job_id, name = see
            hub = getattr(self.server, "see", None)
            root = (hub.see_root if hub else self.server.http_dir / "see")  # type: ignore[attr-defined]
            self._file(root / job_id / name, no_store=True)
            return
        if path.startswith("/audio/"):
            name = _safe_name(path[len("/audio/"):])
            if not name:
                self._json(400, {"error": "bad path"})
                return
            self._file(self.server.http_dir / name)  # type: ignore[attr-defined]
            return
        name = _safe_name(path)
        if name:
            self._file(self.server.http_dir / name)  # type: ignore[attr-defined]
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        if path == "/api/ptt":
            self._ptt(body)
            return
        if path == "/api/say":
            self._say(body)
            return
        if path == "/api/reset":
            conv: Conversation = self.server.conversation  # type: ignore[attr-defined]
            conv.clear()
            self._json(200, {"ok": True, "turns": 0})
            return
        if path == "/api/see":
            hub = getattr(self.server, "see", None)
            if hub is None:
                self._json(503, {"error": "see pipeline not configured (need [camera] in config)"})
                return
            try:
                job = hub.start()
            except RuntimeError as e:
                self._json(409, {"error": str(e)})
                return
            except Exception as e:  # noqa: BLE001
                self._json(500, {"error": str(e)})
                return
            self._json(200, job.to_dict())
            return
        if path == "/api/head":
            hc = getattr(self.server, "head", None)
            if hc is None:
                self._json(503, {"error": "head control not configured (need [head] in config)"})
                return
            hub = getattr(self.server, "see", None)
            try:
                payload = json.loads(body.decode() or "{}")
            except json.JSONDecodeError:
                self._json(400, {"error": "JSON body required"})
                return
            want_preview = bool(payload.get("preview", True))
            # Always move the head first — never block poses on a long VL job.
            try:
                st = hc.apply(payload)
            except ValueError as e:
                self._json(400, {"error": str(e)})
                return
            except Exception as e:  # noqa: BLE001
                self._json(502, {"error": str(e)})
                return
            if want_preview and hub is not None:
                try:
                    move_ms = int((st.get("move_ms") if st else None) or 500)
                    settle = move_ms / 1000.0 + 0.25
                    prev = hub.preview_still(settle_s=settle)
                    st["preview"] = prev["url"]
                    st["preview_bytes"] = prev["bytes"]
                except Exception as e:  # noqa: BLE001
                    st["preview_error"] = str(e)[:240]
            self._json(200, st)
            return
        if path in ("/v1/skills/run", "/api/skills/run"):
            self._skill_run(body)
            return
        self._json(404, {"error": "not found"})

    def _transcribe(self, body: bytes, suffix: str, cfg: SpeechConfig) -> str:
        if self.server.asr_factory:  # type: ignore[attr-defined]
            from bruhos.listen.normalize import normalize_to_wav
            wav = normalize_to_wav(body, suffix=suffix)
            try:
                return self.server.asr_factory().transcribe(str(wav))  # type: ignore[attr-defined]
            finally:
                wav.unlink(missing_ok=True)
        return transcribe_audio(body, cfg, suffix=suffix)

    def _ptt(self, body: bytes) -> None:
        if not body:
            self._json(400, {"error": "empty body"})
            return
        filename = self.headers.get("X-Filename") or "ptt.webm"
        suffix = Path(filename).suffix or ".webm"
        ctype = self.headers.get("Content-Type", "")
        if "wav" in ctype and suffix == ".webm":
            suffix = ".wav"
        cfg: SpeechConfig = self.server.speech  # type: ignore[attr-defined]
        try:
            text = self._transcribe(body, suffix, cfg)
        except NormalizeError as e:
            self._json(503, {"error": str(e)})
            return
        except Exception as e:  # noqa: BLE001
            self._json(502, {"error": f"stt failed: {e}"})
            return
        if text.strip() in ("[BLANK_AUDIO]", "[blank_audio]"):
            text = ""
        conv: Conversation = self.server.conversation  # type: ignore[attr-defined]
        if not text.strip():
            self._json(200, {
                "text": "",
                "reply": "",
                "assumed_command": True,
                "executed": False,
                "tts": None,
                "turns": conv.turns,
            })
            return

        chat: ChatReply | None = getattr(self.server, "chat", None)
        reply = ""
        tts_path = None
        turns = conv.turns
        if chat is not None:
            try:
                reply = chat.reply(text, history=conv.snapshot())
            except Exception as e:  # noqa: BLE001
                self._json(502, {"error": f"llm failed: {e}", "text": text, "turns": conv.turns})
                return
            if reply:
                turns = conv.append(text, reply)
            if reply and getattr(self.server, "speak_replies", True):
                try:
                    sc = replace(
                        cfg,
                        tts=cfg.tts if cfg.tts != "local" else "openai",
                        playback="http",
                        http_dir=str(self.server.http_dir),  # type: ignore[attr-defined]
                    )
                    make_tts(sc).say(reply)
                    tts_path = "/tts.wav"
                except Exception as e:  # noqa: BLE001
                    self._json(200, {
                        "text": text,
                        "reply": reply,
                        "assumed_command": True,
                        "executed": False,
                        "tts": None,
                        "tts_error": str(e),
                        "turns": turns,
                    })
                    return

        self._json(200, {
            "text": text,
            "reply": reply,
            "assumed_command": True,
            "executed": False,
            "tts": tts_path,
            "turns": turns,
        })

    def _say(self, body: bytes) -> None:
        try:
            payload = json.loads(body.decode() or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "JSON body required"})
            return
        text = (payload.get("text") or "").strip()
        if not text:
            self._json(400, {"error": "text required"})
            return
        cfg: SpeechConfig = self.server.speech  # type: ignore[attr-defined]
        sc = replace(
            cfg,
            tts=cfg.tts if cfg.tts != "local" else "openai",
            playback="http",
            http_dir=str(self.server.http_dir),  # type: ignore[attr-defined]
        )
        try:
            make_tts(sc).say(text)
        except Exception as e:  # noqa: BLE001
            self._json(502, {"error": f"tts failed: {e}"})
            return
        self._json(200, {"ok": True, "path": "/tts.wav"})

    def _skill_run(self, body: bytes) -> None:
        hub = getattr(self.server, "skills", None)
        if hub is None:
            self._json(503, {"error": "skill hub not configured (need agent body / SkillHub)"})
            return
        try:
            payload = json.loads(body.decode() or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "JSON body required"})
            return
        skill = payload.get("skill") or payload.get("name")
        args = payload.get("args") or payload.get("parameters") or {}
        if not skill or not isinstance(args, dict):
            self._json(400, {"error": "need skill + args object"})
            return
        try:
            result = hub.dispatch({"skill": skill, "args": args}, brain="http")
        except Exception as e:  # noqa: BLE001
            self._json(500, {"error": str(e)})
            return
        code = 200 if result.outcome.status.value != "rejected" else 400
        self._json(code, result.to_dict())


class BruhOSServer(ThreadingHTTPServer):
    def __init__(
        self,
        host: str,
        port: int,
        http_dir: Path,
        speech: SpeechConfig,
        chat: ChatReply | None = None,
        *,
        speak_replies: bool = True,
        max_turns: int = 12,
        asr_factory: Callable[[], Any] | None = None,
        conversation: Conversation | None = None,
        see: Any | None = None,
        head: Any | None = None,
        skills: Any | None = None,
    ):
        self.http_dir = Path(http_dir)
        self.http_dir.mkdir(parents=True, exist_ok=True)
        self.speech = speech
        self.chat = chat
        self.speak_replies = speak_replies
        self.conversation = conversation or Conversation(max_turns=max_turns)
        self.asr_factory = asr_factory
        self.see = see
        self.head = head
        self.skills = skills
        super().__init__((host, port), BruhOSHandler)


def serve(
    host: str,
    port: int,
    http_dir: str | Path,
    speech: SpeechConfig,
    chat: ChatReply | None = None,
    *,
    speak_replies: bool = True,
    max_turns: int = 12,
    tls: bool = False,
    cert_dir: str | Path | None = None,
    see_cfg: dict[str, Any] | None = None,
    skills: Any | None = None,
) -> None:
    from bruhos.web.head_ctrl import HeadControl
    from bruhos.web.see import SeeHub

    http_path = Path(http_dir)
    see_hub = None
    head_ctrl = None
    if see_cfg is not None and see_cfg.get("camera", {}).get("enabled", True) is not False:
        if see_cfg.get("camera") or see_cfg.get("scan"):
            see_hub = SeeHub(http_path, see_cfg)
    if see_cfg is not None and see_cfg.get("head"):
        head_ctrl = HeadControl(see_cfg)

    httpd = BruhOSServer(
        host, port, http_path, speech, chat,
        speak_replies=speak_replies, max_turns=max_turns, see=see_hub, head=head_ctrl,
        skills=skills,
    )
    scheme = "http"
    if tls:
        cdir = Path(cert_dir) if cert_dir else http_path / "certs"
        cert, key = ensure_self_signed(cdir)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(cert), str(key))
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"
        print(f"bruhOS web  {scheme}://<pi-ip>:{port}/  (self-signed — Safari: visit, trust cert, then mic works)")
        print(f"  cert={cert}")
    model = chat.model if chat else "(no llm)"
    see_note = "see=on" if see_hub else "see=off"
    head_note = "head=on" if head_ctrl else "head=off"
    print(
        f"bruhOS web  {scheme}://{host}:{port}/  audio={httpd.http_dir}  "
        f"llm={model}  {see_note}  {head_note}  (LAN open, no auth)"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        httpd.server_close()
