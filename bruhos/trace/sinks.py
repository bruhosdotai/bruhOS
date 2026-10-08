"""Trace sinks. Sinks run on a background thread so the control loop never waits on them."""

from __future__ import annotations

import json
import logging
import queue
import threading
import urllib.request
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable

from bruhos.schemas import Episode

from .hashing import canonical
from .writer import TraceWriter

log = logging.getLogger(__name__)


class Sink(ABC):
    name = "sink"

    @abstractmethod
    def emit(self, ep: Episode) -> dict[str, Any]:
        """Persist or publish a sealed episode. Returns a receipt."""


class LocalSink(Sink):
    name = "local"

    def __init__(self, root: str | Path = "traces"):
        self.writer = TraceWriter(root)

    def emit(self, ep: Episode) -> dict[str, Any]:
        return {"path": str(self.writer.write(ep))}


class IpfsSink(Sink):
    """Adds the episode (header + steps) to an IPFS node through the Kubo HTTP API."""

    name = "ipfs"

    def __init__(self, api: str = "http://127.0.0.1:5001", timeout: float = 30.0):
        self.api = api.rstrip("/")
        self.timeout = timeout

    def emit(self, ep: Episode) -> dict[str, Any]:
        payload = canonical({"episode": ep.header(), "steps": [s.to_dict() for s in ep.steps]})
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{ep.episode_id}.json\"\r\n"
                f"Content-Type: application/json\r\n\r\n").encode() + payload + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(f"{self.api}/api/v0/add?pin=true&cid-version=1", data=body, method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            res = json.loads(r.read())
        return {"cid": res["Hash"]}


class ChainSink(Sink):
    """Anchors ``trace_root`` plus skillpack / brain version hashes. Never video, never control.

    v0.1 builds the anchor payload; the transaction itself is delegated to ``submit`` so the
    chain client stays out of this package. Enabled in v0.3.
    """

    name = "chain"

    def __init__(self, submit: Callable[[dict[str, Any]], str] | None = None):
        self.submit = submit

    @staticmethod
    def anchor_payload(ep: Episode) -> dict[str, Any]:
        return {
            "episode_id": ep.episode_id,
            "trace_root": "0x" + ep.trace_root,
            "skillpack_hash": ep.skillpack_hash,
            "brain_version": ep.brain_version,
            "success": ep.success,
            "n_steps": len(ep.steps),
        }

    def emit(self, ep: Episode) -> dict[str, Any]:
        payload = self.anchor_payload(ep)
        if self.submit is None:
            raise NotImplementedError("ChainSink has no submit function configured (enabled in v0.3)")
        return {"tx": self.submit(payload), "payload": payload}


class SinkDispatcher:
    """Fire-and-forget fan-out to sinks on a daemon thread."""

    def __init__(self, sinks: list[Sink]):
        self.sinks = sinks
        self.receipts: list[tuple[str, str, dict[str, Any] | str]] = []
        self._q: queue.Queue[Episode | None] = queue.Queue()
        self._t = threading.Thread(target=self._run, name="bruhos-sinks", daemon=True)
        self._t.start()

    def submit(self, ep: Episode) -> None:
        self._q.put(ep)

    def _run(self) -> None:
        while (ep := self._q.get()) is not None:
            for s in self.sinks:
                try:
                    self.receipts.append((ep.episode_id, s.name, s.emit(ep)))
                except Exception as e:  # a failing sink must never take down the runtime
                    log.warning("sink %s failed for %s: %s", s.name, ep.episode_id, e)
                    self.receipts.append((ep.episode_id, s.name, f"error: {e}"))
            self._q.task_done()

    def flush(self, timeout: float | None = None) -> bool:
        done = threading.Event()
        threading.Thread(target=lambda: (self._q.join(), done.set()), daemon=True).start()
        return done.wait(timeout)

    def close(self) -> None:
        self._q.put(None)
