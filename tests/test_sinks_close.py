"""SinkDispatcher must not lose episodes silently, nor hang after close().

Reproduced on main:

1. ``submit()`` after ``close()`` puts the episode on a queue whose consumer has
   already exited, so the episode is dropped with no error and no receipt.
2. ``_run()`` breaks out of the loop on the ``None`` sentinel *before*
   ``task_done()``, so ``unfinished_tasks`` never reaches zero and ``flush()``
   after ``close()`` waits forever -- an orderly shutdown looks like a hang.
"""

from __future__ import annotations

import threading
import time

import pytest

from bruhos.schemas import Episode
from bruhos.trace.sinks import Sink, SinkDispatcher


class RecordingSink(Sink):
    name = "recording"

    def __init__(self, delay: float = 0.0):
        self.seen: list[str] = []
        self.delay = delay
        self._lock = threading.Lock()

    def emit(self, ep: Episode) -> dict:
        if self.delay:
            time.sleep(self.delay)
        with self._lock:
            self.seen.append(ep.episode_id)
        return {"ok": True}


def _episode(eid: str) -> Episode:
    return Episode(episode_id=eid, instruction="noop", brain="rules", body="sim",
                   started_at=0.0, plan=[], steps=[], success=True, recoveries=0,
                   falls=0, ended_at=1.0, end_reason="completed",
                   skillpack_hash="sha256:x", brain_version="t", trace_root="0" * 64)


def test_flush_after_close_returns_true():
    d = SinkDispatcher([RecordingSink()])
    d.submit(_episode("e1"))
    assert d.flush(timeout=5) is True
    d.close()
    assert d.flush(timeout=2) is True, "flush() hangs after close(): sentinel never task_done()"


def test_submit_after_close_is_rejected_instead_of_dropped():
    d = SinkDispatcher([RecordingSink()])
    d.close()
    with pytest.raises(RuntimeError):
        d.submit(_episode("ghost"))
    assert d.flush(timeout=1) is True


def test_flush_waits_for_every_pending_episode():
    sink = RecordingSink(delay=0.05)
    d = SinkDispatcher([sink])
    for i in range(5):
        d.submit(_episode(f"e{i}"))
    assert d.flush(timeout=5) is True
    assert sink.seen == ["e0", "e1", "e2", "e3", "e4"]
    d.close()
    assert d.flush(timeout=2) is True
