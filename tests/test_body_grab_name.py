"""Hardware bodies must record WHAT they grabbed, not just that they grabbed.

Reproduces the andyboss1995 review on #204: on TonyPi and the eliza bridge,
``grab`` set ``holding = "object"``, so the held-target checks added by #202
(walk_toward) and #204 (look_at/detect) — which parse the held NAME — could
never match on hardware bodies. A held object is invisible to observe() (the
sim hides it, and the bridge drops it from perception once grasped), so
look_at/detect kept failing after a successful pickup. The bodies now name the
grabbed object from the nearest graspable object seen before the grab, and
refuse a grab while the hand is already full (the sim already does).
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
from typing import ClassVar

import pytest

pytest.importorskip("websockets")
from websockets.sync.server import serve

from bruhos.body.base import grabbed_name
from bruhos.body.eliza_bridge import ElizaBridgeBody
from bruhos.body.tonypi import TonyPiBody
from bruhos.schemas import Detection, LineReading, SkillCall, Status
from bruhos.skills import SkillContext, run_skill

PROFILE = {"profile": {"actions": {"groups": {
    "stand": {"duration_s": 0.05}, "wave": {"duration_s": 0.05}, "grab": {"duration_s": 0.05}}}}}


def _red_block(dist=0.09, bearing=0.0):
    return Detection("block", "red", bearing, dist)


def _blue_block(dist=0.8, bearing=10.0):
    return Detection("block", "blue", bearing, dist)


# ---------------------------------------------------------------- eliza bridge


class NamingBridge:
    """Fake eliza bridge whose perception drops the grasped object."""

    def __init__(self, entities):
        self.entities = list(entities)
        self.commands: list[tuple[str, dict]] = []
        self.server = serve(self._handle, "127.0.0.1", 0)
        self.url = f"ws://127.0.0.1:{self.server.socket.getsockname()[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _event(self, ws, event, data):
        ws.send(json.dumps({"type": "event", "event": event, "timestamp": "t",
                            "backend": "fake", "data": data}))

    def _handle(self, ws):
        self._event(ws, "session.hello", {"capabilities": {}})
        self._event(ws, "telemetry.basic", {"imu_roll": 0.0, "imu_pitch": 0.0})
        self._event(ws, "telemetry.perception", {"entities": self.entities})
        for raw in ws:
            msg = json.loads(raw)
            self.commands.append((msg["command"], msg["payload"]))
            data = PROFILE if msg["command"] == "profile.describe" else {}
            ws.send(json.dumps({"type": "response", "request_id": msg["request_id"],
                                "timestamp": "t", "ok": True, "backend": "fake",
                                "message": "ok", "data": data}))
            if msg["command"] == "action.play" and msg["payload"].get("name") == "grab":
                # The grasped object leaves the scene: a held object is not in perception.
                self.entities = []
                self._event(ws, "telemetry.perception", {"entities": []})

    def close(self):
        self.server.shutdown()


def _bridge_body(bridge):
    cfg = {"body": {"url": bridge.url, "fall_deg": 55.0},
           "walk": {"forward_s": 0.05, "turn_s": 0.05, "heartbeat_s": 0.01},
           "actions": {"stand": "stand", "wave": "wave", "grab": "grab", "release": "release"}}
    return ElizaBridgeBody(cfg)


def _entities(*specs):
    # (label, x, y) in the bridge robot frame: +x forward, +y left.
    return [{"label": label, "x": x, "y": y, "confidence": 0.9} for label, x, y in specs]


@pytest.fixture
def red_bridge():
    b = NamingBridge(_entities(("red block", 0.09, 0.0), ("blue block", 0.8, 0.3)))
    yield b
    b.close()


def _wait_entities(body, timeout=2.0):
    end = time.monotonic() + timeout
    while not body._entities and time.monotonic() < end:
        time.sleep(0.01)
    assert body._entities


def _wait_entities_empty(body, timeout=2.0):
    end = time.monotonic() + timeout
    while body._entities and time.monotonic() < end:
        time.sleep(0.01)
    assert body._entities == []


def test_bridge_grab_names_the_nearest_seen_block(red_bridge):
    body = _bridge_body(red_bridge)
    try:
        _wait_entities(body)
        assert body.observe().holding is None
        assert body.act("grab")
        assert body.observe().holding == "red_block"
    finally:
        body.close()


def test_bridge_grab_skips_the_box_for_a_block_in_reach():
    bridge = NamingBridge(_entities(("box", 0.09, 0.0), ("red block", 0.15, 0.0)))
    body = _bridge_body(bridge)
    try:
        _wait_entities(body)
        assert body.act("grab")
        assert body.observe().holding == "red_block"
    finally:
        body.close()
        bridge.close()


def test_bridge_grab_falls_back_to_object_when_nothing_in_reach():
    bridge = NamingBridge(_entities(("red block", 2.0, 0.0)))
    body = _bridge_body(bridge)
    try:
        _wait_entities(body)
        assert body.act("grab")
        assert body.observe().holding == "object"
    finally:
        body.close()
        bridge.close()


def test_bridge_grab_refuses_a_full_hand(red_bridge):
    body = _bridge_body(red_bridge)
    try:
        _wait_entities(body)
        assert body.act("grab")
        assert body.observe().holding == "red_block"
        assert not body.act("grab")
        assert body.observe().holding == "red_block"
    finally:
        body.close()


def test_bridge_grab_and_release_report_the_held_name(red_bridge):
    # On main the skill layer has no held-target check yet (#204 adds it);
    # what the body fix changes here is the name the skills report and the
    # trace records: "red_block" instead of the generic "object".
    body = _bridge_body(red_bridge)
    try:
        _wait_entities(body)
        ctx = SkillContext(body)
        out = run_skill(ctx, SkillCall("grab", {}))
        assert out.status is Status.OK and out.data == {"holding": "red_block"}
        out = run_skill(ctx, SkillCall("release", {}))
        assert out.status is Status.OK and out.data == {"released": "red_block"}
    finally:
        body.close()


def _held_matches(holding: str | None, target: str) -> bool:
    """The held-name rule #204's ``_held`` uses, inlined: the body fix is
    only complete once a held name parses back to its target this way."""
    if not holding:
        return False
    color, _, label = holding.partition("_")
    if not label:
        color, label = None, holding
    return Detection(label, color).matches(target)


def test_bridge_held_name_matches_target_under_the_204_rule(red_bridge):
    body = _bridge_body(red_bridge)
    try:
        _wait_entities(body)
        assert body.act("grab")
        _wait_entities_empty(body)
        holding = body.observe().holding
        assert holding == "red_block"
        assert _held_matches(holding, "red_block")
        assert _held_matches(holding, "red")
        assert _held_matches(holding, "block")
        assert not _held_matches(holding, "blue_block")
        # The old value never matched any named target.
        assert not _held_matches("object", "red_block")
    finally:
        body.close()


def test_grabbed_name_picks_nearest_graspable():
    assert grabbed_name([_blue_block(), _red_block()]) == "red_block"
    assert grabbed_name([Detection("box", None, 0.0, 0.09), _red_block(0.15)]) == "red_block"
    assert grabbed_name([Detection("ball", "yellow", 0.0, 0.09)]) is None
    assert grabbed_name([_red_block(2.0)]) is None
    assert grabbed_name([]) is None


# ------------------------------------------------------------------- TonyPi


class _FakeAGC(types.ModuleType):
    calls: ClassVar[list[str]] = []

    @staticmethod
    def runActionGroup(group, **kw):
        _FakeAGC.calls.append(group)


class _FakeCap:
    def read(self):
        return True, None

    def release(self):
        pass


class _FakeDetector:
    def __init__(self, detections):
        self.detections = list(detections)

    def detect(self, frame):
        return list(self.detections)


class _FakeLine:
    def read(self, frame):
        return LineReading()


@pytest.fixture
def tonypi(monkeypatch):
    monkeypatch.setitem(sys.modules, "hiwonder.ActionGroupControl", _FakeAGC("hiwonder.ActionGroupControl"))
    _FakeAGC.calls = []
    cfg = {"body": {"name": "tonypi"}, "camera": {"enabled": False}, "imu": {"enabled": False},
           "actions": {"grab": "grab", "release": "release"}}
    body = TonyPiBody(cfg)
    body.cap = _FakeCap()
    body.detector = _FakeDetector([_red_block(), _blue_block()])
    body.line = _FakeLine()
    yield body
    body.close()


def test_tonypi_grab_names_the_nearest_seen_block(tonypi):
    assert tonypi.observe().holding is None
    assert tonypi.act("grab")
    assert tonypi.observe().holding == "red_block"


def test_tonypi_grab_refuses_a_full_hand(tonypi):
    tonypi.observe()  # the runtime always observes before a skill acts
    assert tonypi.act("grab")
    assert tonypi.observe().holding == "red_block"
    assert not tonypi.act("grab")
    assert tonypi.observe().holding == "red_block"


def test_tonypi_grab_falls_back_to_object_when_nothing_seen(tonypi):
    tonypi.detector.detections = []
    tonypi.observe()
    assert tonypi.act("grab")
    assert tonypi.observe().holding == "object"


def test_tonypi_grab_and_release_report_the_held_name(tonypi):
    tonypi.observe()  # the runtime always observes before a skill acts
    ctx = SkillContext(tonypi)
    out = run_skill(ctx, SkillCall("grab", {}))
    assert out.status is Status.OK and out.data == {"holding": "red_block"}
    out = run_skill(ctx, SkillCall("release", {}))
    assert out.status is Status.OK and out.data == {"released": "red_block"}
