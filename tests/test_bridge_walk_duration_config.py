"""The [walk] durations must be usable at load time.

``ElizaBridgeBody._walk`` runs a walk primitive for ``forward_s`` (or
``turn_s``) seconds and then stops. Those durations are local: unlike
``speed`` / ``height`` / ``forward_x`` / ``turn_yaw`` they are never sent to
the bridge, so the bridge cannot reject a bad value. On ``main`` any value
was accepted at load: with ``forward_s = 0`` (or negative, or NaN) the
start/stop pair goes out at once and ``act("forward")`` still returns True
in about a millisecond — a walk is certified as done while the robot never
moved, and ``walk_toward`` burns its whole step budget on no-op forwards.
With ``inf`` the walk never ends on its own. Reject non-finite and
non-positive durations at construction, next to the ``heartbeat_s`` check.
"""

import math
import tomllib

import pytest

pytest.importorskip("websockets")

from bruhos.body.eliza_bridge import ElizaBridgeBody


def _construct(monkeypatch, cfg):
    # Construction must not need a live bridge for config validation.
    monkeypatch.setattr(ElizaBridgeBody, "_connect", lambda self: None)
    monkeypatch.setattr(ElizaBridgeBody, "_action_durations", lambda self: {})
    return ElizaBridgeBody(cfg)


def _cfg(**walk):
    return {"body": {"url": "ws://127.0.0.1:1"}, "walk": walk,
            "actions": {"stand": "stand"}}


@pytest.mark.parametrize("key", ["forward_s", "turn_s"])
@pytest.mark.parametrize("bad", [0.0, -0.5, math.nan, math.inf])
def test_non_positive_or_nonfinite_walk_duration_is_rejected(monkeypatch, key, bad):
    with pytest.raises(ValueError, match=key):
        _construct(monkeypatch, _cfg(**{key: bad}))


@pytest.mark.parametrize("key", ["forward_s", "turn_s"])
@pytest.mark.parametrize("good", [0.05, 1.0, 30.0])
def test_positive_finite_walk_duration_is_accepted(monkeypatch, key, good):
    body = _construct(monkeypatch, _cfg(**{key: good}))
    assert body.walk[key] == good


def test_shipped_eliza_config_still_constructs(monkeypatch):
    with open("configs/eliza_ainex.toml", "rb") as f:
        cfg = tomllib.load(f)
    assert _construct(monkeypatch, cfg).walk["forward_s"] == 1.5
