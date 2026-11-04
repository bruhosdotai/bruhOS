"""The [walk] heartbeat must stay under the bridge deadman at load time.

``ElizaBridgeBody._walk`` re-sends ``walk.set`` every ``heartbeat_s`` while a
walk primitive runs, and the bridge stops motion if no motion command arrives
within its deadman timeout (1 s on AiNex — see the module docstring and the
``heartbeat_s`` comment in configs/eliza_ainex.toml). On ``main`` any value
was accepted at load: with ``heartbeat_s = 1.5`` the first re-send lands after
the deadman has already fired, so every walk is cut mid-motion and the
primitive reports failure with nothing pointing at the config value; with
``heartbeat_s <= 0`` the loop either floods the bridge or ``time.sleep``
raises mid-walk.
"""

import tomllib

import pytest

pytest.importorskip("websockets")

from bruhos.body.eliza_bridge import ElizaBridgeBody  # noqa: E402


def _construct(monkeypatch, cfg):
    # Construction must not need a live bridge for config validation.
    monkeypatch.setattr(ElizaBridgeBody, "_connect", lambda self: None)
    monkeypatch.setattr(ElizaBridgeBody, "_action_durations", lambda self: {})
    return ElizaBridgeBody(cfg)


def _cfg(**walk):
    return {"body": {"url": "ws://127.0.0.1:1"}, "walk": walk,
            "actions": {"stand": "stand"}}


@pytest.mark.parametrize("heartbeat_s", [1.0, 1.5, 30.0, 0.0, -0.5])
def test_heartbeat_outside_deadman_range_is_rejected(monkeypatch, heartbeat_s):
    with pytest.raises(ValueError, match="heartbeat_s"):
        _construct(monkeypatch, _cfg(heartbeat_s=heartbeat_s))


@pytest.mark.parametrize("heartbeat_s", [0.05, 0.25, 0.99])
def test_heartbeat_inside_deadman_range_is_accepted(monkeypatch, heartbeat_s):
    body = _construct(monkeypatch, _cfg(heartbeat_s=heartbeat_s))
    assert body.walk["heartbeat_s"] == heartbeat_s


def test_shipped_eliza_config_still_constructs(monkeypatch):
    with open("configs/eliza_ainex.toml", "rb") as f:
        cfg = tomllib.load(f)
    body = _construct(monkeypatch, cfg)
    assert body.walk["heartbeat_s"] == 0.25
