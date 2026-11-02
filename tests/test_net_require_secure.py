"""`require_secure` must not be bypassed by dropping the scheme.

The guard promises: "Refuse to send a credential over plain http/ws to anything
but this machine." On main it only looked at ``u.scheme in ("http", "ws")``, so a
URL written without a scheme -- ``192.168.1.50:8000/v1``, the way people type a
LAN box into a config -- had ``scheme == ""`` and sailed through:

    BLOCKED  http://192.168.1.50:8000/v1
    ALLOWED  192.168.1.50:8000/v1     <- same cleartext destination

Both `OpenAICompatPlanner` (API key) and the eliza bridge (token) call this with
whatever `base_url`/`url` the config holds.
"""

from __future__ import annotations

import pytest

from bruhos._net import require_secure


@pytest.mark.parametrize("url", ["192.168.1.50:8000/v1", "10.0.0.7/v1", "robot.lan:8000/v1"])
def test_schemeless_remote_host_is_refused(url: str):
    with pytest.raises(ValueError):
        require_secure(url, "the API key")


@pytest.mark.parametrize("url", ["127.0.0.1:8000/v1", "localhost:8000/v1", "[::1]:8000/v1"])
def test_schemeless_loopback_is_allowed(url: str):
    assert require_secure(url, "the API key") is None


def test_allow_insecure_still_overrides():
    assert require_secure("192.168.1.50:8000/v1", "the API key", allow_insecure=True) is None


def test_remote_https_is_allowed():
    assert require_secure("https://api.openai.com/v1", "the API key") is None
    assert require_secure("wss://bridge.example/sock", "the token") is None


def test_remote_cleartext_is_still_refused():
    for url in ("http://192.168.1.50:8000/v1", "ws://192.168.1.50/sock"):
        with pytest.raises(ValueError):
            require_secure(url, "the API key")
