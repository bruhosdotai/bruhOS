"""Small network helpers shared by brains and bodies."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit


def is_loopback(host: str | None) -> bool:
    host = (host or "").strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def require_secure(url: str, what: str, allow_insecure: bool = False) -> None:
    """Refuse to send a credential over plain http/ws to anything but this machine."""
    u = urlsplit(url)
    if u.scheme in ("http", "ws") and not is_loopback(u.hostname) and not allow_insecure:
        raise ValueError(f"refusing to send {what} in cleartext to {u.scheme}://{u.hostname}; use "
                         f"{'https' if u.scheme == 'http' else 'wss'}, a loopback address or an SSH tunnel "
                         "(or set allow_insecure=True if this network is trusted)")
