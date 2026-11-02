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
    # Without "://" there is no authority, so urlsplit reads the host as the scheme
    # ("192.168.1.50:8000" -> scheme='192.168.1.50', hostname=None) and the guard
    # let cleartext to a LAN box through. Parse a scheme-less URL as a bare authority.
    u = urlsplit(url if "://" in url else "//" + url)
    if u.scheme in ("https", "wss"):
        return
    if u.scheme not in ("http", "ws", ""):
        return  # some other scheme (sftp, file, ...) -- not this guard's business
    if is_loopback(u.hostname) or allow_insecure:
        return
    cleartext = f"{u.scheme or 'http'}://{u.hostname}"
    raise ValueError(f"refusing to send {what} in cleartext to {cleartext}; use "
                     f"{'https' if (u.scheme or 'http') == 'http' else 'wss'}, a loopback address "
                     "or an SSH tunnel (or set allow_insecure=True if this network is trusted)")
