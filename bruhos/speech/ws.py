"""Minimal WebSocket client (RFC 6455) using the stdlib. Text frames only."""

from __future__ import annotations

import base64
import os
import socket
import ssl
import struct
import threading
from urllib.parse import urlparse


class WebSocketError(RuntimeError):
    pass


class WebSocket:
    def __init__(self, url: str, timeout: float = 30.0):
        parsed = urlparse(url)
        if parsed.scheme not in ("ws", "wss"):
            raise WebSocketError(f"not a websocket url: {url}")
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        raw = socket.create_connection((host, port), timeout=timeout)
        if parsed.scheme == "wss":
            ctx = ssl.create_default_context()
            raw = ctx.wrap_socket(raw, server_hostname=host)
        key = base64.b64encode(os.urandom(16)).decode()
        host_hdr = host if port in (80, 443) else f"{host}:{port}"
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host_hdr}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        raw.sendall(req.encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = raw.recv(4096)
            if not chunk:
                raw.close()
                raise WebSocketError("websocket handshake closed")
            buf += chunk
            if len(buf) > 65536:
                raw.close()
                raise WebSocketError("websocket handshake too large")
        head, self._buf = buf.split(b"\r\n\r\n", 1)
        status = head.split(b"\r\n", 1)[0].decode("latin1", "replace")
        if " 101 " not in status:
            raw.close()
            raise WebSocketError(status)
        self._sock = raw
        self._sock.settimeout(timeout)
        self._send_lock = threading.Lock()

    def send_text(self, text: str) -> None:
        self._send(0x1, text.encode())

    def recv_text(self) -> str | None:
        """Next text frame. None when the server closes. Pings are answered."""
        while True:
            opcode, payload = self._recv()
            if opcode is None:
                return None
            if opcode == 0x1:
                return payload.decode("utf-8", "replace")
            if opcode == 0x8:
                return None
            if opcode == 0x9:
                self._send(0xA, payload)
            # ignore pong and binary

    def close(self) -> None:
        try:
            self._send(0x8, b"")
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass

    def settimeout(self, timeout: float | None) -> None:
        self._sock.settimeout(timeout)

    def _send(self, opcode: int, payload: bytes) -> None:
        with self._send_lock:
            self._send_unlocked(opcode, payload)

    def _send_unlocked(self, opcode: int, payload: bytes) -> None:
        n = len(payload)
        mask = os.urandom(4)
        header = bytearray([0x80 | opcode])
        if n < 126:
            header.append(0x80 | n)
        elif n < 65536:
            header.append(0x80 | 126)
            header += struct.pack("!H", n)
        else:
            header.append(0x80 | 127)
            header += struct.pack("!Q", n)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self._sock.sendall(bytes(header) + mask + masked)

    def _recv(self) -> tuple[int | None, bytes]:
        hdr = self._exact(2)
        if hdr is None:
            return None, b""
        b0, b1 = hdr[0], hdr[1]
        opcode = b0 & 0x0F
        ln = b1 & 0x7F
        if ln == 126:
            ln = int.from_bytes(self._exact(2) or b"", "big")
        elif ln == 127:
            ln = int.from_bytes(self._exact(8) or b"", "big")
        if b1 & 0x80:
            mask = self._exact(4) or b""
            data = self._exact(ln) or b""
            data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
        else:
            data = self._exact(ln) or b""
        return opcode, data

    def _exact(self, n: int) -> bytes | None:
        while len(self._buf) < n:
            try:
                chunk = self._sock.recv(max(4096, n - len(self._buf)))
            except TimeoutError:
                raise
            if not chunk:
                return None
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out
