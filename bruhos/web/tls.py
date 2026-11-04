"""Self-signed TLS for Safari / Chrome mic on LAN (getUserMedia needs a secure context)."""

from __future__ import annotations

import shutil
import socket
import subprocess
from pathlib import Path


def _lan_ips() -> list[str]:
    ips = {"127.0.0.1"}
    try:
        out = subprocess.check_output(["hostname", "-I"], text=True)
        ips.update(p for p in out.split() if ":" not in p)
    except (OSError, subprocess.CalledProcessError):
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(ips)


def ensure_self_signed(cert_dir: Path, common_name: str = "bruhos.local") -> tuple[Path, Path]:
    cert_dir = Path(cert_dir)
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert = cert_dir / "cert.pem"
    key = cert_dir / "key.pem"
    stamp = cert_dir / "sans.txt"
    ips = _lan_ips()
    sans_blob = "\n".join(["DNS:localhost", f"DNS:{common_name}", *[f"IP:{ip}" for ip in ips]])
    if cert.is_file() and key.is_file() and stamp.is_file() and stamp.read_text() == sans_blob:
        return cert, key
    openssl = shutil.which("openssl")
    if not openssl:
        raise RuntimeError("openssl not found (needed for --tls / Safari mic)")
    cfg = cert_dir / "openssl.cnf"
    alt_lines = ["DNS.1=localhost", f"DNS.2={common_name}"]
    for i, ip in enumerate(ips, start=1):
        alt_lines.append(f"IP.{i}={ip}")
    cfg.write_text(
        "[req]\n"
        "distinguished_name=req_distinguished_name\n"
        "x509_extensions=v3_req\n"
        "prompt=no\n"
        "[req_distinguished_name]\n"
        f"CN={common_name}\n"
        "[v3_req]\n"
        "subjectAltName=@alt\n"
        "basicConstraints=CA:FALSE\n"
        "keyUsage=digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth\n"
        "[alt]\n"
        + "\n".join(alt_lines)
        + "\n"
    )
    subprocess.run(
        [
            openssl, "req", "-x509", "-newkey", "rsa:2048", "-sha256", "-days", "825",
            "-nodes", "-keyout", str(key), "-out", str(cert), "-config", str(cfg),
        ],
        check=True,
        capture_output=True,
    )
    stamp.write_text(sans_blob)
    return cert, key
