"""WiFi ranging for roam: associated station RSSI + nl80211 neighbor sweeps + pose-graph fit.

Does not open Broadcom vendor dcmds. One epoch per finished scan; dual-band pairs share (x,y).
"""

from __future__ import annotations

import math
import os
import re
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# Minimal nl80211 constants
NLMSG_ERROR = 0x2
NLMSG_DONE = 0x3
NLM_F_REQUEST = 0x1
NLM_F_ACK = 0x4
NLM_F_ROOT = 0x100
NLM_F_MATCH = 0x200
NLM_F_DUMP = NLM_F_ROOT | NLM_F_MATCH
NETLINK_GENERIC = 16

NL80211_CMD_GET_WIPHY = 1
NL80211_CMD_GET_INTERFACE = 5
NL80211_CMD_TRIGGER_SCAN = 33
NL80211_CMD_NEW_SCAN_RESULTS = 34
NL80211_CMD_GET_SCAN = 32
NL80211_CMD_SCAN_ABORTED = 35
NL80211_CMD_GET_STATION = 17

NL80211_ATTR_WIPHY = 1
NL80211_ATTR_IFINDEX = 3
NL80211_ATTR_MAC = 6
NL80211_ATTR_GENERATION = 46
NL80211_ATTR_BSS = 47
NL80211_ATTR_SCAN_FREQUENCIES = 44
NL80211_ATTR_SCAN_SSIDS = 45
NL80211_ATTR_STA_INFO = 21

NL80211_BSS_BSSID = 1
NL80211_BSS_FREQUENCY = 2
NL80211_BSS_SIGNAL_MBM = 7
NL80211_BSS_INFORMATION_ELEMENTS = 11

NL80211_STA_INFO_SIGNAL = 7
NL80211_STA_INFO_SIGNAL_AVG = 13

CTRL_CMD_GETFAMILY = 3
CTRL_ATTR_FAMILY_ID = 1
CTRL_ATTR_FAMILY_NAME = 2

PATH_LOSS_N = 2.5
MIN_SPAN_DB = 6.0
HUBER_DB = 8.0
MIN_DIST_M = 0.3


@dataclass
class BssReading:
    bssid: str
    ssid: str
    freq_mhz: int
    dbm: float


@dataclass
class WifiEpoch:
    t_mid: float
    x: float
    y: float
    heading_deg: float
    readings: list[BssReading]


@dataclass
class ApSite:
    id: str
    ssid: str
    x: float
    y: float
    p0: dict[str, float] = field(default_factory=dict)  # bssid -> p0
    bssids: list[str] = field(default_factory=list)


@dataclass
class WifiFitResult:
    applied: bool
    aps: list[ApSite]
    keyframe_xy: list[tuple[float, float]]
    median_residual_db: float
    detail: str = ""


_PROC_RE = re.compile(
    r"^\s*(\w+):\s*\S+\s+\S+\s+(-?\d+(?:\.\d+)?)\.",
    re.M,
)


def parse_proc_wireless(text: str, iface: str = "wlan0") -> float | None:
    for line in text.splitlines():
        if not line.strip().startswith(iface):
            continue
        parts = line.split()
        # Inter-| sta-| Quality | ...  → wlan0: 0000  42.  -68.  -256
        if len(parts) >= 4:
            try:
                return float(parts[3].rstrip("."))
            except ValueError:
                return None
    return None


def read_proc_wireless(iface: str = "wlan0", path: str = "/proc/net/wireless") -> float | None:
    try:
        return parse_proc_wireless(open(path).read(), iface)
    except OSError:
        return None


def parse_iw_scan_text(text: str) -> list[BssReading]:
    """Parse `iw scan` / `iw scan dump` text (tests and fallback)."""
    out: list[BssReading] = []
    bssid = ssid = ""
    freq = 0
    dbm = None
    for line in text.splitlines():
        if line.startswith("BSS "):
            if bssid and dbm is not None:
                out.append(BssReading(bssid.lower(), ssid, freq, dbm))
            token = line.split()[1]
            bssid = token.split("(")[0].strip().lower()
            ssid = ""
            freq = 0
            dbm = None
        elif "signal:" in line:
            try:
                dbm = float(line.split("signal:")[1].split()[0])
            except (IndexError, ValueError):
                pass
        elif line.strip().startswith("freq:"):
            try:
                freq = int(float(line.split("freq:")[1].split()[0]))
            except (IndexError, ValueError):
                pass
        elif "SSID:" in line:
            ssid = line.split("SSID:", 1)[1].strip()
    if bssid and dbm is not None:
        out.append(BssReading(bssid.lower(), ssid, freq, dbm))
    return out


def pair_dual_band(readings: list[BssReading]) -> list[list[BssReading]]:
    """Group 2.4/5 GHz same-SSID BSSIDs whose last octet differs by ≤2."""
    by_ssid: dict[str, list[BssReading]] = {}
    for r in readings:
        by_ssid.setdefault(r.ssid or r.bssid, []).append(r)
    groups: list[list[BssReading]] = []
    used: set[str] = set()
    for ssid, items in by_ssid.items():
        band24 = [r for r in items if r.freq_mhz < 3000]
        band5 = [r for r in items if r.freq_mhz >= 3000]
        for a in band24:
            if a.bssid in used:
                continue
            try:
                last_a = int(a.bssid.split(":")[-1], 16)
            except ValueError:
                continue
            partner = None
            for b in band5:
                if b.bssid in used:
                    continue
                try:
                    last_b = int(b.bssid.split(":")[-1], 16)
                except ValueError:
                    continue
                if abs(last_a - last_b) <= 2 and a.bssid[:-2] == b.bssid[:-2]:
                    partner = b
                    break
            if partner:
                groups.append([a, partner])
                used.add(a.bssid)
                used.add(partner.bssid)
        for r in items:
            if r.bssid not in used:
                groups.append([r])
                used.add(r.bssid)
    return groups


def radio_span_db(epochs: list[WifiEpoch], bssid: str) -> float:
    vals = [r.dbm for e in epochs for r in e.readings if r.bssid == bssid]
    if len(vals) < 2:
        return 0.0
    return max(vals) - min(vals)


def _nla_put(attr_type: int, data: bytes) -> bytes:
    payload = data
    pad = (4 - (len(payload) % 4)) % 4
    return struct.pack("<HH", 4 + len(payload), attr_type) + payload + (b"\x00" * pad)


def _parse_nlas(buf: bytes) -> dict[int, bytes]:
    out: dict[int, bytes] = {}
    i = 0
    while i + 4 <= len(buf):
        alen, atype = struct.unpack_from("<HH", buf, i)
        if alen < 4 or i + alen > len(buf):
            break
        out[atype] = buf[i + 4:i + alen]
        i += (alen + 3) & ~3
    return out


def _mac_str(raw: bytes) -> str:
    return ":".join(f"{b:02x}" for b in raw[:6])


def parse_station_signal(sta_info_attrs: dict[int, bytes]) -> float | None:
    """Prefer SIGNAL_AVG over SIGNAL (dBm as signed byte)."""
    for key in (NL80211_STA_INFO_SIGNAL_AVG, NL80211_STA_INFO_SIGNAL):
        if key in sta_info_attrs and sta_info_attrs[key]:
            return float(struct.unpack("b", sta_info_attrs[key][:1])[0])
    return None


def parse_bss_nla(bss_blob: bytes) -> BssReading | None:
    attrs = _parse_nlas(bss_blob)
    if NL80211_BSS_BSSID not in attrs or NL80211_BSS_SIGNAL_MBM not in attrs:
        return None
    bssid = _mac_str(attrs[NL80211_BSS_BSSID])
    mbm = struct.unpack("<i", attrs[NL80211_BSS_SIGNAL_MBM][:4])[0]
    dbm = mbm / 100.0
    freq = 0
    if NL80211_BSS_FREQUENCY in attrs:
        freq = struct.unpack("<I", attrs[NL80211_BSS_FREQUENCY][:4])[0]
    ssid = ""
    ie = attrs.get(NL80211_BSS_INFORMATION_ELEMENTS, b"")
    i = 0
    while i + 2 <= len(ie):
        eid, elen = ie[i], ie[i + 1]
        if i + 2 + elen > len(ie):
            break
        if eid == 0:
            ssid = ie[i + 2:i + 2 + elen].decode("utf-8", "replace")
            break
        i += 2 + elen
    return BssReading(bssid, ssid, freq, dbm)


class Nl80211:
    """Tiny netlink client for scan + station. Falls back to `iw` if family resolve fails."""

    def __init__(self, iface: str = "wlan0"):
        self.iface = iface
        self.ifindex = self._ifindex(iface)
        self._sock: socket.socket | None = None
        self._family: int | None = None
        self._seq = 0

    @staticmethod
    def _ifindex(iface: str) -> int:
        return int(open(f"/sys/class/net/{iface}/ifindex").read().strip())

    def _connect(self) -> None:
        if self._sock is not None:
            return
        s = socket.socket(socket.AF_NETLINK, socket.SOCK_RAW, NETLINK_GENERIC)
        s.bind((0, 0))
        s.settimeout(5.0)
        self._sock = s
        self._family = self._resolve_family(b"nl80211")

    def _resolve_family(self, name: bytes) -> int:
        assert self._sock
        self._seq += 1
        payload = _nla_put(CTRL_ATTR_FAMILY_NAME, name + b"\x00")
        # genetlink header: cmd, version, reserved
        ge = struct.pack("<BBH", CTRL_CMD_GETFAMILY, 1, 0) + payload
        hdr = struct.pack("<IHHII", 16 + len(ge), 0x10, NLM_F_REQUEST | NLM_F_ACK, self._seq, os.getpid())
        # CTRL family id is 0x10 historically but resolve via GENL_ID_CTRL=0x10
        # Actually NETLINK_GENERIC uses family id in nlmsg_type; GENL_ID_CTRL = 0x10
        hdr = struct.pack("<IHHII", 16 + len(ge), 0x10, NLM_F_REQUEST, self._seq, os.getpid())
        self._sock.send(hdr + ge)
        data = self._sock.recv(65535)
        if len(data) < 20:
            raise OSError("ctrl family short")
        attrs = _parse_nlas(data[20:])  # skip nl + genl hdr
        # genl is 4 bytes after nlmsg hdr 16
        attrs = _parse_nlas(data[16 + 4:])
        if CTRL_ATTR_FAMILY_ID not in attrs:
            raise OSError("nl80211 family id missing")
        return struct.unpack("<H", attrs[CTRL_ATTR_FAMILY_ID][:2])[0]

    def _send(self, cmd: int, flags: int, attrs: bytes) -> None:
        self._connect()
        assert self._sock and self._family is not None
        self._seq += 1
        ge = struct.pack("<BBH", cmd, 0, 0) + attrs
        hdr = struct.pack("<IHHII", 16 + len(ge), self._family, flags, self._seq, os.getpid())
        self._sock.send(hdr + ge)

    def _recv_until(self, want: set[int], timeout: float = 8.0) -> list[bytes]:
        assert self._sock
        self._sock.settimeout(timeout)
        msgs = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                data = self._sock.recv(65535)
            except socket.timeout:
                break
            off = 0
            while off + 16 <= len(data):
                ln, typ, fl, seq, pid = struct.unpack_from("<IHHII", data, off)
                if ln < 16:
                    break
                body = data[off + 16:off + ln]
                if typ == NLMSG_ERROR:
                    errno = struct.unpack_from("<i", body)[0]
                    if errno != 0:
                        raise OSError(-errno, "nl80211 error")
                elif typ == NLMSG_DONE:
                    return msgs
                elif typ == self._family:
                    if len(body) >= 4:
                        cmd = body[0]
                        if cmd in want:
                            msgs.append(body)
                        if cmd == NL80211_CMD_SCAN_ABORTED:
                            raise OSError("scan aborted")
                        if cmd == NL80211_CMD_NEW_SCAN_RESULTS:
                            return msgs
                off += (ln + 3) & ~3
        return msgs

    def trigger_scan(self) -> None:
        attrs = _nla_put(NL80211_ATTR_IFINDEX, struct.pack("<I", self.ifindex))
        # empty SSID to scan all
        attrs += _nla_put(NL80211_ATTR_SCAN_SSIDS, _nla_put(1, b""))
        self._send(NL80211_CMD_TRIGGER_SCAN, NLM_F_REQUEST | NLM_F_ACK, attrs)
        self._recv_until({NL80211_CMD_NEW_SCAN_RESULTS, NL80211_CMD_SCAN_ABORTED}, timeout=12.0)

    def get_scan(self) -> list[BssReading]:
        attrs = _nla_put(NL80211_ATTR_IFINDEX, struct.pack("<I", self.ifindex))
        self._send(NL80211_CMD_GET_SCAN, NLM_F_REQUEST | NLM_F_DUMP, attrs)
        assert self._sock
        self._sock.settimeout(3.0)
        out: list[BssReading] = []
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                data = self._sock.recv(65535)
            except socket.timeout:
                break
            off = 0
            while off + 16 <= len(data):
                ln, typ, fl, seq, pid = struct.unpack_from("<IHHII", data, off)
                if ln < 16:
                    break
                if typ == NLMSG_DONE:
                    return out
                if typ == self._family and ln > 20:
                    body = data[off + 16:off + ln]
                    nlas = _parse_nlas(body[4:])
                    if NL80211_ATTR_BSS in nlas:
                        r = parse_bss_nla(nlas[NL80211_ATTR_BSS])
                        if r:
                            out.append(r)
                off += (ln + 3) & ~3
        return out

    def associated_signal(self) -> tuple[str | None, float | None]:
        """Return (bssid, dbm) for the associated station if possible."""
        # Get BSSID from iw link as fallback path via station dump needs MAC
        bssid = None
        try:
            link = subprocess.check_output(["iw", "dev", self.iface, "link"], text=True, timeout=2)
            for line in link.splitlines():
                if "Connected to" in line:
                    bssid = line.split("Connected to")[1].split()[0].lower()
                    break
        except (OSError, subprocess.SubprocessError):
            pass
        if not bssid:
            return None, read_proc_wireless(self.iface)
        try:
            mac = bytes(int(x, 16) for x in bssid.split(":"))
            attrs = _nla_put(NL80211_ATTR_IFINDEX, struct.pack("<I", self.ifindex))
            attrs += _nla_put(NL80211_ATTR_MAC, mac)
            self._send(NL80211_CMD_GET_STATION, NLM_F_REQUEST, attrs)
            assert self._sock
            data = self._sock.recv(65535)
            if len(data) > 20 and self._family:
                nlas = _parse_nlas(data[16 + 4:])
                if NL80211_ATTR_STA_INFO in nlas:
                    nested = _parse_nlas(nlas[NL80211_ATTR_STA_INFO])
                    sig = parse_station_signal(nested)
                    if sig is not None:
                        return bssid, sig
        except OSError:
            pass
        return bssid, read_proc_wireless(self.iface)

    def close(self) -> None:
        if self._sock:
            self._sock.close()
            self._sock = None


def scan_once_iw(iface: str = "wlan0") -> list[BssReading]:
    """Fallback: subprocess iw scan."""
    try:
        text = subprocess.check_output(["iw", "dev", iface, "scan"], text=True, timeout=20)
        return parse_iw_scan_text(text)
    except (OSError, subprocess.SubprocessError):
        return []


class WifiScanner:
    """Background back-to-back neighbor sweeps + associated reading once per epoch."""

    def __init__(self, iface: str = "wlan0", pose_fn: Callable[[], tuple[float, float, float]] | None = None):
        self.iface = iface
        self.pose_fn = pose_fn or (lambda: (0.0, 0.0, 0.0))
        self.epochs: list[WifiEpoch] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._nl: Nl80211 | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="wifi-scan", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)
        if self._nl:
            self._nl.close()

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            x0, y0, h0 = self.pose_fn()
            readings: list[BssReading] = []
            try:
                if self._nl is None:
                    try:
                        self._nl = Nl80211(self.iface)
                    except OSError:
                        self._nl = None
                if self._nl is not None:
                    try:
                        self._nl.trigger_scan()
                        readings = self._nl.get_scan()
                    except OSError:
                        readings = scan_once_iw(self.iface)
                else:
                    readings = scan_once_iw(self.iface)
            except Exception:  # noqa: BLE001
                readings = []
            t1 = time.monotonic()
            x1, y1, h1 = self.pose_fn()
            # Mid-window pose
            xm, ym = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
            hm = h0  # heading mid approx
            # Associated once
            try:
                if self._nl is not None:
                    bssid, dbm = self._nl.associated_signal()
                else:
                    bssid, dbm = None, read_proc_wireless(self.iface)
                if bssid and dbm is not None and not any(r.bssid == bssid for r in readings):
                    readings.append(BssReading(bssid, "", 0, dbm))
                elif dbm is not None and bssid:
                    for i, r in enumerate(readings):
                        if r.bssid == bssid:
                            readings[i] = BssReading(r.bssid, r.ssid, r.freq_mhz, dbm)
            except Exception:  # noqa: BLE001
                pass
            with self._lock:
                self.epochs.append(WifiEpoch(0.5 * (t0 + t1), xm, ym, hm, readings))
            # no extra sleep — back to back; stop check

    def snapshot(self) -> list[WifiEpoch]:
        with self._lock:
            return list(self.epochs)


def _huber(r: float, delta: float = HUBER_DB) -> float:
    a = abs(r)
    if a <= delta:
        return 0.5 * r * r
    return delta * (a - 0.5 * delta)


def _model_dbm(p0: float, dist: float, n: float = PATH_LOSS_N) -> float:
    d = max(MIN_DIST_M, dist)
    return p0 - 10.0 * n * math.log10(d)


def fit_pose_graph(
    epochs: list[WifiEpoch],
    *,
    n: float = PATH_LOSS_N,
    min_span_db: float = MIN_SPAN_DB,
) -> WifiFitResult:
    """Pose-graph: keyframe poses + AP (x,y,p0). Reject if median residual > Huber delta."""
    if len(epochs) < 3:
        return WifiFitResult(False, [], [], 0.0, "need ≥3 epochs")

    # Collect BSSIDs that span enough
    all_bss: dict[str, BssReading] = {}
    for e in epochs:
        for r in e.readings:
            all_bss[r.bssid] = r
    varied = [b for b in all_bss if radio_span_db(epochs, b) >= min_span_db]
    if len(varied) < 3:
        return WifiFitResult(False, [], [], 0.0, f"need ≥3 varied radios, have {len(varied)}")

    # Dual-band groups among varied
    varied_readings = [all_bss[b] for b in varied]
    groups = pair_dual_band(varied_readings)
    # Only keep groups whose members are all varied
    sites_meta: list[list[str]] = []
    for g in groups:
        ids = [r.bssid for r in g if r.bssid in varied]
        if ids:
            sites_meta.append(ids)

    if len(sites_meta) < 2:
        return WifiFitResult(False, [], [], 0.0, "not enough AP sites")

    # Initialize AP at loudest epoch, model distance out
    ap_xy: list[list[float]] = []
    ap_p0: list[dict[str, float]] = []
    for bssids in sites_meta:
        best_e, best_dbm, best_b = epochs[0], -999.0, bssids[0]
        for e in epochs:
            for r in e.readings:
                if r.bssid in bssids and r.dbm > best_dbm:
                    best_e, best_dbm, best_b = e, r.dbm, r.bssid
        # assume p0 ~ -30 at 1m; invert for init distance
        p0_guess = -30.0
        d = 10 ** ((p0_guess - best_dbm) / (10 * n))
        d = max(0.5, min(15.0, d))
        h = math.radians(best_e.heading_deg)
        ax = best_e.x + d * math.cos(h)
        ay = best_e.y + d * math.sin(h)
        ap_xy.append([ax, ay])
        p0s = {}
        for b in bssids:
            # estimate p0 from that reading if present
            db = next((r.dbm for r in best_e.readings if r.bssid == b), best_dbm)
            p0s[b] = db + 10 * n * math.log10(max(MIN_DIST_M, d))
        ap_p0.append(p0s)

    # Keyframe poses — start at IMU, allow soft adjust
    kx = [e.x for e in epochs]
    ky = [e.y for e in epochs]

    def residual_list() -> list[float]:
        rs = []
        for ei, e in enumerate(epochs):
            for r in e.readings:
                for si, bssids in enumerate(sites_meta):
                    if r.bssid not in bssids:
                        continue
                    dist = math.hypot(kx[ei] - ap_xy[si][0], ky[ei] - ap_xy[si][1])
                    pred = _model_dbm(ap_p0[si][r.bssid], dist, n)
                    rs.append(r.dbm - pred)
        return rs

    # Coordinate descent
    for _ in range(40):
        # odometry soft pull
        for i in range(1, len(epochs)):
            dx0 = epochs[i].x - epochs[i - 1].x
            dy0 = epochs[i].y - epochs[i - 1].y
            dx = kx[i] - kx[i - 1]
            dy = ky[i] - ky[i - 1]
            kx[i] -= 0.3 * (dx - dx0)
            ky[i] -= 0.3 * (dy - dy0)
        # range gradients (finite difference)
        for si, bssids in enumerate(sites_meta):
            for axis in (0, 1):
                for step in (-0.15, 0.15):
                    ap_xy[si][axis] += step
                    c1 = sum(_huber(r) for r in residual_list())
                    ap_xy[si][axis] -= step
                    c0 = sum(_huber(r) for r in residual_list())
                    if c1 < c0:
                        ap_xy[si][axis] += step
                        break
            for b in bssids:
                for step in (-1.0, 1.0):
                    ap_p0[si][b] += step
                    c1 = sum(_huber(r) for r in residual_list())
                    ap_p0[si][b] -= step
                    c0 = sum(_huber(r) for r in residual_list())
                    if c1 < c0:
                        ap_p0[si][b] += step
                        break
        # loop closure: similar fingerprint → pull keyframes
        for i in range(len(epochs)):
            for j in range(i + 2, len(epochs)):
                if math.hypot(epochs[i].x - epochs[j].x, epochs[i].y - epochs[j].y) < 0.4:
                    continue
                shared = 0
                diff = 0.0
                nsh = 0
                mi = {r.bssid: r.dbm for r in epochs[i].readings}
                mj = {r.bssid: r.dbm for r in epochs[j].readings}
                for b in set(mi) & set(mj):
                    if b not in varied:
                        continue
                    shared += 1
                    diff += abs(mi[b] - mj[b])
                    nsh += 1
                if shared >= 2 and nsh and diff / nsh < 4.0:
                    # pull together a little
                    mx, my = 0.5 * (kx[i] + kx[j]), 0.5 * (ky[i] + ky[j])
                    kx[i] = 0.7 * kx[i] + 0.3 * mx
                    ky[i] = 0.7 * ky[i] + 0.3 * my
                    kx[j] = 0.7 * kx[j] + 0.3 * mx
                    ky[j] = 0.7 * ky[j] + 0.3 * my

    rs = residual_list()
    if not rs:
        return WifiFitResult(False, [], [], 0.0, "no residuals")
    rs_sorted = sorted(abs(r) for r in rs)
    med = rs_sorted[len(rs_sorted) // 2]
    aps = []
    for si, bssids in enumerate(sites_meta):
        ssid = all_bss[bssids[0]].ssid
        aps.append(ApSite(
            id=bssids[0],
            ssid=ssid,
            x=ap_xy[si][0],
            y=ap_xy[si][1],
            p0=dict(ap_p0[si]),
            bssids=list(bssids),
        ))
    applied = med <= HUBER_DB
    return WifiFitResult(
        applied,
        aps,
        list(zip(kx, ky)),
        med,
        "ok" if applied else f"rejected med={med:.1f}dB",
    )


class SimWifi:
    """Synthetic epochs for tests — no wlan0."""

    def __init__(
        self,
        aps: list[tuple[float, float, float]],  # x, y, p0
        pose_fn: Callable[[], tuple[float, float, float]],
        noise: float = 1.5,
        period_s: float = 4.0,
        seed: int = 0,
    ):
        import random
        self.aps = aps
        self.pose_fn = pose_fn
        self.noise = noise
        self.period_s = period_s
        self.rng = random.Random(seed)
        self.epochs: list[WifiEpoch] = []
        self._t = 0.0

    def tick(self, sim_t: float) -> None:
        if sim_t - self._t < self.period_s and self.epochs:
            return
        self._t = sim_t
        x, y, h = self.pose_fn()
        readings = []
        for i, (ax, ay, p0) in enumerate(self.aps):
            d = max(MIN_DIST_M, math.hypot(x - ax, y - ay))
            dbm = p0 - 10 * PATH_LOSS_N * math.log10(d) + self.rng.uniform(-self.noise, self.noise)
            bssid = f"aa:bb:cc:dd:00:{i:02x}"
            readings.append(BssReading(bssid, f"sim{i}", 2412 + i, dbm))
            # dual-band twin for first AP
            if i == 0:
                readings.append(BssReading(
                    f"aa:bb:cc:dd:00:{i + 1:02x}", f"sim{i}", 5200,
                    p0 - 2 - 10 * PATH_LOSS_N * math.log10(d) + self.rng.uniform(-self.noise, self.noise),
                ))
        self.epochs.append(WifiEpoch(sim_t, x, y, h, readings))
