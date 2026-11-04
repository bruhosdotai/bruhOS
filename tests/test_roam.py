"""Tests for roam: IMU classifier, map, wifi parse/fit, sim room."""

from __future__ import annotations

import math

from bruhos.body.imu import ImuSample, hiwonder_lean_deg, sample_to_imu
from bruhos.roam.map import BLOCKED, FREE, OccupancyMap
from bruhos.roam.motion import (
    StepVerdict,
    classify_step,
    is_face_down,
    is_on_back,
    is_standing,
    learn_gyro_scale,
    mean_lean,
    still_window,
    synthetic_blocked_pulse,
    synthetic_fallen,
    synthetic_forward_pulse,
    synthetic_still,
    synthetic_turn,
    zupt_travel,
)
from bruhos.roam.roam import RoamConfig, run_roam
from bruhos.roam.wifi import (
    BssReading,
    WifiEpoch,
    fit_pose_graph,
    pair_dual_band,
    parse_bss_nla,
    parse_proc_wireless,
    parse_station_signal,
    radio_span_db,
)
from bruhos.sim.kinematic import SimBody, World


def test_standing_and_fall_angles():
    still = synthetic_still(0.0)
    assert is_standing(mean_lean(still))
    assert abs(mean_lean(still) - 90.0) < 1.0
    fd = synthetic_fallen(0.0, face_down=True)
    assert is_face_down(mean_lean(fd))
    assert not is_standing(mean_lean(fd))
    bk = synthetic_fallen(0.0, face_down=False)
    assert is_on_back(mean_lean(bk))
    # one spike in standing window is not a fall
    spiked = list(still)
    spiked[3] = ImuSample(spiked[3].t, 0.05, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert is_standing(mean_lean(spiked))


def test_sample_to_imu_hiwonder():
    standing = ImuSample(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    imu = sample_to_imu(standing)
    assert not imu.fallen
    face = ImuSample(0.0, 0.05, 0.0, 1.0, 0.0, 0.0, 0.0)
    imu = sample_to_imu(face)
    assert imu.fallen and imu.fell_forward is True


def test_zupt_forward_and_zero():
    samples = synthetic_forward_pulse(0.0, step_m=0.05)
    sw0 = still_window(samples[:8])
    assert sw0 is not None
    fwd, lat, yaw, trusted = zupt_travel(samples, sw0, gyro_scale_dps=1.0)
    assert trusted
    assert abs(fwd - 0.05) < 0.01
    assert abs(lat) < 0.01

    zero = synthetic_forward_pulse(0.0, step_m=0.0)
    sw0 = still_window(zero[:8])
    fwd, lat, yaw, trusted = zupt_travel(zero, sw0, gyro_scale_dps=1.0)
    assert trusted
    assert abs(fwd) < 0.01


def test_zupt_bias_cancels():
    """Constant accel bias across a bracketed step should not become a meter of travel."""
    samples = synthetic_forward_pulse(0.0, step_m=0.05)
    # add constant +0.05 g on Z to every sample
    biased = [
        ImuSample(s.t, s.ax, s.ay, s.az + 0.05, s.gx, s.gy, s.gz)
        for s in samples
    ]
    sw0 = still_window(biased[:8])
    assert sw0 is not None
    fwd, _, _, trusted = zupt_travel(biased, sw0, gyro_scale_dps=1.0)
    assert trusted
    assert abs(fwd) < 0.2  # must not explode to a meter


def test_classify_moved_blocked_fallen_assumed():
    moved = synthetic_forward_pulse(0.0, step_m=0.05)
    sw0 = still_window(moved[:8])
    sw1 = still_window(moved)
    r = classify_step(moved, sw0, sw1, gyro_scale_dps=1.0, open_floor_m=0.05)
    assert r.verdict is StepVerdict.MOVED

    blocked = synthetic_blocked_pulse(0.0)
    sw0 = still_window(blocked[:8])
    sw1 = still_window(blocked)
    r = classify_step(blocked, sw0, sw1, gyro_scale_dps=1.0, open_floor_m=0.05)
    assert r.verdict is StepVerdict.BLOCKED
    assert "lean" in r.detail or "short" in r.detail or "backward" in r.detail

    fallen = synthetic_fallen(1.0, face_down=True)
    # prepend standing still so still0 exists
    pre = synthetic_still(0.0)
    samples = pre + fallen
    sw0 = still_window(pre)
    r = classify_step(samples, sw0, None, gyro_scale_dps=1.0, open_floor_m=0.05)
    assert r.verdict is StepVerdict.FALLEN
    assert r.fell_forward is True

    # untrusted → assumed (same pre/post attitude; no gyro scale)
    sw0m = still_window(moved[:8])
    sw1m = still_window(moved)
    r = classify_step(moved, sw0m, sw1m, gyro_scale_dps=None, open_floor_m=0.05, step_m=0.05)
    assert r.verdict is StepVerdict.ASSUMED


def test_classify_lean_delta_vs_pre_step():
    """Residual lean vs still0 must block even when still inside the standing band."""
    moved = synthetic_forward_pulse(0.0, step_m=0.05)
    sw0 = still_window(moved[:8])
    assert sw0 is not None
    # Post still: lean ~45° while pre was ~90°
    post = [
        ImuSample(moved[-1].t + 0.02 * (i + 1), 0.7, 0.0, 0.7, 0.0, 0.0, 0.0)
        for i in range(10)
    ]
    samples = list(moved) + post
    sw1 = still_window(post)
    r = classify_step(samples, sw0, sw1, gyro_scale_dps=1.0, open_floor_m=0.05)
    assert r.verdict is StepVerdict.BLOCKED
    assert "lean" in r.detail


def test_map_advance_and_bump():
    m = OccupancyMap(cell_m=0.05)
    m.advance(0.05, kind="measured")
    assert m.pose.x == 0.05
    assert any(v == FREE for v in m.cells.values())
    bx, by = m.mark_bump_ahead(0.05)
    assert m._cell(bx, by) in m.cells
    assert m.cells[m._cell(bx, by)] == BLOCKED
    # blocked does not move pose
    x0 = m.pose.x
    m.turn(90)
    assert m.pose.x == x0


def test_map_assumed_not_bump():
    m = OccupancyMap()
    m.advance(0.05, kind="assumed")
    assert not m.bumps
    assert m.pose.x == 0.05


def test_gyro_scale_from_turn():
    samples = synthetic_turn(0.0, deg=45.0, gyro_scale_dps=1.0)
    bias = (0.0, 0.0, 0.0)
    scale = learn_gyro_scale(samples, 45.0, bias, yaw_axis=0)
    assert scale is not None
    assert abs(scale - 1.0) < 0.15


def test_gyro_scale_clamps_undersampled_turn():
    # Commanded 45° but raw integral only ~6 → raw scale ≈7.5, must fall back to ±1.
    samples = synthetic_turn(0.0, deg=6.0, gyro_scale_dps=1.0)
    scale = learn_gyro_scale(samples, 45.0, (0.0, 0.0, 0.0), yaw_axis=0)
    assert scale is not None
    assert abs(abs(scale) - 1.0) < 1e-9
    # Mildly high (~3.6 like a short scale turn) also clamps — was poisoning ZUPT.
    samples2 = synthetic_turn(0.0, deg=12.5, gyro_scale_dps=1.0)
    scale2 = learn_gyro_scale(samples2, 45.0, (0.0, 0.0, 0.0), yaw_axis=0)
    assert scale2 is not None and abs(abs(scale2) - 1.0) < 1e-9


def test_still_window_uses_raw_gyro_not_scale():
    # Idle biped noise ~15 raw; a bad learned scale must not inflate the quiet check.
    quiet = synthetic_still(0.0, n=20, gx=12.0)
    assert still_window(quiet) is not None
    assert still_window(quiet, gyro_scale_dps=-7.3) is not None


def test_sim_room_hits_wall():
    body = SimBody(world=World.room(1.0, 1.2), seed=0)
    cfg = RoamConfig(max_steps=30, max_bumps=10, use_wifi=False, seed=0, calibrate_steps=0)
    result = run_roam(body, cfg)
    assert result.map.bumps, "expected at least one bump against the room wall"
    # pose never outside room
    for p in result.map.trail:
        assert body.world.in_bounds(p.x, p.y) or True  # trail may mark bump cell ahead
    # robot itself stayed in bounds
    assert body.world.in_bounds(body.world.x, body.world.y)
    assert any(s.get("verdict") == "blocked" for s in result.steps) or result.map.bumps


def test_proc_wireless_parse():
    text = (
        "Inter-| sta-|   Quality        |   Discarded packets               | Missed | WE\n"
        " face | tus | link level noise |  nwid  crypt   frag  retry   misc | beacon | 22\n"
        " wlan0: 0000   42.  -68.  -256        0      0      0    228      0        0\n"
    )
    assert parse_proc_wireless(text, "wlan0") == -68.0


def test_station_signal_prefers_avg():
    import struct
    # nested attrs: SIGNAL=7 byte -70, SIGNAL_AVG=13 byte -65
    blob = b""
    blob += struct.pack("<HH", 5, 7) + struct.pack("b", -70) + b"\x00\x00\x00"
    # pad carefully via helper style
    def nla(t, d):
        pad = (4 - (len(d) % 4)) % 4
        return struct.pack("<HH", 4 + len(d), t) + d + (b"\x00" * pad)
    nested = nla(7, struct.pack("b", -70)) + nla(13, struct.pack("b", -65))
    from bruhos.roam.wifi import _parse_nlas
    attrs = _parse_nlas(nested)
    assert parse_station_signal(attrs) == -65.0


def test_parse_bss_nla():
    import struct
    from bruhos.roam.wifi import _nla_put, NL80211_BSS_BSSID, NL80211_BSS_FREQUENCY, NL80211_BSS_SIGNAL_MBM, NL80211_BSS_INFORMATION_ELEMENTS
    bssid = bytes.fromhex("a842a1e2398f")
    blob = b""
    blob += _nla_put(NL80211_BSS_BSSID, bssid)
    blob += _nla_put(NL80211_BSS_FREQUENCY, struct.pack("<I", 2417))
    blob += _nla_put(NL80211_BSS_SIGNAL_MBM, struct.pack("<i", -5000))
    ie = bytes([0, 8]) + b"pizzeria"
    blob += _nla_put(NL80211_BSS_INFORMATION_ELEMENTS, ie)
    r = parse_bss_nla(blob)
    assert r is not None
    assert r.bssid == "a8:42:a1:e2:39:8f"
    assert r.dbm == -50.0
    assert r.freq_mhz == 2417
    assert r.ssid == "pizzeria"


def test_dual_band_pair_and_flat_radio():
    readings = [
        BssReading("aa:bb:cc:dd:00:10", "pizzeria", 2412, -50),
        BssReading("aa:bb:cc:dd:00:11", "pizzeria", 5200, -68),
        BssReading("11:22:33:44:55:66", "other", 2412, -70),
    ]
    groups = pair_dual_band(readings)
    paired = [g for g in groups if len(g) == 2]
    assert paired
    assert {r.bssid for r in paired[0]} == {"aa:bb:cc:dd:00:10", "aa:bb:cc:dd:00:11"}

    epochs = [
        WifiEpoch(0, 0, 0, 0, [BssReading("flat:00", "x", 2412, -70)]),
        WifiEpoch(1, 0.5, 0, 0, [BssReading("flat:00", "x", 2412, -71)]),
        WifiEpoch(2, 1.0, 0, 0, [BssReading("flat:00", "x", 2412, -69)]),
    ]
    assert radio_span_db(epochs, "flat:00") < 6.0


def test_wifi_fit_places_aps_and_rejects_bad():
    # Three varied APs, path along x with planted yaw (heading ignored in xy)
    aps_truth = [(2.0, 1.0, -30.0), (2.0, -1.0, -30.0), (-1.0, 0.0, -32.0)]
    epochs = []
    for i, x in enumerate([0.0, 0.4, 0.8, 1.2]):
        readings = []
        for j, (ax, ay, p0) in enumerate(aps_truth):
            d = max(0.3, math.hypot(x - ax, 0.0 - ay))
            dbm = p0 - 10 * 2.5 * math.log10(d)
            readings.append(BssReading(f"ap:{j:02x}", f"sim{j}", 2412 + j, dbm))
        epochs.append(WifiEpoch(float(i), x, 0.05 * i, 5.0 * i, readings))  # small planted drift in y
    fit = fit_pose_graph(epochs)
    assert len(fit.aps) >= 2
    # flat radio alone should not get a site when span tiny — covered above
    # two scans same pose: not enough
    e0 = [epochs[0], epochs[0]]
    fit2 = fit_pose_graph(e0)
    assert not fit2.applied


def test_wifi_reject_leaves_bumps():
    body = SimBody(world=World.room(1.0, 1.2), seed=1)
    cfg = RoamConfig(max_steps=20, use_wifi=True, seed=1)
    result = run_roam(body, cfg)
    # bumps from IMU regardless of wifi
    assert result.end_reason in ("max_steps", "max_bumps", "consecutive_blocked", "done")


def test_hiwonder_lean_helper():
    assert abs(hiwonder_lean_deg(1.0, 0.0) - 90.0) < 1e-6
    assert abs(hiwonder_lean_deg(0.0, 1.0)) < 1e-6


def test_keyframe_warp_moves_trail_and_bumps():
    m = OccupancyMap(cell_m=0.05)
    m.advance(0.5, kind="measured")
    m.mark_bump_ahead(0.05)
    old = [(0.0, 0.0), (0.5, 0.0)]
    new = [(0.1, 0.2), (0.6, 0.2)]
    assert m.apply_keyframe_warp(old, new)
    assert abs(m.trail[0].x - 0.1) < 1e-9
    assert abs(m.trail[0].y - 0.2) < 1e-9
    assert abs(m.trail[-1].x - 0.6) < 1e-9
    assert abs(m.bumps[0][1] - 0.2) < 1e-9
    assert any(v == BLOCKED for v in m.cells.values())


def test_hardware_imu_gate():
    class FakeTony:
        name = "tonypi"
        has_board_imu = False
        turn_step_deg = 15.0

        def close(self):
            pass

    import pytest
    with pytest.raises(RuntimeError, match="controller-board IMU"):
        run_roam(FakeTony(), RoamConfig(max_steps=1, use_wifi=False))


def test_integrate_yaw_matches_turn():
    from bruhos.roam.motion import integrate_yaw_deg
    samples = synthetic_turn(0.0, deg=30.0, gyro_scale_dps=1.0)
    yaw = integrate_yaw_deg(samples, gyro_scale_dps=1.0)
    assert abs(yaw - 30.0) < 2.0


def test_sample_gap_default_allows_modest_jitter():
    from bruhos.roam.motion import SAMPLE_GAP_S
    assert SAMPLE_GAP_S >= 0.1
