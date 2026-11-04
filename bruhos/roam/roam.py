"""Bump-and-turn roam loop: IMU decides gait; WiFi fits separately."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from bruhos.body.base import Body
from bruhos.body.imu import ImuSample, hiwonder_lean_deg
from bruhos.roam.map import OccupancyMap
from bruhos.roam.motion import (
    StepResult,
    StepVerdict,
    StillWindow,
    classify_step,
    integrate_yaw_deg,
    learn_gyro_scale,
    still_window,
)
from bruhos.roam.wifi import SimWifi, WifiFitResult, WifiScanner, fit_pose_graph


@dataclass
class RoamConfig:
    step_m: float = 0.05
    cell_m: float = 0.05
    progress_ratio: float = 0.4
    sample_gap_s: float = 0.12
    max_steps: int = 40
    max_bumps: int = 15
    turn_min_deg: float = 60.0
    turn_max_deg: float = 150.0
    fall_forward_deg: float = 15.0
    fall_back_deg: float = 160.0
    wifi_iface: str = "wlan0"
    wifi_n: float = 2.5
    calibrate_steps: int = 0
    use_wifi: bool = True
    seed: int = 0
    forward_sign: float = 1.0
    map_size_m: float = 4.0
    use_depth: bool = False
    depth_every: int = 3
    feet_block_m: float = 0.28
    ahead_block_m: float = 0.45
    depth_input_size: int = 252


@dataclass
class RoamResult:
    map: OccupancyMap
    steps: list[dict[str, Any]] = field(default_factory=list)
    wifi: WifiFitResult | None = None
    open_floor_m: float | None = None
    gyro_scale: float | None = None
    end_reason: str = "done"

    def to_dict(self) -> dict[str, Any]:
        d = self.map.to_dict()
        d.update({
            "steps": self.steps,
            "open_floor_m": self.open_floor_m,
            "gyro_scale": self.gyro_scale,
            "end_reason": self.end_reason,
            "wifi": None if self.wifi is None else {
                "applied": self.wifi.applied,
                "median_residual_db": self.wifi.median_residual_db,
                "detail": self.wifi.detail,
                "aps": [
                    {"id": a.id, "ssid": a.ssid, "x": a.x, "y": a.y, "bssids": a.bssids}
                    for a in self.wifi.aps
                ],
            },
        })
        return d


def _is_sim(body: Any) -> bool:
    return getattr(body, "name", "") == "sim"


def _drain_board(body: Any, duration_s: float = 0.25) -> list[ImuSample]:
    """Host-time drain of the board IMU. Skip on sim (act_watched already brackets stills)."""
    if _is_sim(body):
        return []
    samples: list[ImuSample] = []
    t_end = time.monotonic() + duration_s
    while time.monotonic() < t_end:
        if hasattr(body, "poll_imu"):
            s = body.poll_imu()
            if s is not None:
                samples.append(s)
        time.sleep(0.01)
    return samples


def _wait_still(
    body: Any,
    cfg: RoamConfig,
    *,
    timeout_s: float = 2.0,
    seed: list[ImuSample] | None = None,
) -> tuple[StillWindow | None, list[ImuSample]]:
    """Poll until a still window forms or timeout.

    Returns (window, newly polled samples). ``seed`` is only used for detection.
    """
    if _is_sim(body):
        return None, []
    buf = list(seed or [])
    new_samples: list[ImuSample] = []
    t_end = time.monotonic() + timeout_s
    sw = still_window(buf, forward_sign=cfg.forward_sign) if len(buf) >= 2 else None
    while sw is None and time.monotonic() < t_end:
        if hasattr(body, "poll_imu"):
            s = body.poll_imu()
            if s is not None:
                buf.append(s)
                new_samples.append(s)
        else:
            time.sleep(0.01)
            continue
        sw = still_window(buf, forward_sign=cfg.forward_sign)
        if sw is None:
            time.sleep(0.01)
    return sw, new_samples


def _act_watched(body: Body, primitive: str) -> list[ImuSample]:
    if hasattr(body, "act_watched"):
        return list(body.act_watched(primitive))
    body.act(primitive)
    return []


def _scale_turn(body: Body, cfg: RoamConfig, rng: random.Random) -> tuple[float | None, StillWindow | None, list[dict]]:
    """In-place turn to learn gyro scale before any forward step."""
    log: list[dict] = []
    if _is_sim(body):
        scale = getattr(body, "gyro_scale_dps", 1.0)
        log.append({"kind": "scale_turn", "commanded_deg": 45.0, "gyro_scale": scale, "sim": True})
        print(f"[roam] gyro scale (sim) → scale={scale}", flush=True)
        return scale, None, log

    pre = _drain_board(body, 0.3)
    sw = still_window(pre, forward_sign=cfg.forward_sign) if pre else None
    n_turns = max(1, int(round(45.0 / max(1.0, body.turn_step_deg))))
    direction = "turn_left"
    samples: list[ImuSample] = []
    if sw:
        samples.extend(sw.samples)
    for _ in range(n_turns):
        samples.extend(_act_watched(body, direction))
    post, post_samples = _wait_still(body, cfg, timeout_s=1.5)
    samples.extend(post_samples)
    if post is not None:
        sw = post
    bias = sw.gyro_bias if sw else (0.0, 0.0, 0.0)
    commanded = n_turns * body.turn_step_deg
    raw_scale = learn_gyro_scale(samples, commanded, bias, yaw_axis=0)
    # learn_gyro_scale already clamps; log the value used for ZUPT/yaw.
    scale = raw_scale
    log.append({"kind": "scale_turn", "commanded_deg": commanded, "gyro_scale": scale})
    print(f"[roam] gyro scale turn {commanded:.0f}° → scale={scale}", flush=True)
    return scale, sw, log


def _calibrate_open_floor(
    body: Body,
    cfg: RoamConfig,
    scale: float | None,
    still: StillWindow | None,
) -> tuple[float | None, list[dict]]:
    log: list[dict] = []
    progresses: list[float] = []
    for i in range(cfg.calibrate_steps):
        sw0 = still
        if sw0 is None:
            sw0, _ = _wait_still(body, cfg, timeout_s=1.5)
        samples = list(_act_watched(body, "forward"))
        sw1, post = _wait_still(body, cfg, timeout_s=1.5, seed=samples[-5:] if samples else None)
        samples = samples + post
        if sw0 is None:
            sw0 = still_window(samples[: max(2, len(samples) // 5)], forward_sign=cfg.forward_sign)
        if sw0 is None:
            log.append({"kind": "calibrate", "i": i, "error": "no still"})
            print(f"[roam] calibrate {i + 1}/{cfg.calibrate_steps} skipped (no still)", flush=True)
            continue
        # Loose open_floor; progress is raw |fwd| (not map-clamped).
        res = classify_step(
            samples, sw0, sw1,
            gyro_scale_dps=scale,
            open_floor_m=max(cfg.step_m, 0.05),
            progress_ratio=0.0,
            sample_gap_s=cfg.sample_gap_s,
            fall_forward_deg=cfg.fall_forward_deg,
            fall_back_deg=cfg.fall_back_deg,
            step_m=cfg.step_m,
        )
        if res.trusted and res.progress > 0:
            progresses.append(res.progress)
        log.append({"kind": "calibrate", "i": i, "progress": res.progress, "trusted": res.trusted})
        print(f"[roam] calibrate {i + 1}/{cfg.calibrate_steps} progress={res.progress:.4f}m", flush=True)
        still = sw1 or sw0
    if not progresses:
        return None, log
    # One gait cycle is centimetres; metre-scale ZUPT is noise — keep a sane open floor.
    lo = 0.5 * cfg.step_m
    hi = max(2.0 * cfg.step_m, 0.08)
    sane = [p for p in progresses if lo <= p <= hi]
    if not sane:
        print(
            f"[roam] calibrate ZUPT out of band "
            f"({min(progresses):.3f}–{max(progresses):.3f}m); using step_m={cfg.step_m}",
            flush=True,
        )
        return cfg.step_m, log
    sane.sort()
    median = sane[len(sane) // 2]
    print(f"[roam] open_floor={median:.4f}m (from {len(sane)}/{len(progresses)} sane)", flush=True)
    return median, log


def _random_turn(
    body: Body,
    cfg: RoamConfig,
    rng: random.Random,
    mmap: OccupancyMap,
    *,
    gyro_scale_dps: float | None,
    prefer_left: bool | None = None,
) -> float:
    target = rng.uniform(cfg.turn_min_deg, cfg.turn_max_deg)
    left = (prefer_left if prefer_left is not None else rng.random() < 0.5)
    prim = "turn_left" if left else "turn_right"
    step = max(1.0, body.turn_step_deg)
    n = max(1, int(round(target / step)))
    yaw = 0.0
    measured = 0.0
    for _ in range(n):
        samples = _act_watched(body, prim)
        commanded = step if left else -step
        if gyro_scale_dps is not None and samples:
            measured += integrate_yaw_deg(samples, gyro_scale_dps=gyro_scale_dps)
        else:
            measured += commanded
        yaw += commanded
    # Prefer gyro integral once scale is known; fall back to commanded.
    if gyro_scale_dps is not None and abs(measured) > 1.0:
        yaw = measured
    mmap.turn(yaw)
    return yaw


def _pose_lean(body: Body, cfg: RoamConfig) -> tuple[float | None, bool | None]:
    """Return (lean_deg, fell_forward). fell_forward True=face-down, False=on-back."""
    if _is_sim(body):
        if hasattr(body, "observe"):
            obs = body.observe()
            return obs.imu.pitch_deg, obs.imu.fell_forward
        return None, None
    samples = _drain_board(body, 0.4)
    if not samples:
        return None, None
    lean = sum(hiwonder_lean_deg(s.ax, s.az) for s in samples) / len(samples)
    if abs(lean) < cfg.fall_forward_deg:
        return lean, True
    if abs(lean) > cfg.fall_back_deg:
        return lean, False
    return lean, None


def _recover(body: Body, fell_forward: bool | None) -> bool:
    order = ["stand_up_front", "stand_up_back"] if fell_forward is not False else ["stand_up_back", "stand_up_front"]
    for prim in order:
        print(f"[roam] recover {prim}", flush=True)
        body.act(prim)
        # Get-up finishes standing late; give the board time, then sample lean.
        time.sleep(1.0)
        for _ in range(8):
            if hasattr(body, "poll_imu"):
                samples = _drain_board(body, 0.35)
                if samples:
                    lean = sum(hiwonder_lean_deg(s.ax, s.az) for s in samples) / len(samples)
                    # Standing band ≈ ±90°; face-down ~0, on-back ~180.
                    if 40.0 <= abs(lean) <= 140.0:
                        print(f"[roam] recover ok lean={lean:.1f}", flush=True)
                        if hasattr(body, "act"):
                            body.act("stand")
                        return True
            elif hasattr(body, "observe"):
                obs = body.observe()
                if not obs.imu.fallen:
                    return True
            time.sleep(0.25)
    return False


def _ensure_upright(body: Body, cfg: RoamConfig, result: RoamResult) -> bool:
    """If already fallen at start, stand up before any gait. False → abort."""
    lean, fell_fwd = _pose_lean(body, cfg)
    if lean is None or fell_fwd is None:
        return True
    print(f"[roam] start lean={lean:.1f} fallen={'face-down' if fell_fwd else 'on-back'}", flush=True)
    result.steps.append({"kind": "start_pose", "lean_deg": lean, "fell_forward": fell_fwd})
    if not _recover(body, fell_fwd):
        result.end_reason = "stand_up_failed"
        return False
    result.steps.append({"kind": "start_recover", "ok": True})
    return True


def run_roam(
    body: Body,
    cfg: RoamConfig | None = None,
    *,
    should_stop: Callable[[], bool] | None = None,
) -> RoamResult:
    cfg = cfg or RoamConfig()
    rng = random.Random(cfg.seed)
    mmap = OccupancyMap(size_m=cfg.map_size_m, cell_m=cfg.cell_m)
    result = RoamResult(map=mmap)
    stop = should_stop or (lambda: False)

    # Hardware must have the controller-board IMU stream (not merely act_watched).
    if not _is_sim(body) and not getattr(body, "has_board_imu", False):
        raise RuntimeError("roam needs controller-board IMU; none opened")

    if not _is_sim(body) and not _ensure_upright(body, cfg, result):
        return result

    scale, still, slog = _scale_turn(body, cfg, rng)
    result.steps.extend(slog)
    result.gyro_scale = scale
    if scale is None and getattr(body, "gyro_scale_dps", None) is not None:
        scale = body.gyro_scale_dps
        result.gyro_scale = scale

    open_floor = cfg.step_m
    calibrated = False
    if cfg.calibrate_steps > 0:
        med, clog = _calibrate_open_floor(body, cfg, scale, still)
        result.steps.extend(clog)
        if med is not None:
            open_floor = med
            calibrated = True
    elif getattr(body, "name", "") == "sim":
        open_floor = cfg.step_m
    result.open_floor_m = open_floor

    wifi_scanner: WifiScanner | None = None
    sim_wifi: SimWifi | None = None
    if cfg.use_wifi:
        if getattr(body, "name", "") == "sim":
            # APs outside a ~1x1.2 room centered at 0
            sim_wifi = SimWifi(
                aps=[(1.2, 0.8, -30), (-1.0, 0.9, -32), (1.1, -1.0, -31), (-1.2, -0.8, -33)],
                pose_fn=lambda: (mmap.pose.x, mmap.pose.y, mmap.pose.heading_deg),
                seed=cfg.seed,
            )
        else:
            wifi_scanner = WifiScanner(
                cfg.wifi_iface,
                pose_fn=lambda: (mmap.pose.x, mmap.pose.y, mmap.pose.heading_deg),
            )
            wifi_scanner.start()

    scanner = None
    if cfg.use_depth and not _is_sim(body):
        head = getattr(body, "head", None)
        cap = getattr(body, "cap", None)
        if head is None or cap is None:
            print("[roam] depth requested but head/camera missing — IMU only", flush=True)
        else:
            try:
                from bruhos.roam.depth import MetricDepth
                from bruhos.roam.look import VerticalScanner, format_look

                scanner = VerticalScanner(
                    head, cap,
                    MetricDepth(input_size=cfg.depth_input_size),
                    feet_block_m=cfg.feet_block_m,
                    ahead_block_m=cfg.ahead_block_m,
                )
                print(
                    f"[roam] vertical depth on "
                    f"(every {cfg.depth_every} steps; feet<{cfg.feet_block_m}m "
                    f"ahead<{cfg.ahead_block_m}m)",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                print(f"[roam] depth disabled: {e}", flush=True)
                scanner = None

    bumps = 0
    consecutive_blocked = 0
    consecutive_no_still = 0
    n_forward = 0
    last_look = None

    try:
        while n_forward < cfg.max_steps and bumps < cfg.max_bumps and not stop():
            if hasattr(body, "_halted") and body._halted.is_set():  # noqa: SLF001
                result.end_reason = "halted"
                break

            # Periodic feet+ahead depth before the gait (slow on Pi — every N).
            prefer_left: bool | None = None
            vision_block = False
            if scanner is not None and (n_forward % max(1, cfg.depth_every) == 0):
                from bruhos.roam.look import format_look

                last_look = scanner.look()
                print(f"[roam] look {format_look(last_look)}", flush=True)
                result.steps.append({
                    "kind": "look",
                    "feet_m": None if last_look.feet is None else last_look.feet.center_m,
                    "ahead_m": None if last_look.ahead is None else last_look.ahead.center_m,
                    "blocked": last_look.blocked,
                    "feet_blocked": last_look.feet_blocked,
                    "ahead_blocked": last_look.ahead_blocked,
                })
                if last_look.blocked:
                    vision_block = True
                    prefer_left = last_look.prefer_turn_left()

            if vision_block:
                n_forward += 1
                mmap.mark_bump_ahead(cfg.step_m)
                bumps += 1
                consecutive_blocked += 1
                detail = "vision feet" if last_look and last_look.feet_blocked else "vision ahead"
                result.steps.append({
                    "i": n_forward, "verdict": "blocked", "forward_m": 0.0,
                    "yaw_deg": 0.0, "trusted": True, "detail": detail, "progress": 0.0,
                })
                print(f"[roam] step {n_forward} blocked {detail}", flush=True)
                if consecutive_blocked >= 5:
                    result.end_reason = "consecutive_blocked"
                    break
                _random_turn(body, cfg, rng, mmap, gyro_scale_dps=scale, prefer_left=prefer_left)
                still = None
                continue

            sw0, _ = _wait_still(body, cfg, timeout_s=1.5)
            if sw0 is None:
                sw0 = still
            samples = list(_act_watched(body, "forward"))
            sw1, post = _wait_still(
                body, cfg, timeout_s=1.5,
                seed=samples[-8:] if samples else None,
            )
            samples = samples + post
            if not samples and sw0 is None:
                if not _is_sim(body):
                    result.end_reason = "no_imu"
                    break
            if sw0 is None and samples:
                sw0 = still_window(
                    samples[: max(2, len(samples) // 5)],
                    forward_sign=cfg.forward_sign,
                )
            if sw0 is None:
                consecutive_no_still += 1
                print(
                    f"[roam] no still before/after step "
                    f"({consecutive_no_still}/3); assuming nominal step",
                    flush=True,
                )
                if consecutive_no_still >= 3:
                    result.end_reason = "no_still"
                    break
                # Keep walking so a slow settle does not kill the run after calibrate.
                mmap.advance(cfg.step_m, kind="assumed")
                n_forward += 1
                result.steps.append({
                    "i": n_forward, "verdict": "assumed", "forward_m": cfg.step_m,
                    "yaw_deg": 0.0, "trusted": False, "detail": "no_still", "progress": 0.0,
                })
                continue

            consecutive_no_still = 0
            of = open_floor
            # No usable open-floor median → never treat travel as a hard bump.
            if not _is_sim(body) and cfg.calibrate_steps > 0 and not calibrated:
                of = None

            res: StepResult = classify_step(
                samples or sw0.samples, sw0, sw1,
                gyro_scale_dps=scale,
                open_floor_m=of,
                progress_ratio=cfg.progress_ratio,
                sample_gap_s=cfg.sample_gap_s,
                fall_forward_deg=cfg.fall_forward_deg,
                fall_back_deg=cfg.fall_back_deg,
                step_m=cfg.step_m,
            )

            # Depth says clear → IMU short/backward is noise; keep walking.
            if (
                res.verdict is StepVerdict.BLOCKED
                and last_look is not None
                and last_look.clear
                and res.detail.startswith(("short", "backward", "yaw stall"))
            ):
                res = StepResult(
                    StepVerdict.ASSUMED, forward_m=cfg.step_m, yaw_deg=0.0,
                    trusted=False, detail=f"imu {res.detail}; depth clear",
                    progress=res.progress,
                )

            # Confirm IMU bump with a fresh look when we weren't due for one.
            if (
                res.verdict is StepVerdict.BLOCKED
                and scanner is not None
                and n_forward % max(1, cfg.depth_every) != 0
            ):
                from bruhos.roam.look import format_look

                last_look = scanner.look()
                print(f"[roam] look (confirm) {format_look(last_look)}", flush=True)
                if last_look.clear:
                    res = StepResult(
                        StepVerdict.ASSUMED, forward_m=cfg.step_m, yaw_deg=0.0,
                        trusted=False, detail=f"imu {res.detail}; depth clear",
                        progress=res.progress,
                    )
                else:
                    prefer_left = last_look.prefer_turn_left()

            n_forward += 1
            entry = {
                "i": n_forward,
                "verdict": res.verdict.value,
                "forward_m": res.forward_m,
                "yaw_deg": res.yaw_deg,
                "trusted": res.trusted,
                "detail": res.detail,
                "progress": res.progress,
            }
            result.steps.append(entry)
            print(
                f"[roam] step {n_forward} {res.verdict.value} "
                f"fwd={res.forward_m:.3f} yaw={res.yaw_deg:.1f} {res.detail}",
                flush=True,
            )

            if sim_wifi is not None:
                sim_wifi.tick(getattr(body, "t", float(n_forward)))

            if res.verdict is StepVerdict.FALLEN:
                mmap.mark_bump_ahead(cfg.step_m)
                bumps += 1
                consecutive_blocked += 1
                if not _recover(body, res.fell_forward):
                    result.end_reason = "stand_up_failed"
                    break
                _random_turn(body, cfg, rng, mmap, gyro_scale_dps=scale)
                still = None
                continue

            if res.verdict is StepVerdict.BLOCKED:
                mmap.mark_bump_ahead(cfg.step_m)
                bumps += 1
                consecutive_blocked += 1
                if consecutive_blocked >= 5:
                    result.end_reason = "consecutive_blocked"
                    break
                _random_turn(body, cfg, rng, mmap, gyro_scale_dps=scale, prefer_left=prefer_left)
                still = sw1
                continue

            consecutive_blocked = 0
            kind = "assumed" if res.verdict is StepVerdict.ASSUMED else "measured"
            fwd = res.forward_m if res.verdict is StepVerdict.MOVED else cfg.step_m
            if res.verdict is StepVerdict.ASSUMED:
                fwd = cfg.step_m
            mmap.advance(fwd, res.lateral_m if abs(res.lateral_m) > 0.01 else 0.0, res.yaw_deg, kind=kind)
            still = sw1 or sw0

        else:
            if n_forward >= cfg.max_steps:
                result.end_reason = "max_steps"
            elif bumps >= cfg.max_bumps:
                result.end_reason = "max_bumps"
    finally:
        if wifi_scanner is not None:
            wifi_scanner.stop()

    # WiFi fit
    epochs = []
    if wifi_scanner is not None:
        epochs = wifi_scanner.snapshot()
    elif sim_wifi is not None:
        epochs = sim_wifi.epochs
    if cfg.use_wifi and epochs:
        fit = fit_pose_graph(epochs, n=cfg.wifi_n)
        result.wifi = fit
        if fit.applied and fit.keyframe_xy:
            old_xy = [(e.x, e.y) for e in epochs]
            new_xy = list(fit.keyframe_xy)
            warped = mmap.apply_keyframe_warp(old_xy, new_xy)
            mmap.aps = [
                {"id": a.id, "ssid": a.ssid, "x": a.x, "y": a.y, "bssids": a.bssids}
                for a in fit.aps
            ]
            print(
                f"[roam] wifi fit applied med={fit.median_residual_db:.1f}dB "
                f"aps={len(fit.aps)} warp={'yes' if warped else 'no'}",
                flush=True,
            )
        else:
            print(f"[roam] wifi fit {fit.detail}", flush=True)

    return result


def format_result(result: RoamResult) -> str:
    lines = [
        f"end={result.end_reason} open_floor={result.open_floor_m} gyro_scale={result.gyro_scale}",
        result.map.ascii(),
        json.dumps(result.to_dict(), indent=2),
    ]
    return "\n".join(lines)
