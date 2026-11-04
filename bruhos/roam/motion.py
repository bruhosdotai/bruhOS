"""Step classifier: still-window upright/fall + strapdown ZUPT torso travel.

Hiwonder axes: X up, Z forward when standing, yaw rate on gx.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from bruhos.body.imu import ImuSample, hiwonder_lean_deg

G = 9.81
STILL_GYRO_MAX = 35.0          # raw gyro units (this board ≈ deg/s); biped settles noisy
STILL_WINDOW_S = 0.25
SAMPLE_GAP_S = 0.12            # board queue drops packets; 40ms was rejecting real gait
LEAN_RETURN_DEG = 20.0         # |post lean − pre lean|; shove / stuck against something
YAW_STALL_DEG = 40.0           # biped gait swings; only stall when travel is also short
GYRO_SCALE_MIN = 0.5           # |learned scale| outside this band → treat stream as deg/s
GYRO_SCALE_MAX = 2.0
# ZUPT on a biped over one gait cycle is noisy; beyond this, don't trust metres.
MAX_TRUSTED_TRAVEL_STEPS = 4.0


class StepVerdict(str, Enum):
    FALLEN = "fallen"
    BLOCKED = "blocked"
    MOVED = "moved"
    ASSUMED = "assumed"


@dataclass(frozen=True)
class AxisFrame:
    """Dominant up axis index and forward axis index in sensor frame (0=x,1=y,2=z)."""

    up: int
    forward: int
    forward_sign: float = 1.0


@dataclass
class StillWindow:
    samples: list[ImuSample]
    accel_scale: float          # sensor units per g
    gyro_bias: tuple[float, float, float]
    axes: AxisFrame
    lean_deg: float


@dataclass
class StepResult:
    verdict: StepVerdict
    forward_m: float = 0.0
    lateral_m: float = 0.0
    yaw_deg: float = 0.0
    trusted: bool = False
    fell_forward: bool | None = None
    detail: str = ""
    progress: float = 0.0       # |forward_m| when trusted


def _vec(s: ImuSample) -> tuple[float, float, float]:
    return (s.ax, s.ay, s.az)


def _gyro(s: ImuSample) -> tuple[float, float, float]:
    return (s.gx, s.gy, s.gz)


def detect_axes(samples: Sequence[ImuSample], forward_sign: float = 1.0) -> AxisFrame:
    """Dominant accel axis is up; Hiwonder face-down uses az with ax up → forward is Z."""
    if not samples:
        return AxisFrame(0, 2, forward_sign)
    mx = sum(abs(s.ax) for s in samples) / len(samples)
    my = sum(abs(s.ay) for s in samples) / len(samples)
    mz = sum(abs(s.az) for s in samples) / len(samples)
    up = max(range(3), key=lambda i: (mx, my, mz)[i])
    # Remaining axes: prefer Z as forward when X is up (Hiwonder).
    if up == 0:
        forward = 2
    elif up == 2:
        forward = 0
    else:
        forward = 2 if mz >= mx else 0
    return AxisFrame(up, forward, float(forward_sign))


def accel_magnitude(samples: Sequence[ImuSample]) -> float:
    if not samples:
        return 1.0
    mags = [math.hypot(s.ax, s.ay, s.az) for s in samples]
    return max(1e-6, sum(mags) / len(mags))


def gyro_bias(samples: Sequence[ImuSample]) -> tuple[float, float, float]:
    if not samples:
        return (0.0, 0.0, 0.0)
    n = len(samples)
    return (
        sum(s.gx for s in samples) / n,
        sum(s.gy for s in samples) / n,
        sum(s.gz for s in samples) / n,
    )


def mean_lean(samples: Sequence[ImuSample]) -> float:
    if not samples:
        return 90.0
    return sum(hiwonder_lean_deg(s.ax, s.az) for s in samples) / len(samples)


def is_standing(lean_deg: float, fall_forward_deg: float = 15.0, fall_back_deg: float = 160.0) -> bool:
    return fall_forward_deg <= abs(lean_deg) <= fall_back_deg


def is_face_down(lean_deg: float, fall_forward_deg: float = 15.0) -> bool:
    return abs(lean_deg) < fall_forward_deg


def is_on_back(lean_deg: float, fall_back_deg: float = 160.0) -> bool:
    return abs(lean_deg) > fall_back_deg


def still_window(
    samples: Sequence[ImuSample],
    *,
    window_s: float = STILL_WINDOW_S,
    forward_sign: float = 1.0,
    gyro_quiet: float | None = None,
    gyro_scale_dps: float | None = None,  # kept for call-site compat; quiet check is raw
) -> StillWindow | None:
    """Trailing quiet window, or None if not enough quiet samples.

    Gyro quiet is judged in **raw** board units (≈ deg/s here). Do not multiply by
    learned scale — a bad scale turn made |scale|≈7 and rejected every still sample.
    """
    del gyro_scale_dps  # unused; signature kept stable for callers
    if len(samples) < 2:
        return None
    t_end = samples[-1].t
    win = [s for s in samples if s.t >= t_end - window_s]
    if len(win) < 2:
        return None
    bias = gyro_bias(win)
    thr = gyro_quiet if gyro_quiet is not None else STILL_GYRO_MAX
    for s in win:
        rates = (
            abs(s.gx - bias[0]),
            abs(s.gy - bias[1]),
            abs(s.gz - bias[2]),
        )
        if max(rates) > thr:
            return None
    scale = accel_magnitude(win)
    axes = detect_axes(win, forward_sign)
    return StillWindow(list(win), scale, bias, axes, mean_lean(win))


def integrate_yaw_deg(
    samples: Sequence[ImuSample],
    *,
    gyro_scale_dps: float,
    bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
    yaw_axis: int = 0,
) -> float:
    """Integrate yaw about the vertical body axis (gx when X is up)."""
    yaw = 0.0
    for i in range(1, len(samples)):
        dt = samples[i].t - samples[i - 1].t
        if dt <= 0 or dt > 0.5:
            continue
        g = _gyro(samples[i])[yaw_axis] - bias[yaw_axis]
        yaw += g * gyro_scale_dps * dt
    return yaw


def learn_gyro_scale(
    samples: Sequence[ImuSample],
    commanded_deg: float,
    bias: tuple[float, float, float],
    yaw_axis: int = 0,
) -> float | None:
    """deg per raw gyro unit from an in-place turn. None if integral ~0.

    If the turn was undersampled (|scale| outside a sane band), keep the sign and
    fall back to magnitude 1 (board stream is already ≈ deg/s).
    """
    if abs(commanded_deg) < 1.0 or len(samples) < 2:
        return None
    integ = 0.0
    for i in range(1, len(samples)):
        dt = samples[i].t - samples[i - 1].t
        if dt <= 0 or dt > 0.5:
            continue
        g = _gyro(samples[i])[yaw_axis] - bias[yaw_axis]
        integ += g * dt
    if abs(integ) < 1e-3:
        return None
    scale = commanded_deg / integ
    if not (GYRO_SCALE_MIN <= abs(scale) <= GYRO_SCALE_MAX):
        return math.copysign(1.0, scale)
    return scale


def _has_gap(samples: Sequence[ImuSample], max_gap: float) -> bool:
    for i in range(1, len(samples)):
        if samples[i].t - samples[i - 1].t > max_gap:
            return True
    return False


def _rot_update(q: list[float], gx: float, gy: float, gz: float, dt: float) -> None:
    """Integrate body rates (rad/s) into quaternion [w,x,y,z] in place."""
    half = 0.5 * dt
    qw, qx, qy, qz = q
    dw = (-qx * gx - qy * gy - qz * gz) * half
    dx = (qw * gx + qy * gz - qz * gy) * half
    dy = (qw * gy - qx * gz + qz * gx) * half
    dz = (qw * gz + qx * gy - qy * gx) * half
    q[0] = qw + dw
    q[1] = qx + dx
    q[2] = qy + dy
    q[3] = qz + dz
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]) or 1.0
    q[0] /= n
    q[1] /= n
    q[2] /= n
    q[3] /= n


def _rotate(q: Sequence[float], v: Sequence[float]) -> tuple[float, float, float]:
    """Rotate vector by quaternion (body → world)."""
    qw, qx, qy, qz = q
    vx, vy, vz = v
    # t = 2 * cross(q_xyz, v)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    # v' = v + qw*t + cross(q_xyz, t)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def _quat_from_up(up_body: tuple[float, float, float]) -> list[float]:
    """Quaternion that maps body up toward world +Z (simplified standing attitude)."""
    ux, uy, uz = up_body
    n = math.hypot(ux, uy, uz) or 1.0
    ux, uy, uz = ux / n, uy / n, uz / n
    # Rotate body up onto (0,0,1)
    # axis = up × world_up, world_up = (0,0,1)
    ax, ay, az = uy, -ux, 0.0
    c = uz  # dot
    s = math.hypot(ax, ay, az)
    if s < 1e-8:
        if c > 0:
            return [1.0, 0.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0, 0.0]  # 180° about X
    ax, ay, az = ax / s, ay / s, az / s
    half = math.acos(max(-1.0, min(1.0, c))) * 0.5
    return [math.cos(half), ax * math.sin(half), ay * math.sin(half), az * math.sin(half)]


def zupt_travel(
    samples: Sequence[ImuSample],
    still0: StillWindow,
    *,
    gyro_scale_dps: float | None,
    sample_gap_s: float = SAMPLE_GAP_S,
) -> tuple[float, float, float, bool]:
    """Return (forward_m, lateral_m, yaw_deg, trusted).

    World: +Z up, +X along starting forward. Forward/lateral are in the starting horizontal frame.
    """
    if len(samples) < 3 or gyro_scale_dps is None:
        return 0.0, 0.0, 0.0, False
    if _has_gap(samples, sample_gap_s):
        return 0.0, 0.0, 0.0, False

    scale = still0.accel_scale
    bias = still0.gyro_bias
    up_i = still0.axes.up
    fwd_i = still0.axes.forward
    fsign = still0.axes.forward_sign

    # Body unit vectors from mean still accel
    mean_a = (
        sum(s.ax for s in still0.samples) / len(still0.samples),
        sum(s.ay for s in still0.samples) / len(still0.samples),
        sum(s.az for s in still0.samples) / len(still0.samples),
    )
    up_body = (mean_a[0] / scale, mean_a[1] / scale, mean_a[2] / scale)
    q = _quat_from_up(up_body)

    # Forward in body: axis unit, then into world at t0
    fb = [0.0, 0.0, 0.0]
    fb[fwd_i] = fsign
    f0 = _rotate(q, fb)
    # Project forward onto horizontal (drop world Z)
    f_h = (f0[0], f0[1], 0.0)
    fn = math.hypot(f_h[0], f_h[1]) or 1.0
    fx, fy = f_h[0] / fn, f_h[1] / fn
    # Left is Z×forward in horizontal: (-fy, fx)
    lx, ly = -fy, fx

    rad_scale = math.radians(gyro_scale_dps)
    vx = vy = vz = 0.0
    times: list[float] = []
    vels: list[tuple[float, float, float]] = []
    yaw = 0.0

    for i in range(1, len(samples)):
        dt = samples[i].t - samples[i - 1].t
        if dt <= 0:
            continue
        s = samples[i]
        gx = (s.gx - bias[0]) * rad_scale
        gy = (s.gy - bias[1]) * rad_scale
        gz = (s.gz - bias[2]) * rad_scale
        _rot_update(q, gx, gy, gz, dt)
        # Yaw about world up ≈ body up axis rate (gx when X is up)
        yaw += (s.gx - bias[0]) * gyro_scale_dps * dt

        ab = (s.ax / scale, s.ay / scale, s.az / scale)
        aw = _rotate(q, ab)
        # Specific force in g → m/s², remove gravity along world Z
        axw = aw[0] * G
        ayw = aw[1] * G
        azw = (aw[2] - 1.0) * G
        vx += axw * dt
        vy += ayw * dt
        vz += azw * dt
        times.append(samples[i].t)
        vels.append((vx, vy, vz))

    if not times:
        return 0.0, 0.0, yaw, False

    T = times[-1] - times[0]
    if T <= 0:
        return 0.0, 0.0, yaw, False
    vTx, vTy, vTz = vels[-1]
    t0 = times[0]
    px = py = pz = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        frac = (times[i] - t0) / T
        # ZUPT: remove linear ramp to end velocity
        cx = vels[i][0] - frac * vTx
        cy = vels[i][1] - frac * vTy
        cz = vels[i][2] - frac * vTz
        # Midpoint with previous corrected
        pfrac = (times[i - 1] - t0) / T
        px += 0.5 * ((vels[i - 1][0] - pfrac * vTx) + cx) * dt
        py += 0.5 * ((vels[i - 1][1] - pfrac * vTy) + cy) * dt
        pz += 0.5 * ((vels[i - 1][2] - pfrac * vTz) + cz) * dt

    forward = px * fx + py * fy
    lateral = px * lx + py * ly
    return forward, lateral, yaw, True


def classify_step(
    samples: Sequence[ImuSample],
    still0: StillWindow,
    still1: StillWindow | None,
    *,
    gyro_scale_dps: float | None,
    open_floor_m: float | None,
    progress_ratio: float = 0.4,
    sample_gap_s: float = SAMPLE_GAP_S,
    fall_forward_deg: float = 15.0,
    fall_back_deg: float = 160.0,
    step_m: float = 0.05,
) -> StepResult:
    """Judge one finished gait cycle from bracketing still windows + mid samples."""
    post = still1 or still_window(samples, forward_sign=still0.axes.forward_sign)
    lean = post.lean_deg if post else mean_lean(samples[-max(1, len(samples) // 5):] or samples)

    if is_face_down(lean, fall_forward_deg):
        return StepResult(StepVerdict.FALLEN, fell_forward=True, detail="face-down")
    if is_on_back(lean, fall_back_deg):
        return StepResult(StepVerdict.FALLEN, fell_forward=False, detail="on-back")

    fwd, lat, yaw, trusted = zupt_travel(
        samples, still0, gyro_scale_dps=gyro_scale_dps, sample_gap_s=sample_gap_s,
    )

    # Settled attitude vs pre-step still (not vs absolute ±90 — standing band is wide).
    lean_delta = abs(lean - still0.lean_deg)
    if lean_delta > LEAN_RETURN_DEG:
        return StepResult(
            StepVerdict.BLOCKED, forward_m=fwd, lateral_m=lat, yaw_deg=yaw,
            trusted=trusted, detail=f"lean Δ={lean_delta:.1f} ({still0.lean_deg:.1f}→{lean:.1f})",
            progress=abs(fwd),
        )

    raw_progress = abs(fwd)
    if abs(lat) < 0.01:
        lat = 0.0

    # Runaway ZUPT (metre-scale "one step") → assume nominal; do not bump.
    max_travel = MAX_TRUSTED_TRAVEL_STEPS * max(step_m, 0.03)
    if trusted and raw_progress > max_travel:
        return StepResult(
            StepVerdict.ASSUMED, forward_m=step_m, yaw_deg=0.0,
            trusted=False, detail=f"zupt runaway {raw_progress:.2f}m", progress=raw_progress,
        )

    if not trusted or open_floor_m is None or open_floor_m <= 0:
        return StepResult(
            StepVerdict.ASSUMED, forward_m=step_m, yaw_deg=yaw if trusted else 0.0,
            trusted=False, detail="untrusted travel", progress=raw_progress,
        )

    short_thresh = progress_ratio * open_floor_m
    # Yaw alone is normal gait sway; only stall when travel is also short.
    if abs(yaw) > YAW_STALL_DEG and raw_progress < short_thresh:
        return StepResult(
            StepVerdict.BLOCKED, forward_m=fwd, lateral_m=lat, yaw_deg=yaw,
            trusted=True, detail=f"yaw stall {yaw:.1f}", progress=raw_progress,
        )

    # Tiny negative ZUPT is sign noise, not a shove-back — need a real reverse.
    if fwd < -max(short_thresh, step_m):
        return StepResult(
            StepVerdict.BLOCKED, forward_m=fwd, lateral_m=lat, yaw_deg=yaw,
            trusted=True, detail="backward", progress=raw_progress,
        )
    if raw_progress < short_thresh:
        return StepResult(
            StepVerdict.BLOCKED, forward_m=fwd, lateral_m=lat, yaw_deg=yaw,
            trusted=True,
            detail=f"short {raw_progress:.3f}<{short_thresh:.3f}",
            progress=raw_progress,
        )
    # Tiny / negative ZUPT with enough |progress| via abs check above: still require +fwd.
    if fwd <= 0:
        return StepResult(
            StepVerdict.ASSUMED, forward_m=step_m, yaw_deg=0.0,
            trusted=False, detail=f"nonpositive fwd {fwd:.3f}", progress=raw_progress,
        )
    # Map advance uses physical step_m, not the (noisy) open_floor estimate.
    lo, hi = 0.5 * step_m, 2.0 * step_m
    adv = max(lo, min(hi, fwd))
    return StepResult(
        StepVerdict.MOVED, forward_m=adv, lateral_m=lat, yaw_deg=yaw,
        trusted=True, detail="ok", progress=raw_progress,
    )


def synthetic_still(
    t0: float,
    *,
    n: int = 10,
    dt: float = 0.02,
    ax: float = 1.0,
    ay: float = 0.0,
    az: float = 0.0,
    gx: float = 0.0,
) -> list[ImuSample]:
    """Standing still on +X (Hiwonder). ax=1g in sensor units."""
    return [
        ImuSample(t0 + i * dt, ax, ay, az, gx, 0.0, 0.0)
        for i in range(n)
    ]


def synthetic_forward_pulse(
    t0: float,
    *,
    step_m: float = 0.05,
    duration_s: float = 0.4,
    dt: float = 0.02,
    accel_scale: float = 1.0,
) -> list[ImuSample]:
    """Still / +a then −a along Z / still. ZUPT recovers ~step_m.

    With ZUPT ramp removal, a triangular velocity of peak V and duration T
    yields displacement ≈ V*T/4 for +a/−a of equal length. Choose peak accel
    so that works out to step_m.
    """
    samples = synthetic_still(t0, n=8, dt=dt, ax=accel_scale)
    t = samples[-1].t + dt
    half = duration_s / 2.0
    # For equal +a/−a over T: displacement after ZUPT ≈ a*(T/2)^2 / 2 ... use known formula.
    # Simpler: integrate analytically — peak vel = a*half; ZUPT kills end vel;
    # position ≈ a * half^2 / 2 for the accel phase + coast of corrected vel.
    # Empirically pick a so zupt_travel ≈ step_m; tests will assert.
    # With ZUPT ramp removal, equal +a/−a over duration T=2*half yields ≈ a * half^2.
    a_mps2 = step_m / (half * half) if half > 0 else 0.0
    a_g = a_mps2 / G
    n_half = max(1, int(round(half / dt)))
    for i in range(n_half):
        samples.append(ImuSample(t + i * dt, accel_scale, 0.0, a_g * accel_scale, 0.0, 0.0, 0.0))
    t = samples[-1].t + dt
    for i in range(n_half):
        samples.append(ImuSample(t + i * dt, accel_scale, 0.0, -a_g * accel_scale, 0.0, 0.0, 0.0))
    t = samples[-1].t + dt
    samples.extend(synthetic_still(t, n=8, dt=dt, ax=accel_scale))
    return samples


def synthetic_blocked_pulse(t0: float, **kw) -> list[ImuSample]:
    """Net-zero pulse plus residual lean toward face-down (Δlean ≳ LEAN_RETURN_DEG)."""
    samples = synthetic_forward_pulse(t0, step_m=0.0, **{k: v for k, v in kw.items() if k != "step_m"})
    # Standing was ~90°; ax=0.7, az=0.7 → lean ~45° (Δ≈45 from still0).
    for i in range(-5, 0):
        s = samples[i]
        samples[i] = ImuSample(s.t, 0.7, 0.0, 0.7, 0.0, 0.0, 0.0)
    return samples


def synthetic_fallen(t0: float, *, face_down: bool = True, dt: float = 0.02) -> list[ImuSample]:
    if face_down:
        # atan2(ax, az) ~ 0 → ax~0, az>0
        return [ImuSample(t0 + i * dt, 0.05, 0.0, 1.0, 0.0, 0.0, 0.0) for i in range(15)]
    # on back ~180 → ax~0, az<0
    return [ImuSample(t0 + i * dt, 0.05, 0.0, -1.0, 0.0, 0.0, 0.0) for i in range(15)]


def synthetic_turn(
    t0: float,
    *,
    deg: float = 15.0,
    duration_s: float = 0.4,
    dt: float = 0.02,
    gyro_scale_dps: float = 1.0,
    accel_scale: float = 1.0,
) -> list[ImuSample]:
    """In-place yaw about X at constant rate."""
    samples = synthetic_still(t0, n=5, dt=dt, ax=accel_scale)
    t = samples[-1].t + dt
    rate = deg / duration_s  # deg/s in scaled units → raw = rate / scale
    raw = rate / gyro_scale_dps
    n = max(1, int(round(duration_s / dt)))
    for i in range(n):
        samples.append(ImuSample(t + i * dt, accel_scale, 0.0, 0.0, raw, 0.0, 0.0))
    t = samples[-1].t + dt
    samples.extend(synthetic_still(t, n=5, dt=dt, ax=accel_scale))
    return samples
