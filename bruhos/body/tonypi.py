"""TonyPi / TonyPi Pro body: Hiwonder action groups + camera + IMU.

Primitives map to action-group files (``.d6a``) shipped with the robot image. Names differ between
firmware images, so the mapping lives in ``configs/tonypi*.toml`` — check them against the
``ActionGroups/`` directory on your robot before the first run.

Roam uses the controller-board IMU stream (``board.get_imu``) after ``enable_reception``.
The optional smbus MPU6050 remains for a raw chip / unit tests.
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
import time
from typing import Any

from bruhos.body.base import PRIMITIVES, Body
from bruhos.body.imu import BoardImu, ImuSample
from bruhos.schemas import Imu, LineReading, Observation

SDK_MODULES = ("hiwonder.ActionGroupControl", "HiwonderSDK.ActionGroupControl", "ActionGroupControl")


def _load_agc(sdk_path: str | None):
    candidates = []
    if sdk_path:
        candidates.append(sdk_path)
        # Common layout: TonyPi/HiwonderSDK/hiwonder/…
        from pathlib import Path
        nested = Path(sdk_path) / "HiwonderSDK"
        if nested.is_dir():
            candidates.append(str(nested))
    for p in candidates:
        if p not in sys.path:
            sys.path.insert(0, p)
    errors = []
    for mod in SDK_MODULES:
        try:
            return importlib.import_module(mod)
        except ImportError as e:
            errors.append(f"{mod}: {e}")
    raise ImportError("Hiwonder ActionGroupControl not found; set [body].sdk_path. Tried: " + "; ".join(errors))


class TonyPiBody(Body):
    name = "tonypi"

    def __init__(self, cfg: dict[str, Any]):
        body = cfg.get("body", {})
        self.name = body.get("name", "tonypi")
        self.turn_step_deg = float(body.get("turn_step_deg", 15.0))
        self.actions: dict[str, str | None] = {p: None for p in PRIMITIVES} | cfg.get("actions", {})
        self.agc = _load_agc(body.get("sdk_path"))
        self.action_path = body.get("action_path")
        self._lock = threading.Lock()
        self._halted = threading.Event()
        self.frame = 0
        self.holding: str | None = None
        self._last_seen: dict[str, Any] = {}
        self.board_imu: BoardImu | None = None
        self.imu = None  # smbus MPU6050 optional
        self.head = None
        self._setup_sensors(cfg)

    def _setup_sensors(self, cfg: dict[str, Any]) -> None:
        cam = cfg.get("camera", {})
        self.cap = self.detector = self.line = None
        self.remote_detect = None
        if cam.get("enabled", True):
            try:
                import cv2

                from bruhos.detect import make_hot_loop_detector, make_remote_client
                from .perception import LineDetector

                self.cap = cv2.VideoCapture(cam.get("device", 0))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.get("width", 640))
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.get("height", 480))
                # Hot-loop: always local (hsv / onnx). Never zerollama on-device.
                self.detector = make_hot_loop_detector(cfg)
                self.line = LineDetector()
                try:
                    self.remote_detect = make_remote_client(cfg)
                except ValueError:
                    self.remote_detect = None
            except Exception as e:  # noqa: BLE001
                print(f"[tonypi] camera disabled: {e}", flush=True)
                self.cap = self.detector = self.line = self.remote_detect = None

        if cfg.get("head"):
            try:
                from bruhos.body.head import Head

                # Reuse AGC's board — a second Board() would steal /dev/ttyAMA0.
                ctl = None
                board = getattr(self.agc, "board", None)
                if board is not None:
                    try:
                        from hiwonder.Controller import Controller

                        ctl = Controller(board)
                    except ImportError:
                        ctl = None
                self.head = Head.from_config(cfg, ctl=ctl)
                self.head.recenter()
                print("[tonypi] head PWM enabled", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[tonypi] head disabled: {e}", flush=True)
                self.head = None

        imu = cfg.get("imu", {})
        roam = cfg.get("roam", {})
        # Prefer controller-board stream (same Board ActionGroupControl already opened).
        board = getattr(self.agc, "board", None)
        if board is not None and imu.get("enabled", True):
            try:
                self.board_imu = BoardImu(
                    board,
                    fall_forward_deg=float(roam.get("fall_forward_deg", 15.0)),
                    fall_back_deg=float(roam.get("fall_back_deg", 160.0)),
                    forward_sign=int(imu.get("forward_sign", 1)),
                )
                self.board_imu.enable()
                print("[tonypi] board IMU reception enabled", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[tonypi] board imu disabled: {e}", flush=True)
                self.board_imu = None

        # Optional smbus chip — only if board stream missing
        if self.board_imu is None and imu.get("enabled", True) and imu.get("smbus", False):
            try:
                from .imu import MPU6050

                self.imu = MPU6050(
                    imu.get("bus", 1),
                    int(str(imu.get("address", "0x68")), 0),
                    imu.get("fall_deg", 55.0),
                    imu.get("forward_sign", 1),
                )
            except Exception as e:  # noqa: BLE001
                print(f"[tonypi] imu disabled: {e}", flush=True)
                self.imu = None

    @property
    def has_board_imu(self) -> bool:
        return self.board_imu is not None

    def poll_imu(self) -> ImuSample | None:
        if self.board_imu is None:
            return None
        return self.board_imu.poll()

    def observe(self) -> Observation:
        self.frame += 1
        objects, line = [], LineReading()
        if self.cap is not None:
            ok, frame = self.cap.read()
            if ok:
                objects = self.detector.detect(frame)
                line = self.line.read(frame)
        if self.board_imu is not None:
            imu = self.board_imu.read()
        elif self.imu is not None:
            imu = self.imu.read()
        else:
            imu = Imu()
        return Observation(t=round(time.monotonic(), 3), frame=self.frame, objects=objects,
                           imu=imu, line=line, holding=self.holding)

    def _run_action(self, group: str) -> bool:
        path = self.action_path or "/home/pi/TonyPi/ActionGroups/"
        if not path.endswith("/"):
            path += "/"
        d6a = path + group + ".d6a"
        if not os.path.isfile(d6a):
            print(f"[tonypi] missing action file {d6a}", flush=True)
            return False
        with self._lock:
            try:
                # runAction plays the file once. runActionGroup('go_forward') only
                # plays go_forward_start on the first call and never the step.
                self.agc.runAction(group, path=path)
            except TypeError:
                self.agc.runActionGroup(group, path=path)
        return True

    def act(self, primitive: str) -> bool:
        if self._halted.is_set():
            return False
        group = self.actions.get(primitive)
        if not group:
            return False
        ok = self._run_action(group)
        if not ok:
            return False
        if primitive == "grab":
            self.holding = "object"
        elif primitive == "release":
            self.holding = None
        return True

    def act_watched(self, primitive: str) -> list[ImuSample]:
        """Play one action while draining board IMU. Does not halt on a single spike."""
        if self._halted.is_set():
            return []
        group = self.actions.get(primitive)
        if not group:
            return []
        samples: list[ImuSample] = []
        stop = threading.Event()

        def _sampler() -> None:
            while not stop.is_set():
                s = self.poll_imu()
                if s is not None:
                    samples.append(s)
                else:
                    time.sleep(0.005)

        th = threading.Thread(target=_sampler, daemon=True)
        th.start()
        try:
            self.act(primitive)
        finally:
            stop.set()
            th.join(timeout=2.0)
        return samples

    def halt(self) -> None:
        self._halted.set()
        stop = getattr(self.agc, "stopAction", None) or getattr(self.agc, "stop_action_group", None)
        if stop:
            stop()
        if self.actions.get("stand"):
            threading.Thread(target=self.agc.runActionGroup, args=(self.actions["stand"],), daemon=True).start()

    def resume(self) -> None:
        self._halted.clear()

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        if self.board_imu:
            self.board_imu.close()
        if self.imu:
            self.imu.close()
