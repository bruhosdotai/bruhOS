"""TonyPi / TonyPi Pro body: Hiwonder action groups + camera + IMU.

Primitives map to action-group files (``.d6a``) shipped with the robot image. Names differ between
firmware images, so the mapping lives in ``configs/tonypi*.toml`` — check them against the
``ActionGroups/`` directory on your robot before the first run.
"""

from __future__ import annotations

import importlib
import sys
import threading
import time
from typing import Any

from bruhos.body.base import PRIMITIVES, Body
from bruhos.schemas import Imu, LineReading, Observation

SDK_MODULES = ("hiwonder.ActionGroupControl", "HiwonderSDK.ActionGroupControl", "ActionGroupControl")


def _load_agc(sdk_path: str | None):
    if sdk_path and sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)
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
        self._setup_sensors(cfg)

    def _setup_sensors(self, cfg: dict[str, Any]) -> None:
        cam = cfg.get("camera", {})
        self.cap = self.detector = self.line = self.imu = None
        if cam.get("enabled", True):
            import cv2

            from .perception import CameraModel, ColorBlobDetector, LineDetector
            self.cap = cv2.VideoCapture(cam.get("device", 0))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.get("width", 640))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.get("height", 480))
            model = CameraModel(cam.get("width", 640), cam.get("height", 480), cam.get("hfov_deg", 62.0))
            hsv = cfg.get("perception", {}).get("hsv")
            self.detector = ColorBlobDetector(model, **({"hsv": hsv} if hsv else {}))
            self.line = LineDetector()
        imu = cfg.get("imu", {})
        if imu.get("enabled", True):
            from .imu import MPU6050
            self.imu = MPU6050(imu.get("bus", 1), int(str(imu.get("address", "0x68")), 0),
                               imu.get("fall_deg", 55.0), imu.get("forward_sign", 1))

    def observe(self) -> Observation:
        self.frame += 1
        objects, line = [], LineReading()
        if self.cap is not None:
            ok, frame = self.cap.read()
            if ok:
                objects = self.detector.detect(frame)
                line = self.line.read(frame)
        imu = self.imu.read() if self.imu else Imu()
        return Observation(t=round(time.monotonic(), 3), frame=self.frame, objects=objects,
                           imu=imu, line=line, holding=self.holding)

    def act(self, primitive: str) -> bool:
        if self._halted.is_set():
            return False
        group = self.actions.get(primitive)
        if not group:
            return False
        with self._lock:
            kw = {"path": self.action_path} if self.action_path else {}
            try:
                self.agc.runActionGroup(group, **kw)
            except TypeError:
                self.agc.runActionGroup(group)
        if primitive == "grab":
            self.holding = "object"
        elif primitive == "release":
            self.holding = None
        return True

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
        if self.imu:
            self.imu.close()
