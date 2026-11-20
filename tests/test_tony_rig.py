"""Tony Mixamo puppet + whole-clip playback scale."""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RIG = ROOT / "rig"


def _load(name: str):
    path = RIG / f"{name}.py"
    mod_name = f"tony_rig_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    # export_tony imports tony_bones by bare name
    sys.path.insert(0, str(RIG))
    try:
        spec.loader.exec_module(mod)
    finally:
        if sys.path[0] == str(RIG):
            sys.path.pop(0)
    return mod


tony_bones = _load("tony_bones")
timing = _load("timing")


def _parse_glb(path: Path) -> dict:
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    assert magic == b"glTF" and version == 2 and length == len(data)
    offset = 12
    json_len, json_type = struct.unpack_from("<I4s", data, offset)
    offset += 8
    assert json_type == b"JSON"
    return json.loads(data[offset : offset + json_len].decode("utf-8"))


def _short(name: str) -> str:
    return name.replace(tony_bones.PREFIX, "")


def _world_from_nodes(gltf: dict) -> dict[str, tuple[float, float, float]]:
    nodes = gltf["nodes"]
    by_name = {_short(n["name"]): i for i, n in enumerate(nodes) if n.get("name", "").startswith(tony_bones.PREFIX)}
    cache: dict[int, tuple[float, float, float]] = {}

    def walk(i: int) -> tuple[float, float, float]:
        if i in cache:
            return cache[i]
        n = nodes[i]
        t = n.get("translation", [0.0, 0.0, 0.0])
        # Find parent: the node that lists i in children
        parent = None
        for pi, pn in enumerate(nodes):
            if i in pn.get("children", []):
                parent = pi
                break
        if parent is None:
            cache[i] = (float(t[0]), float(t[1]), float(t[2]))
        else:
            px, py, pz = walk(parent)
            cache[i] = (px + float(t[0]), py + float(t[1]), pz + float(t[2]))
        return cache[i]

    out = {}
    for name, i in by_name.items():
        out[name] = walk(i)
    return out


@pytest.fixture(scope="module")
def gltf():
    glb = RIG / "tony.glb"
    if not glb.is_file():
        export = _load("export_tony")
        export.main([])
    return _parse_glb(glb)


def test_required_mixamo_bones(gltf):
    names = {_short(n["name"]) for n in gltf["nodes"] if n.get("name", "").startswith(tony_bones.PREFIX)}
    for need in tony_bones.MIXAMO_BODY:
        assert need in names, f"missing {need}"
    for need in tony_bones.MIXAMO_BODY:
        assert sum(1 for n in names if n == need) == 1


def test_finger_bones_outward_and_thumb_forward(gltf):
    pos = _world_from_nodes(gltf)
    for side, outward_sign in (("Left", 1.0), ("Right", -1.0)):
        hand = pos[f"{side}Hand"]
        mid = pos[f"{side}HandMiddle1"]
        thumb = pos[f"{side}HandThumb1"]
        dx = mid[0] - hand[0]
        dy = mid[1] - hand[1]
        dz = mid[2] - hand[2]
        length = (dx * dx + dy * dy + dz * dz) ** 0.5
        assert length >= 0.004, f"{side}HandMiddle1 too short: {length}"
        assert dx * outward_sign > 0, f"{side} fingers must point along the arm"
        assert thumb[2] > hand[2], f"{side} thumb must point forward (+Z)"
        assert f"{side}HandIndex1" in pos
        assert f"{side}HandPinky1" in pos


def test_left_arm_on_plus_x(gltf):
    pos = _world_from_nodes(gltf)
    assert pos["LeftArm"][0] > pos["RightArm"][0]


def test_chest_basis_nonzero(gltf):
    pos = _world_from_nodes(gltf)
    assert pos["Spine2"][1] > pos["Hips"][1]
    assert pos["Neck"][1] > pos["Hips"][1]


def test_mesh_height_near_tony(gltf):
    acc = gltf["accessors"][0]
    ymin, ymax = acc["min"][1], acc["max"][1]
    height = ymax - ymin
    assert ymin >= -1e-4, ymin
    assert 0.36 <= height <= 0.385, height
    assert abs(height - tony_bones.MESH_HEIGHT_M) < 0.02


def test_stand_pulse_mirror():
    tony_bones.assert_stand_mirror()
    pairs = ((1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15), (8, 16))
    for left, right in pairs:
        assert tony_bones.STAND_PULSES[left] + tony_bones.STAND_PULSES[right] == 1000


def test_grippers_are_not_body_bones():
    assert tony_bones.GRIPPER_SERVOS == (17, 18)
    assert 17 not in tony_bones.SERVO_BONE
    assert 18 not in tony_bones.SERVO_BONE
    for sid, bone in tony_bones.SERVO_BONE.items():
        assert 1 <= sid <= 16
        assert bone in tony_bones.MIXAMO_BODY


def test_arm_servo_map_matches_pose_control():
    assert tony_bones.SERVO_BONE[6] == "LeftForeArm"
    assert tony_bones.SERVO_BONE[7] == "LeftArm"
    assert tony_bones.SERVO_BONE[8] == "LeftHand"
    assert tony_bones.SERVO_BONE[14] == "RightForeArm"
    assert tony_bones.SERVO_BONE[15] == "RightArm"
    assert tony_bones.SERVO_BONE[16] == "RightHand"


def test_vrm_same_bytes_as_glb():
    glb = (RIG / "tony.glb").read_bytes()
    vrm = (RIG / "tony.vrm").read_bytes()
    assert glb == vrm
    assert glb[:4] == b"glTF"


def test_vrm_humanoid_has_22_slots(gltf):
    bones = gltf["extensions"]["VRM"]["humanoid"]["humanBones"]
    assert len(bones) == 22
    names = {b["bone"] for b in bones}
    assert names == set(tony_bones.VRM_HUMAN_BONE.values())


def test_vrm_meta_present(gltf):
    meta = gltf["extensions"]["VRM"]["meta"]
    assert meta["title"] == "Tony"
    assert meta["author"] == "bruhOS"


def test_scale_from_stand_reports_limiting_step():
    stand = {i: tony_bones.STAND_PULSES[i] for i in range(1, 17)}
    # 180° on servo 7 (= 750 counts) in 0.2 s; half that on servo 6.
    # 180 deg → 180 * (1000/240) = 750 counts.
    frame = dict(stand)
    frame[7] = stand[7] + 750
    frame[6] = stand[6] + 375
    result = timing.scale_take(stand, [frame], [0.2])
    # needed = 180/300 = 0.6; stretch = 0.6/0.2 = 3; *1.15 = 3.45
    assert result.limiting_step == 0
    assert result.scale == pytest.approx(3.45)
    assert result.step_ms[0] == 690
    # Both joints share the stretched time — ratio survives by construction.


def test_scale_never_speeds_up():
    stand = {i: 500 for i in range(1, 17)}
    frame = {i: 500 for i in range(1, 17)}
    frame[3] = 510  # tiny move
    result = timing.scale_take(stand, [frame], [1.0])
    assert result.scale == 1.0
    assert result.step_ms[0] == 1000


def test_scale_includes_departure_from_stand_as_step_zero():
    stand = {i: tony_bones.STAND_PULSES[i] for i in range(1, 17)}
    quiet = dict(stand)
    quiet[8] = stand[8] + 10
    loud = dict(quiet)
    loud[8] = quiet[8] + 750  # 180° later
    result = timing.scale_take(stand, [quiet, loud], [0.5, 0.2])
    assert result.limiting_step == 1
    assert result.scale == pytest.approx(3.45)
