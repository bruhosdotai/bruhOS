"""Mixamo T-pose Tony: bone table, stand pulses, and servo notes.

The puppet is what prompt2rig and RGB MoCap load. It is not the robot's
servo tree. Proportions are for the preview; copied rotations do not depend
on them. Stand pulses are stored here and not applied — the Mixamo rest pose
stays a T-pose.
"""

from __future__ import annotations

from dataclasses import dataclass

# RGB MoCap / Mixamo body set — each name must appear exactly once.
MIXAMO_BODY = (
    "Hips",
    "LeftUpLeg",
    "RightUpLeg",
    "Spine",
    "LeftLeg",
    "RightLeg",
    "Spine1",
    "LeftFoot",
    "RightFoot",
    "Spine2",
    "LeftToeBase",
    "RightToeBase",
    "Neck",
    "LeftShoulder",
    "RightShoulder",
    "Head",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
)

# prompt2rig aim() needs a Middle1 child; palm roll also wants Index1 and Pinky1.
FINGER_SUFFIXES = ("HandThumb1", "HandIndex1", "HandMiddle1", "HandRing1", "HandPinky1")

PREFIX = "mixamorig:"

# Mesh height ≈ published TonyPi height (373 mm).
MESH_HEIGHT_M = 0.373

# Arm links from /home/pi/TonyPi/Functions/pose_control.py.
UPPER_ARM_M = 0.06
FOREARM_M = 0.11
FINGER_M = 0.008

# stand.d6a Servo1..Servo18. Right-side pairs mirror through 500 (sum to 1000).
STAND_PULSES: dict[int, int] = {
    1: 500,
    2: 390,
    3: 500,
    4: 600,
    5: 500,
    6: 575,
    7: 800,
    8: 725,
    9: 500,
    10: 610,
    11: 500,
    12: 400,
    13: 500,
    14: 425,
    15: 200,
    16: 275,
    17: 500,  # gripper (not a Mixamo bone)
    18: 500,  # gripper (not a Mixamo bone)
}

# Servo id → Mixamo bone. Shared bones need a later hinge split.
# Arms: pose_control maps upper-arm angle → 7/15, elbow → 6/14; hand files move 8/16.
SERVO_BONE: dict[int, str] = {
    1: "LeftFoot",       # foot roll  (shares with 2)
    2: "LeftFoot",       # ankle pitch
    3: "LeftLeg",        # knee
    4: "LeftUpLeg",      # hip pitch  (shares with 5)
    5: "LeftUpLeg",      # hip roll
    6: "LeftForeArm",    # elbow
    7: "LeftArm",        # upper arm
    8: "LeftHand",       # hand
    9: "RightFoot",
    10: "RightFoot",
    11: "RightLeg",
    12: "RightUpLeg",
    13: "RightUpLeg",
    14: "RightForeArm",
    15: "RightArm",
    16: "RightHand",
}

GRIPPER_SERVOS = (17, 18)

# VRM 0.0 humanoid bone name for each Mixamo body bone (RGB MoCap mapping.py).
VRM_HUMAN_BONE = {
    "Hips": "hips",
    "LeftUpLeg": "leftUpperLeg",
    "RightUpLeg": "rightUpperLeg",
    "Spine": "spine",
    "LeftLeg": "leftLowerLeg",
    "RightLeg": "rightLowerLeg",
    "Spine1": "chest",
    "LeftFoot": "leftFoot",
    "RightFoot": "rightFoot",
    "Spine2": "upperChest",
    "LeftToeBase": "leftToes",
    "RightToeBase": "rightToes",
    "Neck": "neck",
    "LeftShoulder": "leftShoulder",
    "RightShoulder": "rightShoulder",
    "Head": "head",
    "LeftArm": "leftUpperArm",
    "RightArm": "rightUpperArm",
    "LeftForeArm": "leftLowerArm",
    "RightForeArm": "rightLowerArm",
    "LeftHand": "leftHand",
    "RightHand": "rightHand",
}


@dataclass(frozen=True)
class BoneDef:
    """One Mixamo bone: short name, parent short name (or None), local translation (m)."""

    name: str
    parent: str | None
    translation: tuple[float, float, float]


def _finger_bones(side: str, hand: str) -> list[BoneDef]:
    """Palms-down T-pose: fingers along the arm, thumb toward +Z (forward)."""
    # Left arm points +X, right arm −X. Spread runs thumb(+Z) → pinky(−Z).
    outward = FINGER_M if side == "Left" else -FINGER_M
    out: list[BoneDef] = []
    for i, suf in enumerate(FINGER_SUFFIXES):
        across = (2 - i) * 0.004  # Thumb…Pinky: +Z → −Z
        if suf == "HandThumb1":
            # Mostly forward; a little outward so aim() still has a child axis.
            out.append(BoneDef(f"{side}{suf}", hand, (outward * 0.35, 0.0, FINGER_M)))
        else:
            out.append(BoneDef(f"{side}{suf}", hand, (outward, 0.0, across)))
    return out


def bone_defs() -> list[BoneDef]:
    """Parent-first Mixamo hierarchy. Y-up, face +Z, character left = +X."""
    # Sized so the skinned mesh sits near MESH_HEIGHT_M with soles on y = 0.
    thigh = 0.082
    shin = 0.078
    foot = 0.022
    toe = 0.018
    hip_y = thigh + shin + foot  # ≈ 0.182

    half_hip = 0.032
    spine_step = 0.030
    neck_len = 0.024
    head_len = 0.048
    shoulder_out = 0.026
    clavicle = 0.020

    bones: list[BoneDef] = [
        BoneDef("Hips", None, (0.0, hip_y, 0.0)),
        BoneDef("Spine", "Hips", (0.0, spine_step, 0.0)),
        BoneDef("Spine1", "Spine", (0.0, spine_step, 0.0)),
        BoneDef("Spine2", "Spine1", (0.0, spine_step, 0.0)),
        BoneDef("Neck", "Spine2", (0.0, neck_len, 0.0)),
        BoneDef("Head", "Neck", (0.0, head_len, 0.0)),
        # Legs
        BoneDef("LeftUpLeg", "Hips", (half_hip, -0.008, 0.0)),
        BoneDef("LeftLeg", "LeftUpLeg", (0.0, -thigh, 0.0)),
        BoneDef("LeftFoot", "LeftLeg", (0.0, -shin, 0.015)),
        BoneDef("LeftToeBase", "LeftFoot", (0.0, -foot * 0.15, toe)),
        BoneDef("RightUpLeg", "Hips", (-half_hip, -0.008, 0.0)),
        BoneDef("RightLeg", "RightUpLeg", (0.0, -thigh, 0.0)),
        BoneDef("RightFoot", "RightLeg", (0.0, -shin, 0.015)),
        BoneDef("RightToeBase", "RightFoot", (0.0, -foot * 0.15, toe)),
        # Arms — left on +X
        BoneDef("LeftShoulder", "Spine2", (shoulder_out, spine_step * 0.4, 0.0)),
        BoneDef("LeftArm", "LeftShoulder", (clavicle, 0.0, 0.0)),
        BoneDef("LeftForeArm", "LeftArm", (UPPER_ARM_M, 0.0, 0.0)),
        BoneDef("LeftHand", "LeftForeArm", (FOREARM_M, 0.0, 0.0)),
        BoneDef("RightShoulder", "Spine2", (-shoulder_out, spine_step * 0.4, 0.0)),
        BoneDef("RightArm", "RightShoulder", (-clavicle, 0.0, 0.0)),
        BoneDef("RightForeArm", "RightArm", (-UPPER_ARM_M, 0.0, 0.0)),
        BoneDef("RightHand", "RightForeArm", (-FOREARM_M, 0.0, 0.0)),
    ]
    bones.extend(_finger_bones("Left", "LeftHand"))
    bones.extend(_finger_bones("Right", "RightHand"))
    return bones


def prefixed(name: str) -> str:
    return PREFIX + name


def world_positions(bones: list[BoneDef] | None = None) -> dict[str, tuple[float, float, float]]:
    """Accumulate local translations into world positions (identity rotations)."""
    defs = bones or bone_defs()
    by_name = {b.name: b for b in defs}
    cache: dict[str, tuple[float, float, float]] = {}

    def walk(name: str) -> tuple[float, float, float]:
        if name in cache:
            return cache[name]
        b = by_name[name]
        if b.parent is None:
            cache[name] = b.translation
            return cache[name]
        px, py, pz = walk(b.parent)
        tx, ty, tz = b.translation
        cache[name] = (px + tx, py + ty, pz + tz)
        return cache[name]

    for b in defs:
        walk(b.name)
    return cache


def mesh_height_m(bones: list[BoneDef] | None = None) -> float:
    """Approximate skinned height: head tip above the lowest foot/toe."""
    pos = world_positions(bones)
    ys = [p[1] for p in pos.values()]
    head = pos["Head"][1] + 0.03  # head box extends above the Head bone
    sole = min(ys) - 0.01
    return head - sole


def assert_stand_mirror() -> None:
    pairs = ((1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15), (8, 16))
    for left, right in pairs:
        if STAND_PULSES[left] + STAND_PULSES[right] != 1000:
            raise AssertionError(
                f"stand mirror broken: servo {left}={STAND_PULSES[left]} "
                f"+ {right}={STAND_PULSES[right]}"
            )
