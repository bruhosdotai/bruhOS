#!/usr/bin/env python3
"""Build tony.glb / tony.vrm (same bytes): Mixamo T-pose boxes + VRM 0.0 humanoid."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

from tony_bones import (
    MIXAMO_BODY,
    VRM_HUMAN_BONE,
    BoneDef,
    bone_defs,
    mesh_height_m,
    prefixed,
    world_positions,
)

ROOT = Path(__file__).resolve().parent
OUT_GLB = ROOT / "tony.glb"
OUT_VRM = ROOT / "tony.vrm"


def _box(cx: float, cy: float, cz: float, sx: float, sy: float, sz: float) -> tuple[list[float], list[int]]:
    """Axis-aligned box centered at (cx,cy,cz) with size (sx,sy,sz)."""
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    corners = [
        (cx - hx, cy - hy, cz - hz),
        (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
    ]
    # Two triangles per face
    faces = (
        (0, 1, 2, 0, 2, 3),  # -Z
        (4, 6, 5, 4, 7, 6),  # +Z
        (0, 4, 5, 0, 5, 1),  # -Y
        (2, 6, 7, 2, 7, 3),  # +Y
        (0, 3, 7, 0, 7, 4),  # -X
        (1, 5, 6, 1, 6, 2),  # +X
    )
    positions: list[float] = []
    indices: list[int] = []
    base = 0
    for face in faces:
        for vi in face[:3]:
            positions.extend(corners[vi])
        indices.extend((base, base + 1, base + 2))
        base += 3
        for vi in face[3:]:
            positions.extend(corners[vi])
        indices.extend((base, base + 1, base + 2))
        base += 3
    return positions, indices


def _segment_boxes(defs: list[BoneDef], pos: dict[str, tuple[float, float, float]]) -> list[tuple[str, list[float], list[int]]]:
    """One box per body segment, skinned to the parent bone of that segment."""
    parts: list[tuple[str, list[float], list[int]]] = []

    def add(bone: str, center: tuple[float, float, float], size: tuple[float, float, float]) -> None:
        p, i = _box(*center, *size)
        parts.append((bone, p, i))

    # Pelvis / hips
    hx, hy, hz = pos["Hips"]
    add("Hips", (hx, hy, hz), (0.09, 0.05, 0.06))

    # Rigid-looking chest covering Spine..Spine2
    sx, sy, sz = pos["Spine"]
    s2x, s2y, s2z = pos["Spine2"]
    add("Spine1", ((sx + s2x) * 0.5, (sy + s2y) * 0.5, (sz + s2z) * 0.5), (0.10, 0.08, 0.07))

    # Head + camera nub (top lands near MESH_HEIGHT_M once soles sit on y = 0)
    hx, hy, hz = pos["Head"]
    add("Head", (hx, hy + 0.012, hz), (0.055, 0.05, 0.055))
    add("Head", (hx, hy + 0.015, hz + 0.035), (0.028, 0.022, 0.018))

    for side, sign in (("Left", 1.0), ("Right", -1.0)):
        # Upper arm: Midway Arm → ForeArm
        ax, ay, az = pos[f"{side}Arm"]
        fx, fy, fz = pos[f"{side}ForeArm"]
        add(f"{side}Arm", ((ax + fx) * 0.5, (ay + fy) * 0.5, (az + fz) * 0.5), (0.055, 0.03, 0.03))

        # Forearm
        hx2, hy2, hz2 = pos[f"{side}Hand"]
        add(
            f"{side}ForeArm",
            ((fx + hx2) * 0.5, (fy + hy2) * 0.5, (fz + hz2) * 0.5),
            (0.10, 0.025, 0.025),
        )

        # Flat palm
        add(f"{side}Hand", (hx2 + sign * 0.015, hy2, hz2), (0.04, 0.012, 0.035))

        # Thigh / shin / foot
        ux, uy, uz = pos[f"{side}UpLeg"]
        lx, ly, lz = pos[f"{side}Leg"]
        add(f"{side}UpLeg", ((ux + lx) * 0.5, (uy + ly) * 0.5, (uz + lz) * 0.5), (0.04, 0.08, 0.04))

        fx2, fy2, fz2 = pos[f"{side}Foot"]
        add(f"{side}Leg", ((lx + fx2) * 0.5, (ly + fy2) * 0.5, (lz + fz2) * 0.5), (0.035, 0.075, 0.035))

        tx, ty, tz = pos[f"{side}ToeBase"]
        foot_h = 0.022
        add(
            f"{side}Foot",
            ((fx2 + tx) * 0.5, foot_h * 0.5, (fz2 + tz) * 0.5 + 0.01),
            (0.04, foot_h, 0.07),
        )

    return parts


def _pack_f32(values: list[float]) -> bytes:
    return struct.pack("<" + "f" * len(values), *values)


def _pack_u16(values: list[int]) -> bytes:
    return struct.pack("<" + "H" * len(values), *values)


def _pack_u32(values: list[int]) -> bytes:
    return struct.pack("<" + "I" * len(values), *values)


def build_glb() -> bytes:
    defs = bone_defs()
    pos = world_positions(defs)
    # Joint order: body bones first (stable VRM indices), then fingers.
    body = [b for b in defs if b.name in MIXAMO_BODY]
    fingers = [b for b in defs if b.name not in MIXAMO_BODY]
    ordered = body + fingers
    name_to_joint = {b.name: i for i, b in enumerate(ordered)}

    # --- binary blobs ---
    # Inverse bind matrices: world-from-joint inverse at rest (identity rotation → -translation).
    ibm: list[float] = []
    for b in ordered:
        wx, wy, wz = pos[b.name]
        # Column-major 4x4: translation in last column of row-major becomes…
        # glTF uses column-major. Identity with translation -T:
        # [1 0 0 -wx]
        # [0 1 0 -wy]
        # [0 0 1 -wz]
        # [0 0 0  1 ]
        ibm.extend(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                -wx,
                -wy,
                -wz,
                1.0,
            ]
        )

    parts = _segment_boxes(defs, pos)
    positions: list[float] = []
    joints: list[int] = []
    weights: list[float] = []
    indices: list[int] = []
    for bone, p, idx in parts:
        base = len(positions) // 3
        j = name_to_joint[bone]
        nverts = len(p) // 3
        positions.extend(p)
        for _ in range(nverts):
            joints.extend((j, 0, 0, 0))
            weights.extend((1.0, 0.0, 0.0, 0.0))
        indices.extend(base + i for i in idx)

    # Align helpers
    def align4(buf: bytearray) -> None:
        while len(buf) % 4:
            buf.append(0)

    blob = bytearray()
    # accessor layout:
    # 0 POSITION (f32 vec3)
    # 1 JOINTS_0 (u16 vec4)
    # 2 WEIGHTS_0 (f32 vec4)
    # 3 indices (u16 or u32)
    # 4 inverseBindMatrices (f32 mat4)

    pos_bytes = _pack_f32(positions)
    pos_off = len(blob)
    blob.extend(pos_bytes)
    align4(blob)

    joint_bytes = _pack_u16(joints)
    joint_off = len(blob)
    blob.extend(joint_bytes)
    align4(blob)

    weight_bytes = _pack_f32(weights)
    weight_off = len(blob)
    blob.extend(weight_bytes)
    align4(blob)

    if max(indices, default=0) > 65535:
        idx_bytes = _pack_u32(indices)
        idx_comp = 5125  # UNSIGNED_INT
    else:
        idx_bytes = _pack_u16(indices)
        idx_comp = 5123  # UNSIGNED_SHORT
    idx_off = len(blob)
    blob.extend(idx_bytes)
    align4(blob)

    ibm_bytes = _pack_f32(ibm)
    ibm_off = len(blob)
    blob.extend(ibm_bytes)
    align4(blob)

    nverts = len(positions) // 3
    xs = positions[0::3]
    ys = positions[1::3]
    zs = positions[2::3]
    pos_min = [min(xs), min(ys), min(zs)]
    pos_max = [max(xs), max(ys), max(zs)]

    # Nodes: joint nodes, then mesh node
    nodes: list[dict] = []
    for b in ordered:
        node: dict = {"name": prefixed(b.name), "translation": list(b.translation)}
        children = [name_to_joint[c.name] for c in ordered if c.parent == b.name]
        if children:
            node["children"] = children
        nodes.append(node)

    mesh_node_index = len(nodes)
    hips_index = name_to_joint["Hips"]
    nodes.append({"name": "TonyMesh", "mesh": 0, "skin": 0})

    # Scene root holds hips + mesh
    root_index = len(nodes)
    nodes.append({"name": "Tony", "children": [hips_index, mesh_node_index]})

    human_bones = []
    for name in MIXAMO_BODY:
        human_bones.append(
            {
                "bone": VRM_HUMAN_BONE[name],
                "node": name_to_joint[name],
                "useDefaultValues": True,
            }
        )

    gltf = {
        "asset": {"version": "2.0", "generator": "bruhOS/rig/export_tony.py"},
        "extensionsUsed": ["VRM"],
        "extensions": {
            "VRM": {
                "specVersion": "0.0",
                "meta": {
                    "title": "Tony",
                    "version": "0.1",
                    "author": "bruhOS",
                    "allowedUserName": "OnlyAuthor",
                    "violentUssageName": "Disallow",
                    "sexualUssageName": "Disallow",
                    "commercialUssageName": "Disallow",
                    "licenseName": "Redistribution_Prohibited",
                },
                "humanoid": {
                    "armStretch": 0.0,
                    "legStretch": 0.0,
                    "upperArmTwist": 0.5,
                    "lowerArmTwist": 0.5,
                    "upperLegTwist": 0.5,
                    "lowerLegTwist": 0.5,
                    "feetSpacing": 0.0,
                    "hasTranslationDoF": False,
                    "humanBones": human_bones,
                },
            }
        },
        "scene": 0,
        "scenes": [{"name": "Scene", "nodes": [root_index]}],
        "nodes": nodes,
        "meshes": [
            {
                "name": "TonyBody",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "JOINTS_0": 1,
                            "WEIGHTS_0": 2,
                        },
                        "indices": 3,
                        "mode": 4,
                    }
                ],
            }
        ],
        "skins": [
            {
                "name": "TonySkin",
                "inverseBindMatrices": 4,
                "joints": list(range(len(ordered))),
                "skeleton": hips_index,
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": nverts,
                "type": "VEC3",
                "min": pos_min,
                "max": pos_max,
            },
            {
                "bufferView": 1,
                "componentType": 5123,
                "count": nverts,
                "type": "VEC4",
            },
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": nverts,
                "type": "VEC4",
            },
            {
                "bufferView": 3,
                "componentType": idx_comp,
                "count": len(indices),
                "type": "SCALAR",
            },
            {
                "bufferView": 4,
                "componentType": 5126,
                "count": len(ordered),
                "type": "MAT4",
            },
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_off, "byteLength": len(pos_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": joint_off, "byteLength": len(joint_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": weight_off, "byteLength": len(weight_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": idx_off, "byteLength": len(idx_bytes), "target": 34963},
            {"buffer": 0, "byteOffset": ibm_off, "byteLength": len(ibm_bytes)},
        ],
        "buffers": [{"byteLength": len(blob)}],
    }

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    while len(json_bytes) % 4:
        json_bytes += b" "

    total = 12 + 8 + len(json_bytes) + 8 + len(blob)
    out = bytearray()
    out.extend(struct.pack("<4sII", b"glTF", 2, total))
    out.extend(struct.pack("<I4s", len(json_bytes), b"JSON"))
    out.extend(json_bytes)
    out.extend(struct.pack("<I4s", len(blob), b"BIN\x00"))
    out.extend(blob)
    assert len(out) == total
    return bytes(out)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    glb_path = Path(args[0]) if args else OUT_GLB
    vrm_path = Path(args[1]) if len(args) > 1 else OUT_VRM

    data = build_glb()
    glb_path.write_bytes(data)
    vrm_path.write_bytes(data)
    h = mesh_height_m()
    print(f"wrote {glb_path} and {vrm_path} ({len(data)} bytes, height≈{h:.3f} m)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
