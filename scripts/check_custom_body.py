#!/usr/bin/env python3
"""Replay a converted body the way the engine does, and compare it to its source.

`build_custom_pack.py` turns a glTF hierarchy into a matrix-stack program: node
positions, POP/PUSH flags and packed Y-X-Z angles. Nothing about that program is
visible in a structural check -- a wrong stack flag or a mis-signed angle still
produces a perfectly well-formed PKD. So this tool rebuilds every joint's world
transform exactly as `drawViewerNodes` would, and measures it against the same
joint's world transform in the glTF file.

An agreement within the quantisation error proves the hierarchy, the stack flags,
the angle packing and the coordinate flip, without running the ROM.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_custom_pack import (
    DEFAULT_FRAME_RATE, TICKS_PER_SECOND, IDENTITY, ModelSpec, build_joint_mesh,
    build_model, seam_anchors,
    SPLIT_BUDGET, discover_joints, evaluate_pose, load_glyph_strip, load_model,
    mat_mul,
    node_world_matrices, sanitize_name, track_channels,
)

NODE_FLAG_POP = 1 << 0
NODE_FLAG_PUSH = 1 << 1


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]]


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]]


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def translate(x, y, z):
    return [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]


def unpack_angles(word: int) -> tuple[float, float, float]:
    """The engine's DECODE_ANGLES, in radians."""
    ax = (word >> 20) & 0x3FF
    ay = (word >> 10) & 0x3FF
    az = word & 0x3FF
    turn = 2 * math.pi / 1024
    return ax * turn, ay * turn, az * turn


def frame_transform(pos, word) -> list[list[float]]:
    """matrixFrame: translateRel then matrixRotateYXZ, i.e. M * Ry * Rx * Rz."""
    ax, ay, az = unpack_angles(word)
    m = translate(*pos)
    m = mat_mul(m, rot_y(ay))
    m = mat_mul(m, rot_x(ax))
    return mat_mul(m, rot_z(az))


def replay(body: bytes, animation: int, keyframe: int,
           model_index: int = 0, neutral: bool = False) -> list[list[list[float]]]:
    """Rebuild every joint's world matrix the way drawViewerNodes does.

    With `neutral`, every rotation and the root translation are dropped, leaving
    only the skeleton's node offsets. Two bodies built from different clip sets
    have no comparable frame, but they do have a comparable rest shape.
    """
    offsets = struct.unpack_from("<35I", body, 32)
    _, joint_count, _, node_index, anim_index = model_record(body, model_index)

    record = offsets[7] + (anim_index + animation) * 32
    frame_offset, rate = struct.unpack_from("<IB", body, record)
    stride = 20 + 4 * joint_count
    base = offsets[12] + frame_offset + keyframe * stride

    pos = struct.unpack_from("<3h", body, base + 12)
    words = struct.unpack_from(f"<{joint_count}I", body, base + 20)
    if neutral:
        pos = (0, 0, 0)
        words = (0,) * joint_count
    nodes = [
        struct.unpack_from("<3hH", body, offsets[11] + (node_index + i) * 8)
        for i in range(joint_count - 1)
    ]

    current = frame_transform(pos, words[0])
    world = [current]
    stack: list[list[list[float]]] = []
    for i in range(1, joint_count):
        nx, ny, nz, flags = nodes[i - 1]
        if flags & NODE_FLAG_POP:
            if not stack:
                raise SystemExit(f"FAIL: joint {i} pops an empty matrix stack")
            current = stack.pop()
        if flags & NODE_FLAG_PUSH:
            stack.append([row[:] for row in current])
        current = mat_mul(current, frame_transform((nx, ny, nz), words[i]))
        world.append(current)
    return world


def model_record(body: bytes, model_index: int = 0):
    """Return (type, joints, mesh_start, node_index, anim_index) for one model.

    A custom body holds a single model; a Tomb Raider body holds many, so every
    reader has to go through the record rather than assume slot zero.
    """
    offsets = struct.unpack_from("<35I", body, 32)
    counts = struct.unpack_from("<14H", body, 4)
    if not 0 <= model_index < counts[2]:
        raise SystemExit(f"FAIL: model {model_index} is outside 0..{counts[2] - 1}")
    return struct.unpack_from("<BbHHH", body, offsets[13] + model_index * 8)


def find_model(body: bytes, item_type: int) -> int:
    counts = struct.unpack_from("<14H", body, 4)
    for index in range(counts[2]):
        if model_record(body, index)[0] == item_type:
            return index
    raise SystemExit(f"FAIL: no model of type {item_type} in this body")


def decode_mesh(body: bytes, mesh_index: int):
    """Walk one mesh block exactly as faceAddMeshTriangles does.

    Indices are int8 deltas chained from face to face, and a triangle hides its
    third delta in the fourth slot (the engine's "asr hack"). A converter that
    got either wrong still writes a structurally valid PKD, so the only way to
    catch it is to decode the chain and look at the triangles it produces.
    """
    offsets = struct.unpack_from("<35I", body, 32)
    counts = struct.unpack_from("<14H", body, 4)
    mesh_offsets = struct.unpack_from(f"<{counts[3]}I", body, offsets[6])
    start = offsets[5] + mesh_offsets[mesh_index]

    vcount, has_normals = struct.unpack_from("<BB", body, start + 10)
    quads, triangles = struct.unpack_from("<hh", body, start + 12)
    if has_normals:
        raise SystemExit("FAIL: hasNormals must be 0 on this target")
    face_base = start + 20
    vertex_base = face_base + (quads + triangles) * 8
    vertices = [
        struct.unpack_from("<3h", body, vertex_base + i * 6) for i in range(vcount)
    ]

    def check(index, where):
        if not 0 <= index < vcount:
            raise SystemExit(
                f"FAIL: mesh {mesh_index} {where} resolves index {index}, "
                f"outside 0..{vcount - 1}"
            )
        return index

    # Quads come first and carry their own delta chain; triangles restart it.
    quad_faces = []
    previous = 0
    for i in range(quads):
        d0, d1, d2, d3 = struct.unpack_from("<4b", body, face_base + i * 8)
        flags = struct.unpack_from("<H", body, face_base + i * 8 + 4)[0]
        i0 = check(previous + d0, f"quad {i}")
        i1 = check(i0 + d1, f"quad {i}")
        i2 = check(i1 + d2, f"quad {i}")
        i3 = check(i2 + d3, f"quad {i}")
        previous = i3
        quad_faces.append(((i0, i1, i2, i3), flags))

    faces = []
    previous = 0
    for i in range(quads, quads + triangles):
        d0, d1, _, d2 = struct.unpack_from("<4b", body, face_base + i * 8)
        flags = struct.unpack_from("<H", body, face_base + i * 8 + 4)[0]
        i0 = check(previous + d0, f"triangle {i}")
        i1 = check(i0 + d1, f"triangle {i}")
        i2 = check(i1 + d2, f"triangle {i}")
        previous = i2
        faces.append(((i0, i1, i2), flags))
    return vertices, faces, quad_faces


def check_geometry(built, model, joints, scale, windings, two_sided=None,
                   anchors=None) -> tuple[int, int]:
    """Every encoded triangle must carry the same corners the converter built."""
    if isinstance(windings, bool):
        windings = [windings] * len(joints)
    if two_sided is None:
        two_sided = [False] * len(joints)
    total_faces = 0
    mesh_index = 0
    bind_world = node_world_matrices(model, {}, {}, 0.0)
    # The converter pulls the vertices two joints share onto one agreed point, so
    # a checker that skips that step disagrees with a perfectly correct body.
    anchors = anchors if anchors is not None else seam_anchors(
        model, joints, scale, IDENTITY, bind_world)
    for slot, joint in enumerate(joints):
        source = build_joint_mesh(model, joint, scale, windings[slot], bind_world,
                                  two_sided[slot], anchors.get(slot))
        if not source.faces:
            # An empty joint keeps its slot in the mesh table but stores offset
            # zero, so there is nothing to decode there -- including at the root,
            # where an armature usually carries the pose and no mesh.
            mesh_index += 1
            continue
        vertices, faces, _ = decode_mesh(built.body, mesh_index)

        def canonical(corners):
            """Order-preserving key: a triangle may be rotated, never reversed.

            Comparing corner *sets* would pass a triangle whose winding was
            flipped, and the delta chain is free to rotate a face, so the key has
            to be invariant under rotation and sensitive to reversal.
            """
            rotations = [tuple(corners[(i + r) % 3] for i in range(3)) for r in range(3)]
            return min(rotations)

        expected = sorted(canonical([source.vertices[i] for i in face.indices])
                          for face in source.faces)
        actual = sorted(canonical([vertices[i] for i in triple]) for triple, _ in faces)
        if expected != actual:
            raise SystemExit(
                f"FAIL: joint {joint.name!r} decodes {len(actual)} triangles that do not "
                f"match the {len(expected)} it was built from (corners or winding differ)"
            )
        total_faces += len(faces)
        mesh_index += 1
    return mesh_index, total_faces


def model_extent(body: bytes, model_index: int = 0) -> float:
    """Largest bounding-box span across every stored keyframe."""
    offsets = struct.unpack_from("<35I", body, 32)
    _, joint_count, _, _, _ = model_record(body, model_index)
    stride = 20 + 4 * joint_count
    span = 0.0
    at = offsets[12]
    while at + stride <= offsets[13]:
        lo_x, hi_x, lo_y, hi_y, lo_z, hi_z = struct.unpack_from("<6h", body, at)
        span = max(span, hi_x - lo_x, hi_y - lo_y, hi_z - lo_z)
        at += stride
    return span or 1.0


def compare(a, b) -> tuple[float, float]:
    rotation = max(abs(a[i][j] - b[i][j]) for i in range(3) for j in range(3))
    position = max(abs(a[i][3] - b[i][3]) for i in range(3))
    return rotation, position


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help=".glb or .gltf to convert and replay")
    parser.add_argument("--glyphs", required=True, type=Path)
    # The engine stores rotations on a 1024-step grid, so a converted pose can
    # never be exact. Down a deep chain that quantisation compounds, which is why
    # the position gate is a fraction of the model rather than a fixed distance.
    parser.add_argument("--rotation-tolerance", type=float, default=0.03,
                        help="max matrix element error, default 0.03 (~5 angle steps)")
    parser.add_argument("--position-tolerance", type=float, default=1.5,
                        help="max position error as a percent of model extent, default 1.5")
    args = parser.parse_args()

    path = args.model.resolve()
    glyphs = load_glyph_strip(args.glyphs.resolve())
    spec = ModelSpec(path=path, name=sanitize_name(path.stem), slot=0)
    built = build_model(spec, glyphs, verbose=False)

    # Loaded with the budget the build settled on: a checker that prepares the
    # model differently reports failures that are its own.
    model = load_model(path, False, built.report.get("split_budget", SPLIT_BUDGET),
                       built.report.get("seam_overlap", True))
    joints = discover_joints(model)
    scale = built.report["scale"]

    meshes, faces = check_geometry(built, model, joints, built.report["scale"],
                                   built.report["flip_winding"],
                                   built.report["double_sided"])

    node_slot = {joint.node: slot for slot, joint in enumerate(joints)}
    checked = 0
    worst_rot = worst_pos = 0.0
    worst_label = "-"

    for anim_index, animation in enumerate(model.animations or [None]):
        if animation is None:
            rotation, translation, duration = {}, {}, 0.0
        else:
            rotation, translation, duration = track_channels(
                model, animation, node_slot, spec, False
            )
        keyframes = max(1, int(math.ceil(duration * TICKS_PER_SECOND / DEFAULT_FRAME_RATE)) + 1)
        for key in range(keyframes):
            time = key * DEFAULT_FRAME_RATE / TICKS_PER_SECOND
            replayed = replay(built.body, anim_index, key)
            expected = evaluate_pose(model, joints, scale, rotation, translation, time)
            for slot in range(len(joints)):
                rot, pos = compare(replayed[slot], expected[slot])
                if rot > worst_rot or pos > worst_pos:
                    worst_label = (f"{joints[slot].name} "
                                   f"(clip {anim_index}, key {key})")
                worst_rot = max(worst_rot, rot)
                worst_pos = max(worst_pos, pos)
            checked += 1

    extent = model_extent(built.body)
    budget = max(2.0, extent * args.position_tolerance / 100.0)
    relative = (worst_pos / extent * 100.0) if extent else 0.0

    print(f"{path.name}: {len(joints)} joints, {built.clips} clips, scale {scale:.4g}")
    print(f"  decoded {faces} triangles across {meshes} mesh blocks, all corners match")
    print(f"  replayed {checked} keyframes across every clip, model extent {extent:.0f} units")
    print(f"  worst rotation error {worst_rot:.5f} (tolerance {args.rotation_tolerance})")
    print(f"  worst position error {worst_pos:.2f} units = {relative:.2f}% of extent "
          f"(tolerance {args.position_tolerance}%, i.e. {budget:.1f} units)")
    print(f"  worst at: {worst_label}")

    if worst_rot > args.rotation_tolerance or worst_pos > budget:
        print("FAIL: the replayed hierarchy does not match the source", file=sys.stderr)
        return 1
    print("PASS: every replayed keyframe matches the source animation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
