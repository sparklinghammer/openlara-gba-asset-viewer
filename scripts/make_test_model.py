#!/usr/bin/env python3
"""Generate a diagnostic GLB that exercises every path the converter has.

Nothing about this model is decorative. Each choice isolates one thing that can
go wrong between a modeller and the GBA screen:

  * three flat faces in primary colours   -- palette quantisation, FACE_TYPE_F
  * three textured faces, one per axis    -- atlas packing, FACE_TYPE_FT
  * an orientation texture with a corner
    colour key and an arrow               -- UV order, winding, affine mapping
  * a child joint at an arbitrary angle   -- node positions, 10-bit angle packing
  * a rotation track on both joints       -- resampling and frame layout

If the cube renders with the arrow upright, the corners in the right places and
the child box orbiting, the whole chain is sound.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from io import BytesIO
from pathlib import Path

TEXTURE_SIZE = 64


def make_texture() -> bytes:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (TEXTURE_SIZE, TEXTURE_SIZE), (24, 24, 32))
    draw = ImageDraw.Draw(image)
    half = TEXTURE_SIZE // 2
    # One colour per corner: any mirror or 90-degree error is unmistakable.
    draw.rectangle([0, 0, half - 1, half - 1], fill=(214, 68, 52))          # top-left
    draw.rectangle([half, 0, TEXTURE_SIZE - 1, half - 1], fill=(238, 197, 62))
    draw.rectangle([0, half, half - 1, TEXTURE_SIZE - 1], fill=(58, 148, 96))
    draw.rectangle([half, half, TEXTURE_SIZE - 1, TEXTURE_SIZE - 1], fill=(64, 104, 208))
    # An arrow pointing at the top edge: shows which way is up on screen.
    draw.polygon([(half, 8), (half - 13, 30), (half - 5, 30),
                  (half - 5, 54), (half + 5, 54), (half + 5, 30), (half + 13, 30)],
                 fill=(245, 245, 245), outline=(16, 16, 16))
    draw.rectangle([0, 0, TEXTURE_SIZE - 1, TEXTURE_SIZE - 1], outline=(16, 16, 16))

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def box(size: float):
    """A unit box as 6 quads, each with its own 4 vertices and full-face UVs."""
    h = size / 2.0
    faces = [
        # (normal axis, corner order), counter-clockwise seen from outside
        [(-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)],        # +Z front
        [(h, -h, -h), (-h, -h, -h), (-h, h, -h), (h, h, -h)],    # -Z back
        [(h, -h, h), (h, -h, -h), (h, h, -h), (h, h, h)],        # +X right
        [(-h, -h, -h), (-h, -h, h), (-h, h, h), (-h, h, -h)],    # -X left
        [(-h, h, h), (h, h, h), (h, h, -h), (-h, h, -h)],        # +Y top
        [(-h, -h, -h), (h, -h, -h), (h, -h, h), (-h, -h, h)],    # -Y bottom
    ]
    # v grows downward in glTF UV space, so the arrow's tip sits at v = 0.
    uv = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    return faces, uv


class GlbBuilder:
    def __init__(self):
        self.blob = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def _view(self, data: bytes, target: int | None = None) -> int:
        while len(self.blob) % 4:
            self.blob.append(0)
        view = {"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(data)}
        if target:
            view["target"] = target
        self.blob.extend(data)
        self.views.append(view)
        return len(self.views) - 1

    def floats(self, values: list[tuple], kind: str) -> int:
        arity = {"VEC2": 2, "VEC3": 3, "VEC4": 4, "SCALAR": 1, "MAT4": 16}[kind]
        flat = [c for row in values for c in row]
        view = self._view(struct.pack(f"<{len(flat)}f", *flat), 34962)
        accessor = {
            "bufferView": view, "componentType": 5126,
            "count": len(values), "type": kind,
        }
        if kind == "VEC3":
            accessor["min"] = [min(r[i] for r in values) for i in range(3)]
            accessor["max"] = [max(r[i] for r in values) for i in range(3)]
        if kind == "SCALAR":
            accessor["min"] = [min(r[0] for r in values)]
            accessor["max"] = [max(r[0] for r in values)]
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def ubytes(self, values: list[tuple]) -> int:
        """VEC4 of unsigned bytes, which is what glTF wants for JOINTS_0."""
        flat = [c for row in values for c in row]
        view = self._view(struct.pack(f"<{len(flat)}B", *flat), 34962)
        self.accessors.append({
            "bufferView": view, "componentType": 5121,
            "count": len(values), "type": "VEC4",
        })
        return len(self.accessors) - 1

    def indices(self, values: list[int]) -> int:
        view = self._view(struct.pack(f"<{len(values)}H", *values), 34963)
        self.accessors.append({
            "bufferView": view, "componentType": 5123,
            "count": len(values), "type": "SCALAR",
        })
        return len(self.accessors) - 1

    def raw(self, data: bytes) -> int:
        return self._view(data)

    def write(self, doc: dict, path: Path) -> None:
        doc["buffers"] = [{"byteLength": len(self.blob)}]
        doc["bufferViews"] = self.views
        doc["accessors"] = self.accessors
        json_chunk = json.dumps(doc, separators=(",", ":")).encode("utf-8")
        json_chunk += b" " * ((-len(json_chunk)) % 4)
        bin_chunk = bytes(self.blob) + b"\0" * ((-len(self.blob)) % 4)
        total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
        with path.open("wb") as handle:
            handle.write(struct.pack("<4sII", b"glTF", 2, total))
            handle.write(struct.pack("<I4s", len(json_chunk), b"JSON"))
            handle.write(json_chunk)
            handle.write(struct.pack("<I4s", len(bin_chunk), b"BIN\0"))
            handle.write(bin_chunk)


def build(path: Path, size: float) -> None:
    builder = GlbBuilder()
    faces, uv = box(size)

    # Three faces flat, three textured, so both material paths appear at once.
    flat_faces = [2, 3, 5]          # +X, -X, -Y
    textured_faces = [0, 1, 4]      # +Z, -Z, +Y

    primitives = []
    for material, group in ((0, textured_faces), (1, flat_faces[:1]),
                            (2, flat_faces[1:2]), (3, flat_faces[2:])):
        positions: list[tuple] = []
        uvs: list[tuple] = []
        index: list[int] = []
        for face in group:
            base = len(positions)
            positions.extend(faces[face])
            uvs.extend(uv)
            index.extend([base, base + 1, base + 2, base, base + 2, base + 3])
        attributes = {"POSITION": builder.floats(positions, "VEC3")}
        if material == 0:
            attributes["TEXCOORD_0"] = builder.floats(uvs, "VEC2")
        primitives.append({
            "attributes": attributes,
            "indices": builder.indices(index),
            "material": material,
        })

    child_faces, child_uv = box(size * 0.4)
    child_positions: list[tuple] = []
    child_index: list[int] = []
    for face in child_faces:
        base = len(child_positions)
        child_positions.extend(face)
        child_index.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    child_primitive = {
        "attributes": {"POSITION": builder.floats(child_positions, "VEC3")},
        "indices": builder.indices(child_index),
        "material": 4,
    }

    # Angles chosen off the 10-bit grid on purpose: they force the packer to
    # round, so the replay check measures real quantisation error.
    tilt = math.radians(37.0)
    child_rest = (0.0, math.sin(tilt / 2), 0.0, math.cos(tilt / 2))

    seconds = 2.0
    keys = 17
    times = [(i * seconds / (keys - 1),) for i in range(keys)]
    spin = []
    wobble = []
    for i in range(keys):
        t = i / (keys - 1)
        a = 2 * math.pi * t
        spin.append((0.0, math.sin(a / 2), 0.0, math.cos(a / 2)))
        b = math.radians(23.0) * math.sin(2 * math.pi * t)
        wobble.append((math.sin(b / 2), 0.0, 0.0, math.cos(b / 2)))

    time_accessor = builder.floats(times, "SCALAR")
    spin_accessor = builder.floats(spin, "VEC4")
    wobble_accessor = builder.floats(wobble, "VEC4")
    image_view = builder.raw(make_texture())

    doc = {
        "asset": {"version": "2.0", "generator": "make_test_model.py"},
        "scene": 0,
        "scenes": [{"nodes": [2]}],
        # Authored Y-up, the way Blender exports. The models extracted from TR1
        # carry an extra 180-degree node to lift their Y-down data into glTF
        # orientation; adding one here would flip this model upside down.
        "nodes": [
            {"name": "cube_bone0", "mesh": 0, "children": [1],
             "translation": [0.0, 0.0, 0.0]},
            {"name": "cube_bone1", "mesh": 1,
             "translation": [size * 0.9, size * 0.75, 0.0],
             "rotation": list(child_rest)},
            {"name": "cube", "children": [0]},
        ],
        "meshes": [
            {"name": "cube_body", "primitives": primitives},
            {"name": "cube_child", "primitives": [child_primitive]},
        ],
        "materials": [
            {"name": "OL_TEX_faces", "doubleSided": False,
             "pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
                                      "metallicFactor": 0.0, "roughnessFactor": 1.0}},
            {"name": "OL_FLAT_red", "doubleSided": False,
             "pbrMetallicRoughness": {"baseColorFactor": [0.85, 0.20, 0.16, 1.0],
                                      "metallicFactor": 0.0, "roughnessFactor": 1.0}},
            {"name": "OL_FLAT_green", "doubleSided": False,
             "pbrMetallicRoughness": {"baseColorFactor": [0.20, 0.62, 0.36, 1.0],
                                      "metallicFactor": 0.0, "roughnessFactor": 1.0}},
            {"name": "OL_FLAT_blue", "doubleSided": False,
             "pbrMetallicRoughness": {"baseColorFactor": [0.22, 0.38, 0.82, 1.0],
                                      "metallicFactor": 0.0, "roughnessFactor": 1.0}},
            {"name": "OL_FLAT_amber", "doubleSided": False,
             "pbrMetallicRoughness": {"baseColorFactor": [0.94, 0.72, 0.20, 1.0],
                                      "metallicFactor": 0.0, "roughnessFactor": 1.0}},
        ],
        "images": [{"name": "orientation", "bufferView": image_view, "mimeType": "image/png"}],
        "samplers": [{"magFilter": 9728, "minFilter": 9728}],
        "textures": [{"sampler": 0, "source": 0}],
        "animations": [
            {"name": "spin",
             "samplers": [
                 {"input": time_accessor, "output": spin_accessor, "interpolation": "LINEAR"},
                 {"input": time_accessor, "output": wobble_accessor, "interpolation": "LINEAR"},
             ],
             "channels": [
                 {"sampler": 0, "target": {"node": 1, "path": "rotation"}},
                 {"sampler": 1, "target": {"node": 0, "path": "rotation"}},
             ]},
            {"name": "rest",
             "samplers": [
                 {"input": builder.floats([(0.0,)], "SCALAR"),
                  "output": builder.floats([child_rest], "VEC4"),
                  "interpolation": "LINEAR"},
             ],
             "channels": [{"sampler": 0, "target": {"node": 1, "path": "rotation"}}]},
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    builder.write(doc, path)


def build_skinned(path: Path, size: float) -> None:
    """A two-bone bar that bends, with a weight ramp across its middle.

    Nothing else in a normal pack exercises weighted skinning: models ripped
    from DS or GBA era games name exactly one bone per vertex, because the
    hardware of the day could not blend either. So the case needs a model of its
    own, and this is it.

    The ramp is what makes the test worth running. At the bind pose a blend and
    its heaviest bone agree exactly -- every bone maps the vertex to the same
    place -- so a converter that quietly drops the lighter influences looks
    perfect until the joint moves. Bent, the difference is a smooth curve
    against a hard kink.
    """
    builder = GlbBuilder()

    rings = 8
    height = size * 2.0
    half = size * 0.22

    # A square section swept up the Y axis. Corners are kept per ring so the
    # weights can change along the bar.
    section = [(-half, -half), (half, -half), (half, half), (-half, half)]
    positions: list[tuple] = []
    joints: list[tuple] = []
    weights: list[tuple] = []
    for ring in range(rings):
        y = height * ring / (rings - 1)
        # Pure bone 0 low down, pure bone 1 high up, and three rings of ramp
        # between them. Two of those land on neither bone outright.
        upper = min(1.0, max(0.0, (ring - 2) / 3.0))
        for x, z in section:
            positions.append((x, y, z))
            joints.append((0, 1, 0, 0))
            weights.append((1.0 - upper, upper, 0.0, 0.0))

    index: list[int] = []
    for ring in range(rings - 1):
        for corner in range(4):
            a = ring * 4 + corner
            b = ring * 4 + (corner + 1) % 4
            index.extend([a, b, b + 4, a, b + 4, a + 4])
    for corner in range(1, 3):                       # both caps, as fans
        index.extend([0, corner + 1, corner])
        top = (rings - 1) * 4
        index.extend([top, top + corner, top + corner + 1])

    primitive = {
        "attributes": {
            "POSITION": builder.floats(positions, "VEC3"),
            "JOINTS_0": builder.ubytes(joints),
            "WEIGHTS_0": builder.floats(weights, "VEC4"),
        },
        "indices": builder.indices(index),
        "material": 0,
    }

    # Column-major, as glTF stores them: bone 0 sits at the origin, bone 1 half
    # way up, so its inverse bind lowers the bar back onto it.
    identity = (1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0)
    lowered = (1.0, 0.0, 0.0, 0.0,
               0.0, 1.0, 0.0, 0.0,
               0.0, 0.0, 1.0, 0.0,
               0.0, -height * 0.5, 0.0, 1.0)

    seconds = 2.0
    keys = 17
    times = [(i * seconds / (keys - 1),) for i in range(keys)]
    bend = []
    for i in range(keys):
        angle = math.radians(70.0) * math.sin(2 * math.pi * i / (keys - 1))
        bend.append((math.sin(angle / 2), 0.0, 0.0, math.cos(angle / 2)))

    doc = {
        "asset": {"version": "2.0", "generator": "make_test_model.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "bar_root", "children": [1, 3]},
            {"name": "bar_bone0", "children": [2], "translation": [0.0, 0.0, 0.0]},
            {"name": "bar_bone1", "translation": [0.0, height * 0.5, 0.0]},
            {"name": "bar", "mesh": 0, "skin": 0},
        ],
        "meshes": [{"name": "bar_mesh", "primitives": [primitive]}],
        "skins": [{
            "name": "bar_skin",
            "joints": [1, 2],
            "inverseBindMatrices": builder.floats([identity, lowered], "MAT4"),
        }],
        "materials": [
            {"name": "OL_FLAT_amber", "doubleSided": False,
             "pbrMetallicRoughness": {"baseColorFactor": [0.94, 0.72, 0.20, 1.0],
                                      "metallicFactor": 0.0, "roughnessFactor": 1.0}},
        ],
        "animations": [
            {"name": "bend",
             "samplers": [{"input": builder.floats(times, "SCALAR"),
                           "output": builder.floats(bend, "VEC4"),
                           "interpolation": "LINEAR"}],
             "channels": [{"sampler": 0, "target": {"node": 2, "path": "rotation"}}]},
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    builder.write(doc, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", type=float, default=512.0,
                        help="edge length in engine units (default 512, half a sector)")
    parser.add_argument("--skinned", action="store_true",
                        help="a two-bone bar with a weight ramp, for the blended "
                             "skinning path that no ripped model exercises")
    args = parser.parse_args()
    if args.skinned:
        build_skinned(args.output.resolve(), args.size)
    else:
        build(args.output.resolve(), args.size)
    print(f"{args.output.name}: {args.output.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
