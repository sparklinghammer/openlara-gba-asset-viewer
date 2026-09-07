#!/usr/bin/env python3
"""Compare two source files that are meant to describe the same model.

Models are often published in several formats at once -- a .dae beside a .obj,
a .glb beside both -- and they are supposed to agree. When they do, converting
either gives the same ROM and the pair is a free check on the readers: two
independent parse paths reaching the same geometry is hard to arrange by
accident. When they disagree, the difference is worth knowing before it turns
into a model that looks subtly wrong on hardware.

What is compared, in order of how much it matters:

  * the world-space geometry each file yields, as a point cloud, after every
    convention the reader applies -- up axis, V flip, bind pose. This is the
    one that matters, because it is what reaches the screen;
  * the triangle and material counts, which catch a file exported at a
    different level of detail;
  * the UV ranges and the material-to-image mapping, which catch a flipped or
    rescaled texture;
  * what each format can carry that the other cannot, which is usually why a
    pair exists in the first place.

Usage:
    python compare_sources.py sora.dae sora.obj
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from build_custom_pack import (
    BuildError, IDENTITY, discover_joints, evaluate_pose, load_model,
    local_points, rotation_matrix,
)
import gltf_reader


def world_cloud(path: Path, rotate=(0.0, 0.0, 0.0)) -> tuple[list, list]:
    """Every vertex the converter would emit, in engine world space.

    Taken through the same path the builder uses -- joint discovery, the rest
    pose, the flip into engine space -- at unit scale, so two files can be
    compared on shape rather than on how each happens to be measured.
    """
    model = load_model(path)
    joints = discover_joints(model)
    pre = rotation_matrix(rotate)
    bind_world = model_world(model, pre)
    rest = evaluate_pose(model, joints, 1.0, {}, {}, 0.0, pre)

    points: list[tuple[float, float, float]] = []
    for slot, joint in enumerate(joints):
        matrix = rest[slot]
        for primitive in local_points(model, joint, bind_world):
            for point in primitive:
                x, y, z = point[0], -point[1], -point[2]     # FLIP into engine space
                points.append(tuple(
                    matrix[axis][0] * x + matrix[axis][1] * y
                    + matrix[axis][2] * z + matrix[axis][3]
                    for axis in range(3)))
    return points, [model, joints]


def model_world(model, pre):
    from build_custom_pack import node_world_matrices
    return node_world_matrices(model, {}, {}, 0.0, pre)


def normalise(points: list) -> tuple[list, float]:
    """Centre a cloud on its bounding box and scale its longest axis to 1.

    Two files may state the same shape in different units -- metres against
    centimetres -- which the converter's auto-scale absorbs anyway. Removing
    that here leaves only differences of shape.
    """
    lo = [min(p[axis] for p in points) for axis in range(3)]
    hi = [max(p[axis] for p in points) for axis in range(3)]
    extent = max(hi[axis] - lo[axis] for axis in range(3)) or 1.0
    centre = [(hi[axis] + lo[axis]) / 2.0 for axis in range(3)]
    return [tuple((p[axis] - centre[axis]) / extent for axis in range(3))
            for p in points], extent


def nearest_distances(source: list, target: list, cell: float) -> list[float]:
    """For each point of `source`, the distance to the closest point of `target`."""
    grid: dict[tuple[int, int, int], list] = {}
    for point in target:
        key = tuple(int(math.floor(point[axis] / cell)) for axis in range(3))
        grid.setdefault(key, []).append(point)

    out = []
    for point in source:
        base = tuple(int(math.floor(point[axis] / cell)) for axis in range(3))
        best = float("inf")
        radius = 0
        while True:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        if radius and max(abs(dx), abs(dy), abs(dz)) != radius:
                            continue
                        for other in grid.get((base[0] + dx, base[1] + dy,
                                               base[2] + dz), ()):
                            d = math.sqrt(sum((point[a] - other[a]) ** 2
                                              for a in range(3)))
                            best = min(best, d)
            if best <= radius * cell or radius > 6:
                break
            radius += 1
        out.append(best)
    return out


def describe(path: Path):
    """What one file carries, in the terms the converter cares about."""
    import collada_reader, fbx_reader, obj_reader
    readers = {".glb": gltf_reader.load, ".gltf": gltf_reader.load,
               ".dae": collada_reader.load, ".obj": obj_reader.load,
               ".fbx": fbx_reader.load}
    suffix = path.suffix.lower()
    if suffix not in readers:
        raise ValueError(f"{path.name}: unsupported format")
    model = readers[suffix](path)
    triangles = sum(len(p.indices) // 3 for mesh in model.meshes for p in mesh)
    uvs = [uv for mesh in model.meshes for p in mesh if p.uvs for uv in p.uvs]
    textured = {m.name: (model.images[m.base_color_texture].name
                         if m.base_color_texture is not None else None)
                for m in model.materials}
    bones = sum(len(skin.joints) for skin in model.skins)
    clips = [a.name for a in model.animations]
    return {
        "clips": len(clips),
        "triangles": triangles,
        "materials": len(model.materials),
        "images": len(model.images),
        "mesh nodes": sum(1 for n in model.nodes if n.mesh is not None),
        "bones": bones,
        "animations": len(model.animations),
        "u range": (min((u for u, _ in uvs), default=0.0),
                    max((u for u, _ in uvs), default=0.0)),
        "v range": (min((v for _, v in uvs), default=0.0),
                    max((v for _, v in uvs), default=0.0)),
        "textures": textured,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--rotate", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=("X", "Y", "Z"),
                        help="degrees applied to BOTH files before comparing")
    parser.add_argument("--tolerance", type=float, default=0.5,
                        help="percent of the model's size a point may differ by")
    args = parser.parse_args()

    try:
        left = describe(args.first)
        right = describe(args.second)
    except Exception as error:                      # a reader refusing is the answer
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    print(f"{args.first.name}  vs  {args.second.name}")
    width = max(len(key) for key in left)
    disagree = []
    for key in left:
        if key == "textures":
            continue
        a, b = left[key], right[key]
        if isinstance(a, tuple):
            a = f"{a[0]:.4f}..{a[1]:.4f}"
            b = f"{b[0]:.4f}..{b[1]:.4f}"
        mark = "  " if a == b else "  <-- differ"
        if a != b and key not in ("mesh nodes", "bones", "animations"):
            disagree.append(key)
        print(f"  {key:<{width}}  {str(a):>18}   {str(b):>18}{mark}")

    # Materials and images are named differently by every exporter -- COLLADA
    # writes its own ids, FBX the file stem -- so the names are normalised down
    # to the image each material actually ends up drawing, which is the part
    # that has to match.
    def image_key(name: str) -> str:
        name = name.lower()
        for prefix in ("image-", "image_", "video::", "texture::"):
            if name.startswith(prefix):
                name = name[len(prefix):]
        for suffix in ("_png", "-png", ".png"):
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        return name

    images_left = sorted(image_key(v) for v in left["textures"].values() if v)
    images_right = sorted(image_key(v) for v in right["textures"].values() if v)
    if images_left != images_right:
        disagree.append("textures")
        print(f"  textures    {images_left}\n           vs {images_right}   <-- differ")

    try:
        first_points, _ = world_cloud(args.first, args.rotate)
        second_points, _ = world_cloud(args.second, args.rotate)
    except BuildError as error:
        print(f"\ngeometry not comparable: {error}", file=sys.stderr)
        return 1

    first_norm, first_extent = normalise(first_points)
    second_norm, second_extent = normalise(second_points)
    cell = 0.02
    forward = nearest_distances(first_norm, second_norm, cell)
    backward = nearest_distances(second_norm, first_norm, cell)
    worst = max(max(forward), max(backward))
    mean = (sum(forward) + sum(backward)) / (len(forward) + len(backward))
    tolerance = args.tolerance / 100.0

    print(f"\ngeometry: {len(first_points)} vs {len(second_points)} vertices, "
          f"size {first_extent:.4f} vs {second_extent:.4f} "
          f"({abs(first_extent - second_extent) / max(first_extent, 1e-9) * 100:.2f}% apart)")
    print(f"  every point of each file lies within {worst * 100:.3f}% of the other's "
          f"surface (mean {mean * 100:.3f}%, tolerance {args.tolerance}%)")

    if worst > tolerance:
        print("\nDIFFER: the two files do not describe the same shape")
        return 1
    if disagree:
        print(f"\nDIFFER: same shape, but {', '.join(disagree)} do not match")
        return 1
    print("\nSAME: both files describe the same model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
