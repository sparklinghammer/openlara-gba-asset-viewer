#!/usr/bin/env python3
"""Render a converted body the way the GBA does, to a PNG.

A 240x160 screenshot off an emulator is a poor place to judge whether a UV axis
is flipped or a face is culled. This replays the same pipeline offline -- the
engine's own projection, its backface test, its ordering table, its affine
texture walk and its lightmap -- so the result can be looked at at any size and
compared against the source model.

It is a diagnostic, not a renderer: no clipping, no sub-texel precision, and the
ordering table is emulated with a plain depth sort.
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_custom_pack import ModelSpec, build_model, load_glyph_strip, sanitize_name
from check_custom_body import decode_mesh, find_model, model_record, replay

FRAME_WIDTH = 240
FRAME_HEIGHT = 160
FACE_TYPE_SHIFT = 14
FACE_TEXTURE = 0x3FFF
VIEWER_SHADE = 15          # calcLightingStatic(255 << 5) with mesh intensity 0


def div_table(index: int) -> int:
    if index < 2:
        return 0x7FFF
    return min(65536 // index, 0x7FFF)


def project(x: float, y: float, z: float):
    """The engine's MODE4 PERSPECTIVE, in floating point."""
    if z < 64:
        return None
    dz = (int(z) >> 4) + (int(z) >> 6)
    if dz >= 1025:
        dz = 1024
    d = div_table(dz)
    return (x * d) / 4096.0 + FRAME_WIDTH / 2, (y * d) / 4096.0 + FRAME_HEIGHT / 2, z


def backface(a, b, c) -> bool:
    """checkBackface: cull unless the screen-space cross product is positive."""
    return (b[0] - a[0]) * (c[1] - a[1]) <= (c[0] - a[0]) * (b[1] - a[1])


def render(body: bytes, animation: int, keyframe: int,
           yaw: float, pitch: float, distance: float | None,
           model_index: int = 0, centre=None, neutral: bool = False,
           wireframe: bool = False, diagnose: bool = False):
    offsets = struct.unpack_from("<35I", body, 32)
    counts = struct.unpack_from("<14H", body, 4)
    _, joint_count, mesh_start, _, anim_index = model_record(body, model_index)

    palette = struct.unpack_from("<256H", body, offsets[0])
    lightmap = body[offsets[1]:offsets[1] + 256 * 32]
    tiles = body[offsets[2]:offsets[2] + counts[0] * 65536]
    textures = [
        struct.unpack_from("<III", body, offsets[15] + i * 12) for i in range(counts[8])
    ]

    stride = 20 + 4 * joint_count
    frame_at = offsets[12] + struct.unpack_from(
        "<I", body, offsets[7] + (anim_index + animation) * 32)[0]
    frame_at += keyframe * stride
    lo_x, hi_x, lo_y, hi_y, lo_z, hi_z = struct.unpack_from("<6h", body, frame_at)
    root = struct.unpack_from("<3h", body, frame_at + 12)

    world = replay(body, animation, keyframe, model_index, neutral)

    if centre is None:
        centre = ((lo_x + hi_x) / 2, (lo_y + hi_y) / 2, (lo_z + hi_z) / 2)
    radius = max(hi_x - lo_x, hi_y - lo_y, hi_z - lo_z) * 0.87 or 64.0
    if distance is None:
        distance = max(radius * 3.4, 320.0)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    def to_view(p):
        x, y, z = p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]
        x, z = x * cy + z * sy, -x * sy + z * cy       # Ry
        y, z = y * cp - z * sp, y * sp + z * cp        # Rx
        return x, y, z + distance

    faces = []
    edges = []
    mesh_offsets = struct.unpack_from(f"<{counts[3]}I", body, offsets[6])
    for joint in range(joint_count):
        mesh_index = mesh_start + joint
        if mesh_offsets[mesh_index] == 0:
            # Absent slot, marked by the zero sentinel. That includes the root:
            # an armature root carries the pose and often no mesh, and the
            # producer pads its mesh blob so no real block sits at offset zero.
            continue
        vertices, triangles, quads = decode_mesh(body, mesh_index)
        matrix = world[joint]
        screen = []
        for vx, vy, vz in vertices:
            wx, wy, wz = vx * 4, vy * 4, vz * 4
            px = matrix[0][0] * wx + matrix[0][1] * wy + matrix[0][2] * wz + matrix[0][3]
            py = matrix[1][0] * wx + matrix[1][1] * wy + matrix[1][2] * wz + matrix[1][3]
            pz = matrix[2][0] * wx + matrix[2][1] * wy + matrix[2][2] * wz + matrix[2][3]
            screen.append(project(*to_view((px, py, pz))))
        # A quad is rasterised as two triangles that share its UV corners; the
        # engine culls and depth-sorts it as one face, so both halves inherit the
        # quad's own test rather than being judged separately.
        for corners, flags in quads:
            pts = [screen[i] for i in corners]
            if all(p is not None for p in pts):
                # Every edge of every face: a wireframe exists to show the
                # geometry behind the surface, so nothing is culled.
                edges.extend([(pts[0], pts[1]), (pts[1], pts[2]),
                              (pts[2], pts[3]), (pts[3], pts[0])])
            if any(p is None for p in pts):
                continue
            if backface(pts[0], pts[1], pts[2]):
                continue
            depth = sum(p[2] for p in pts) / 4.0
            faces.append((depth, [pts[0], pts[1], pts[2]], flags, (0, 1, 2)))
            faces.append((depth, [pts[0], pts[2], pts[3]], flags, (0, 2, 3)))
        for triple, flags in triangles:
            pts = [screen[i] for i in triple]
            if all(p is not None for p in pts):
                edges.extend([(pts[0], pts[1]), (pts[1], pts[2]),
                              (pts[2], pts[0])])
            if any(p is None for p in pts):
                continue
            if backface(*pts):
                continue
            depth = sum(p[2] for p in pts) / 3.0
            faces.append((depth, pts, flags, (0, 1, 2)))

    faces.sort(key=lambda f: -f[0])

    from PIL import Image

    image = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0))
    pixels = image.load()
    shade = lightmap[VIEWER_SHADE * 256: VIEWER_SHADE * 256 + 256]

    def put(x: int, y: int, index: int) -> None:
        if not (0 <= x < FRAME_WIDTH and 0 <= y < FRAME_HEIGHT):
            return
        if index == 0 and diagnose:
            # A face covered this pixel but read the colour key: the hole comes
            # from the texture, not from missing geometry.
            pixels[x, y] = (255, 0, 255)
            return
        # Index 0 is NOT skipped here. Only the colour-key face type drops it,
        # and that is handled at the call site; an opaque face draws it like any
        # other index, which on this palette is black. Skipping it everywhere is
        # what let this preview show a clean face where the ROM showed a hole.
        colour = palette[shade[index]]
        pixels[x, y] = ((colour & 31) << 3, ((colour >> 5) & 31) << 3, ((colour >> 10) & 31) << 3)

    if wireframe:
        # WIRE_COLOR in the viewer: entry 12 of the glyph palette, which
        # every custom body reserves identically, so it is the one colour
        # a wireframe can count on whatever model is loaded.
        colour = palette[12]
        rgb = ((colour & 31) << 3, ((colour >> 5) & 31) << 3,
               ((colour >> 10) & 31) << 3)
        for (ax, ay, _), (bx, by, _) in edges:
            x0, y0 = int(round(ax)), int(round(ay))
            x1, y1 = int(round(bx)), int(round(by))
            dx, dy = abs(x1 - x0), abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            error = dx - dy
            for _ in range(dx + dy + 1):
                if 0 <= x0 < FRAME_WIDTH and 0 <= y0 < FRAME_HEIGHT:
                    pixels[x0, y0] = rgb
                if x0 == x1 and y0 == y1:
                    break
                doubled = error * 2
                if doubled > -dy:
                    error -= dy
                    x0 += sx
                if doubled < dx:
                    error += dx
                    y0 += sy
        return image

    for _, pts, flags, uv_slots in faces:
        kind = (flags >> FACE_TYPE_SHIFT) & 3
        (x0, y0, _), (x1, y1, _), (x2, y2, _) = pts
        min_x, max_x = int(math.floor(min(x0, x1, x2))), int(math.ceil(max(x0, x1, x2)))
        min_y, max_y = int(math.floor(min(y0, y1, y2))), int(math.ceil(max(y0, y1, y2)))
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-6:
            continue

        if kind == 1:
            index = flags & 0xFF
            corners = None
            tile = None
        else:
            tile_offset, uv01, uv23 = textures[flags & FACE_TEXTURE]
            tile = tile_offset
            all_corners = [
                ((uv01 >> 24) & 0xFF, (uv01 >> 8) & 0xFF),
                ((uv01 >> 16) & 0xFF, uv01 & 0xFF),
                ((uv23 >> 24) & 0xFF, (uv23 >> 8) & 0xFF),
                ((uv23 >> 16) & 0xFF, uv23 & 0xFF),
            ]
            corners = [all_corners[s] for s in uv_slots]
            index = None

        for py in range(max(min_y, 0), min(max_y + 1, FRAME_HEIGHT)):
            for px in range(max(min_x, 0), min(max_x + 1, FRAME_WIDTH)):
                sx, sy = px + 0.5, py + 0.5
                w0 = ((x1 - sx) * (y2 - sy) - (x2 - sx) * (y1 - sy)) / area
                w1 = ((x2 - sx) * (y0 - sy) - (x0 - sx) * (y2 - sy)) / area
                w2 = 1.0 - w0 - w1
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                if kind == 1:
                    put(px, py, index)
                else:
                    u = w0 * corners[0][0] + w1 * corners[1][0] + w2 * corners[2][0]
                    v = w0 * corners[0][1] + w1 * corners[1][1] + w2 * corners[2][1]
                    texel = tiles[tile + (int(v) & 0xFF) * 256 + (int(u) & 0xFF)]
                    if kind == 3 and texel == 0:
                        if diagnose:
                            pixels[px, py] = (255, 0, 255)
                        continue
                    put(px, py, texel)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--glyphs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--animation", type=int, default=0)
    parser.add_argument("--keyframe", type=int, default=0)
    parser.add_argument("--yaw", type=float, default=180.0,
                        help="degrees; the viewer starts at 180 so models face the camera")
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--distance", type=float, default=None)
    parser.add_argument("--zoom", type=int, default=3, help="integer upscale of the PNG")
    parser.add_argument("--wireframe", action="store_true",
                        help="edges only, as the viewer draws with the setting on")
    args = parser.parse_args()

    path = args.model.resolve()
    glyphs = load_glyph_strip(args.glyphs.resolve())
    built = build_model(ModelSpec(path=path, name=sanitize_name(path.stem), slot=0),
                        glyphs, verbose=False)
    image = render(built.body, args.animation, args.keyframe,
                   math.radians(args.yaw), math.radians(args.pitch), args.distance,
                   wireframe=args.wireframe)
    if args.zoom > 1:
        from PIL import Image
        image = image.resize((FRAME_WIDTH * args.zoom, FRAME_HEIGHT * args.zoom),
                             Image.Resampling.NEAREST)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"{args.output.name}: {image.size[0]}x{image.size[1]}, "
          f"clip {args.animation} key {args.keyframe}, yaw {args.yaw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
