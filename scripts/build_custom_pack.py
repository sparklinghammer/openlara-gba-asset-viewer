#!/usr/bin/env python3
"""Build a viewer asset pack from a folder of custom glTF/GLB models.

One PKD body per model, so every model owns its 256-colour palette, its own
lightmap and its own texture pages. Nothing is shared between models except the
glyph strip, which each body carries a remapped copy of.

The GBA target only renders three material kinds, so this converter only emits
three: flat palette colour, textured opaque, textured with colour-key holes.
Everything it cannot represent is reported as an error instead of being dropped.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import collada_reader
import fbx_reader
import gltf_reader
import obj_reader
from pkd_writer import (
    GLYPH_COUNT, GLYPHS_TYPE, LIGHTMAP_SIZE, PALETTE_SIZE, TILE_SIZE,
    PkdParts, body_stats, write_pkd_body,
)

PACK_MAGIC = b"AVP1"
PACK_VERSION = 4
PACK_HEADER_SIZE = 32       # v3 header; v4 appends the skin table offset
PACK_HEADER_SIZE_V4 = 36
SOURCE_ENTRY_SIZE = 16
MODEL_ENTRY_SIZE = 12
NAME_ENTRY_SIZE = 24
SKIN_ENTRY_SIZE = 16        # bone table, then the blend records
PACK_FLAG_NAMES = 1 << 1
# One byte per vertex naming the joint it follows. The engine's own mesh blocks
# are untouched by it; the viewer reads it to pose a model a vertex at a time,
# which is the only way a triangle whose corners sit on different bones can
# stretch between them instead of being frozen onto one of them.
PACK_FLAG_SKIN = 1 << 2

FACE_TYPE_F = 1
FACE_TYPE_FT = 2
FACE_TYPE_FTA = 3
FACE_TYPE_SHIFT = 14

MAX_JOINTS = 32              # the catalog carries a 32-bit visibility mask
MAX_VERTS_PER_MESH = 255     # vCount is a uint8; deltas are checked at encode time
SAFE_VERTS_PER_MESH = 128    # below this, int8 index deltas cannot overflow
SPLIT_BUDGET = MAX_VERTS_PER_MESH   # past this a mesh cannot fit one block at all
MAX_MATRIX_DEPTH = 6         # MAX_MATRICES is 8; the viewer already pushed one
MAX_UV_SPAN = 127            # fixTexCoord clamps beyond this, silently
MAX_FACES_PER_FRAME = 1920   # the renderer drops faces past this, without a word
TILE_DIM = 256
GLYPH_REGION_H = 128         # the TR1 glyph strip occupies rows 0..127 of page 0
TICKS_PER_SECOND = 30
DEFAULT_FRAME_RATE = 2       # ticks between keyframes
AUTO_SCALE_TARGET = 1024     # one TR sector, a sane on-screen size
AUTO_SCALE_BELOW = 64        # below this the >> 2 vertex store would lose the shape
WINDING_MIN_CONFIDENCE = 1e-3  # enclosed volume over bounding box, below which a
                               # mesh is a flat sheet and its winding unknowable

NODE_FLAG_POP = 1 << 0
NODE_FLAG_PUSH = 1 << 1


class BuildError(Exception):
    pass


class ChainOverflow(BuildError):
    """A mesh block whose face chain reaches further back than int8 allows."""


class FaceBudgetExceeded(BuildError):
    """More faces than the renderer draws in one frame."""


MODEL_SUFFIXES = (".glb", ".gltf", ".dae", ".obj", ".fbx")


def load_model(path: Path, verbose: bool = False,
               split_budget: int = SPLIT_BUDGET,
               skirt: bool = True) -> gltf_reader.Gltf:
    """Read a model in whichever format it arrives in, ready for conversion.

    Every reader returns the same structures, so nothing downstream needs to
    know which one ran.

    Binding a skinned mesh to its skeleton happens here rather than in the
    caller so that every consumer -- the builder, the checker, any future tool
    -- sees the same geometry. A checker that rebuilds its reference from a
    differently prepared model reports failures that are its own.
    """
    suffix = Path(path).suffix.lower()
    if suffix in (".glb", ".gltf"):
        model = gltf_reader.load(path)
    elif suffix == ".dae":
        model = collada_reader.load(path)
    elif suffix == ".obj":
        model = obj_reader.load(path)
    elif suffix == ".fbx":
        model = fbx_reader.load(path)
    else:
        raise BuildError(f"{Path(path).name}: unsupported format; expected one of "
                         + ", ".join(MODEL_SUFFIXES))
    # Before anything cuts the mesh up: adjacency is what the repair walks, and
    # binding to a skeleton turns one surface into a pile of open pieces.
    repair_winding(model, verbose)
    bake_static_skins(model, verbose)
    bind_to_skeleton(model, verbose, skirt)
    merge_excess_joints(model, verbose)
    split_oversized_meshes(model, verbose, split_budget)
    return model


# --------------------------------------------------------------------------
# colour
# --------------------------------------------------------------------------

def snap5(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Reduce to the 5 bits per channel the GBA palette actually stores."""
    return tuple((min(max(int(c), 0), 255) >> 3) << 3 for c in rgb)


def to_bgr555(rgb: tuple[int, int, int]) -> int:
    r, g, b = (min(max(int(c), 0), 255) >> 3 for c in rgb)
    return r | (g << 5) | (b << 10)


def from_bgr555(value: int) -> tuple[int, int, int]:
    return ((value & 31) << 3, ((value >> 5) & 31) << 3, ((value >> 10) & 31) << 3)


def nearest_index(palette: list[tuple[int, int, int]], rgb: tuple[int, int, int],
                  skip_zero: bool = True) -> int:
    best = 1 if skip_zero else 0
    best_d = None
    start = 1 if skip_zero else 0
    r, g, b = rgb
    for index in range(start, len(palette)):
        pr, pg, pb = palette[index]
        d = (pr - r) ** 2 + (pg - g) ** 2 + (pb - b) ** 2
        if best_d is None or d < best_d:
            best_d, best = d, index
            if d == 0:
                break
    return best


# --------------------------------------------------------------------------
# glyph strip, borrowed from a TR1 body
# --------------------------------------------------------------------------

@dataclass
class GlyphStrip:
    """The 110 viewer glyphs: their pixels, their colours and their records."""

    pixels: list[list[int]]                 # GLYPH_REGION_H rows of TILE_DIM indices
    colors: list[tuple[int, int, int]]      # source colours, index 0 excluded
    index_map: dict[int, int]               # source palette index -> slot in `colors`
    records: list[bytes]                    # 110 sprite records, l/t/r/b pre-halved


def load_glyph_strip(pkd_path: Path) -> GlyphStrip:
    data = pkd_path.read_bytes()
    if data[:4] != b"GBA ":
        raise BuildError(f"{pkd_path}: not a PKD")
    counts = struct.unpack_from("<14H", data, 4)
    offsets = struct.unpack_from("<35I", data, 32)

    start = None
    for index in range(counts[5]):
        kind, _, count, first = struct.unpack_from("<HHhH", data, offsets[17] + index * 8)
        if kind == GLYPHS_TYPE and abs(count) >= GLYPH_COUNT:
            start = first
            break
    if start is None:
        raise BuildError(f"{pkd_path}: no GLYPHS sprite sequence")

    raw_records = [
        data[offsets[16] + (start + i) * 16: offsets[16] + (start + i + 1) * 16]
        for i in range(GLYPH_COUNT)
    ]

    tiles = {(struct.unpack_from("<I", rec, 0)[0] >> 16) & 0x3FFF for rec in raw_records}
    if len(tiles) != 1:
        raise BuildError(f"{pkd_path}: glyphs span {len(tiles)} tiles, expected one")
    tile_index = tiles.pop()
    page = data[offsets[2] + tile_index * TILE_SIZE: offsets[2] + (tile_index + 1) * TILE_SIZE]

    bottom = 0
    for rec in raw_records:
        uwvh = struct.unpack_from("<I", rec, 4)[0]
        bottom = max(bottom, ((uwvh >> 8) & 0xFF) + (uwvh & 0xFF))
    if bottom > GLYPH_REGION_H:
        raise BuildError(f"{pkd_path}: glyph strip is {bottom} rows, more than {GLYPH_REGION_H}")

    pixels = [list(page[y * TILE_DIM:(y + 1) * TILE_DIM]) for y in range(GLYPH_REGION_H)]

    source_palette = struct.unpack_from("<256H", data, offsets[0])
    used: set[int] = set()
    for row in pixels:
        used.update(row)
    used.discard(0)
    ordered = sorted(used)
    colors = [snap5(from_bgr555(source_palette[i])) for i in ordered]
    index_map = {source: slot for slot, source in enumerate(ordered)}

    # PACK_FLAG_TR1 is off for custom packs, so scaleGlyphs() will not run at
    # load time. Bake the same 50% reduction the viewer's advance table assumes.
    records = []
    for rec in raw_records:
        tile, uwvh = struct.unpack_from("<II", rec, 0)
        l, t, r, b = struct.unpack_from("<4h", rec, 8)
        records.append(struct.pack("<II4h", 0, uwvh, l // 2, t // 2, r // 2, b // 2))
    return GlyphStrip(pixels, colors, index_map, records)


# --------------------------------------------------------------------------
# texture atlas
# --------------------------------------------------------------------------

@dataclass
class Region:
    page: int
    x: int
    y: int
    w: int
    h: int


@dataclass
class Placement:
    """Where an image sits in the atlas, and which UV range it covers.

    A UV outside 0..1 means the artist tiled the texture. The engine samples a
    whole 256-wide page and wraps there, not inside a packed sub-image, so a
    repeat cannot be expressed at draw time -- it has to be baked. The image is
    therefore stamped as many times as the mesh actually uses, and `u0`/`v0`
    record which repeat the region starts at.
    """

    region: Region
    u0: int
    v0: int
    width: int      # one tile, in texels
    height: int


class Atlas:
    """Shelf packer over 256x256 pages, one texel of guard between regions."""

    def __init__(self, reserved_rows: int):
        self.pages: list[list[list[int]]] = [
            [[0] * TILE_DIM for _ in range(TILE_DIM)]
        ]
        self.shelf_y = reserved_rows
        self.shelf_x = 0
        self.shelf_h = 0
        self.page = 0

    def _new_page(self) -> None:
        self.pages.append([[0] * TILE_DIM for _ in range(TILE_DIM)])
        self.page += 1
        self.shelf_x = 0
        self.shelf_y = 0
        self.shelf_h = 0

    def place(self, width: int, height: int) -> Region:
        if width > TILE_DIM or height > TILE_DIM:
            raise BuildError(
                f"texture {width}x{height} does not fit a {TILE_DIM}x{TILE_DIM} page"
            )
        if self.shelf_x + width > TILE_DIM:
            self.shelf_y += self.shelf_h + 1
            self.shelf_x = 0
            self.shelf_h = 0
        if self.shelf_y + height > TILE_DIM:
            self._new_page()
        region = Region(self.page, self.shelf_x, self.shelf_y, width, height)
        self.shelf_x += width + 1
        self.shelf_h = max(self.shelf_h, height)
        return region

    def blit(self, region: Region, rows: list[list[int]]) -> None:
        page = self.pages[region.page]
        for y in range(region.h):
            page[region.y + y][region.x:region.x + region.w] = rows[y]

    def to_bytes(self) -> list[bytes]:
        return [bytes(b for row in page for b in row) for page in self.pages]


# --------------------------------------------------------------------------
# lightmap
# --------------------------------------------------------------------------

def build_lightmap(palette: list[tuple[int, int, int]]) -> bytes:
    """32 shade rows: row 0 brightest, row 16 identity, row 31 black.

    Matches the curve measured on TR1 data: f(s) = (32 - s) / 16, so shade 16
    reproduces the palette unchanged and shade 31 collapses to darkness.
    """
    out = bytearray(LIGHTMAP_SIZE)
    cache: dict[tuple[int, int, int], int] = {}
    for shade in range(32):
        factor = (32 - shade) / 16.0
        base = shade * 256
        out[base] = 0  # index 0 stays the colour key at every shade
        for index in range(1, 256):
            r, g, b = palette[index]
            key = snap5((r * factor, g * factor, b * factor))
            hit = cache.get(key)
            if hit is None:
                hit = nearest_index(palette, key)
                cache[key] = hit
            out[base + index] = hit
    return bytes(out)


# --------------------------------------------------------------------------
# small 4x4 matrix helpers, right-handed, row-major
# --------------------------------------------------------------------------

Mat = list[list[float]]

IDENTITY: Mat = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]
# 180 degrees about X: glTF is Y-up, the engine is Y-down.
FLIP: Mat = [[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 0], [0, 0, 0, 1.0]]


def mat_mul(a: Mat, b: Mat) -> Mat:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def mat_from_trs(t, r, s) -> Mat:
    x, y, z, w = r
    xx, yy, zz = x * x, y * y, z * z
    m = [
        [1 - 2 * (yy + zz), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0],
        [2 * (x * y + z * w), 1 - 2 * (xx + zz), 2 * (y * z - x * w), 0.0],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (xx + yy), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    for i in range(3):
        for j in range(3):
            m[i][j] *= s[j]
    m[0][3], m[1][3], m[2][3] = t
    return m


def conjugate(m: Mat) -> Mat:
    """FLIP * m * FLIP, the change of basis into engine space (FLIP is its own inverse)."""
    return mat_mul(mat_mul(FLIP, m), FLIP)


def scaled_translation(m: Mat, factor: float) -> Mat:
    out = [row[:] for row in m]
    for i in range(3):
        out[i][3] *= factor
    return out


def decompose(m: Mat, label: str) -> tuple[tuple[float, float, float], tuple[int, int, int]]:
    """Split a rigid transform into a translation and TR's packed Y-X-Z angles.

    The engine applies matrixRotateYXZ as M = M * Ry * Rx * Rz, all three the
    standard right-handed matrices, so the decomposition is the classic one for
    R = Ry(b) * Rx(a) * Rz(c).
    """
    for i in range(3):
        length = math.sqrt(sum(m[i][j] ** 2 for j in range(3)))
        if abs(length - 1.0) > 0.02:
            raise BuildError(
                f"{label}: transform is scaled or sheared (row {i} length {length:.4f}); "
                "apply scale in the modeller before exporting"
            )

    sin_x = -m[1][2]
    sin_x = min(max(sin_x, -1.0), 1.0)
    angle_x = math.asin(sin_x)
    if abs(sin_x) > 0.99999:                    # gimbal lock: fold Z into Y
        angle_z = 0.0
        angle_y = math.atan2(-m[2][0], m[0][0])
    else:
        angle_z = math.atan2(m[1][0], m[1][1])
        angle_y = math.atan2(m[0][2], m[2][2])

    def pack(angle: float) -> int:
        return int(round(angle / (2 * math.pi) * 1024)) & 0x3FF

    return (m[0][3], m[1][3], m[2][3]), (pack(angle_x), pack(angle_y), pack(angle_z))


def pack_angles(angles: tuple[int, int, int]) -> int:
    ax, ay, az = angles
    return ((ax & 0x3FF) << 20) | ((ay & 0x3FF) << 10) | (az & 0x3FF)


def quat_slerp(a, b, t):
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0.0:
        b = tuple(-v for v in b)
        dot = -dot
    if dot > 0.9995:
        out = tuple(x + (y - x) * t for x, y in zip(a, b))
    else:
        theta = math.acos(min(max(dot, -1.0), 1.0))
        sin_theta = math.sin(theta)
        wa = math.sin((1 - t) * theta) / sin_theta
        wb = math.sin(t * theta) / sin_theta
        out = tuple(x * wa + y * wb for x, y in zip(a, b))
    length = math.sqrt(sum(v * v for v in out)) or 1.0
    return tuple(v / length for v in out)


# --------------------------------------------------------------------------
# joint tree
# --------------------------------------------------------------------------

@dataclass
class Joint:
    node: int
    parent: int | None       # index into the joint list
    flags: int               # NODE_FLAG_POP / PUSH, unused for joint 0
    name: str


def mat_inverse_rigid(m: Mat) -> Mat:
    """Inverse of a rotation-plus-translation matrix."""
    out = [[m[j][i] for j in range(3)] + [0.0] for i in range(3)] + [[0.0, 0.0, 0.0, 1.0]]
    for i in range(3):
        out[i][3] = -sum(out[i][k] * m[k][3] for k in range(3))
    return out


def mat_inverse_affine(m: Mat, label: str = "matrix") -> Mat:
    """Inverse of any affine transform, scale and shear included.

    `mat_inverse_rigid` transposes, which is only the inverse when the basis is
    orthonormal. Binding a skinned mesh needs the general case: a bind pose
    routinely carries a scale, and transposing one is wrong by the square of it.
    """
    a = [row[:3] for row in m[:3]]
    det = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
           - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
           + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    if abs(det) < 1e-12:
        raise BuildError(
            f"{label}: a transform collapses to zero volume and cannot be inverted"
        )
    inv = [[0.0] * 4 for _ in range(3)] + [[0.0, 0.0, 0.0, 1.0]]
    for row in range(3):
        for column in range(3):
            r1, r2 = [i for i in range(3) if i != column]
            c1, c2 = [i for i in range(3) if i != row]
            minor = a[r1][c1] * a[r2][c2] - a[r1][c2] * a[r2][c1]
            inv[row][column] = ((-1) ** (row + column)) * minor / det
    for row in range(3):
        inv[row][3] = -sum(inv[row][k] * m[k][3] for k in range(3))
    return inv


def rotation_matrix(degrees) -> Mat:
    """A pre-transform applied above the scene, as X, Y then Z degrees."""
    ax, ay, az = (math.radians(d) for d in degrees)
    out = IDENTITY
    for angle, axis in ((ax, 0), (ay, 1), (az, 2)):
        if not angle:
            continue
        c, s = math.cos(angle), math.sin(angle)
        if axis == 0:
            m = [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]]
        elif axis == 1:
            m = [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]]
        else:
            m = [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        out = mat_mul(out, [[float(v) for v in row] for row in m])
    return out


def orthonormalize(m: Mat) -> Mat:
    """The rigid part of a transform: same rotation and position, no scale.

    The engine has no per-joint scale, so a scaled chain cannot be encoded as
    joints. Stripping the scale here and baking it into the vertices instead
    keeps the shape while making the chain expressible. COLLADA needs this
    routinely: it carries unit conversion as a node scale.
    """
    out = [row[:] for row in m]
    for column in range(3):
        length = math.sqrt(sum(m[row][column] ** 2 for row in range(3)))
        if length > 1e-9:
            for row in range(3):
                out[row][column] = m[row][column] / length
    return out


def node_world_matrices(model: gltf_reader.Gltf, rotation, translation, time,
                        pre: Mat = IDENTITY) -> dict[int, Mat]:
    """Every node's world transform in glTF space, with any animation applied."""
    world: dict[int, Mat] = {}

    def walk(index: int, above: Mat) -> None:
        node = model.nodes[index]
        quat = node.rotation
        if index in rotation:
            sampled = sample_rotation(rotation[index], time)
            if sampled:
                quat = sampled
        pos = node.translation
        if index in translation:
            pos = sample_vec3(translation[index], time, node.translation)
        here = mat_mul(above, mat_from_trs(pos, quat, node.scale))
        world[index] = here
        for child in node.children:
            walk(child, here)

    for root in model.roots:
        walk(root, pre)
    return world


def piece_disagrees_with_normals(primitives, triangles, members, flip):
    """Is this piece, as currently wound, facing away from its saved normals?

    Returns None when the file carries no normals for it, which is the one case
    where there is nothing to read and the caller has to fall back on the
    winding order the file was written with.
    """
    agree = disagree = 0.0
    for index in members:
        prim_index, offset, _ = triangles[index]
        prim = primitives[prim_index]
        if not prim.normals:
            continue
        corners = list(prim.indices[offset:offset + 3])
        if flip[index]:
            corners[1], corners[2] = corners[2], corners[1]
        a, b, c = (prim.positions[i] for i in corners)
        u = [b[i] - a[i] for i in range(3)]
        v = [c[i] - a[i] for i in range(3)]
        face = (u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0])
        saved = [sum(prim.normals[i][axis] for i in corners) for axis in range(3)]
        area = (face[0] ** 2 + face[1] ** 2 + face[2] ** 2) ** 0.5 / 2.0
        if area == 0.0:
            continue
        if sum(face[axis] * saved[axis] for axis in range(3)) >= 0.0:
            agree += area
        else:
            disagree += area
    if agree + disagree == 0.0:
        return None
    return disagree > agree


def repair_winding(model: gltf_reader.Gltf, verbose: bool = False) -> bool:
    """Make every face of a mesh agree on which side is out.

    The engine culls back faces, so a mesh whose triangles disagree about their
    facing loses the dissenting half to the cull: holes, edges that look
    doubled, a model that reads as broken. Game rips disagree routinely.

    Emitting every triangle twice hides that, at the price of twice the
    rasterising. Repairing it costs nothing at draw time, and is possible
    whenever the surface is orientable: neighbouring triangles share an edge,
    and two triangles agree exactly when they walk that shared edge in opposite
    directions. Walking the adjacency graph from any seed therefore fixes a
    whole connected piece relative to that seed.

    That settles consistency but not which way is out. Measuring the outside of
    each piece is the wrong way to settle it: a wheel, a lock of hair, any small
    shell gets the answer wrong often enough to matter, and nothing downstream
    can undo it because the winding flip is decided per joint and a joint holds
    many pieces. The file itself is the better authority -- it is consistent
    between its pieces even when a few faces inside one are turned around -- so
    each piece simply keeps the facing the majority of its own faces had.

    A piece that is not orientable at all -- an edge shared by three faces, a
    Moebius strip -- is left alone and reported, and the caller falls back to
    drawing it both ways.
    """
    trusted = True
    flipped_total = 0
    turned_total = 0

    for node in model.nodes:
        if node.mesh is None:
            continue
        primitives = model.meshes[node.mesh]

        # Vertices are welded by position so that adjacency survives both the
        # UV seams inside a primitive and the split between primitives.
        weld: dict[tuple, int] = {}
        corner_id: dict[tuple[int, int], int] = {}
        for pi, prim in enumerate(primitives):
            for ci, point in enumerate(prim.positions):
                key = tuple(round(c, 5) for c in point)
                corner_id[(pi, ci)] = weld.setdefault(key, len(weld))

        triangles: list[tuple[int, int, tuple[int, int, int]]] = []
        for pi, prim in enumerate(primitives):
            for t in range(0, len(prim.indices), 3):
                a, b, c = prim.indices[t:t + 3]
                ids = (corner_id[(pi, a)], corner_id[(pi, b)], corner_id[(pi, c)])
                if len(set(ids)) == 3:
                    triangles.append((pi, t, ids))
        if not triangles:
            continue

        # Every directed edge, so a neighbour's direction can be compared.
        edges: dict[tuple[int, int], list[tuple[int, bool]]] = {}
        for index, (_, _, ids) in enumerate(triangles):
            for k in range(3):
                u, v = ids[k], ids[(k + 1) % 3]
                edges.setdefault((min(u, v), max(u, v)), []).append((index, u < v))

        flip, component, components, orientable = repair_walk(triangles, edges)

        # Which way is out is read, never guessed. The normals saved with the
        # mesh say it outright, so each piece is turned to agree with its own;
        # a file stripped of them says it only through its winding order, and
        # then the piece keeps the facing the majority of its faces already had.
        #
        # What is not allowed is measuring a piece's outside from its shape --
        # by signed volume, or by whether its faces look away from its middle.
        # That turned wheels, hair and other small shapes inside out although
        # they had arrived correct, and nothing downstream could undo it: the
        # winding flip is decided per joint, and a joint holds many pieces.
        turned = set()
        for piece in range(components):
            members = [i for i in range(len(triangles)) if component[i] == piece]
            if not members:
                continue
            against = piece_disagrees_with_normals(primitives, triangles, members, flip)
            if against is None:
                against = sum(1 for i in members if flip[i]) * 2 > len(members)
            if not against:
                continue
            turned.add(piece)
            for i in members:
                flip[i] = not flip[i]

        changed = 0
        for index, (pi, t, _) in enumerate(triangles):
            if not flip[index]:
                continue
            indices = primitives[pi].indices
            indices[t + 1], indices[t + 2] = indices[t + 2], indices[t + 1]
            changed += 1

        flipped_total += changed
        turned_total += len(turned)
        if not orientable:
            trusted = False

    if verbose and (flipped_total or not trusted):
        if trusted:
            print(f"    note: winding repaired, {flipped_total} face(s) turned to agree "
                  f"with their neighbours ({turned_total} piece(s) turned back to the "
                  f"facing the file gave them)")
        else:
            print("    note: a mesh is not orientable -- an edge is shared by more than "
                  "two faces -- so its facing cannot be repaired; it will be drawn "
                  "both ways instead")
    model.json["winding_trusted"] = trusted
    return trusted


def repair_walk(triangles, edges):
    """Spread one orientation across each connected piece of the surface.

    Two triangles sharing an edge agree exactly when they walk it in opposite
    directions, so a neighbour that walks it the same way has to be turned.
    """
    from collections import deque

    neighbours: dict[int, list[tuple[int, bool]]] = {}
    for (u, v), uses in edges.items():
        if len(uses) != 2:
            continue                    # a border, or a non-manifold edge
        (a, a_forward), (b, b_forward) = uses
        # Same direction means one of them is turned over relative to the other.
        neighbours.setdefault(a, []).append((b, a_forward == b_forward))
        neighbours.setdefault(b, []).append((a, a_forward == b_forward))

    flip = [False] * len(triangles)
    component = [-1] * len(triangles)
    components = 0
    orientable = True

    for seed in range(len(triangles)):
        if component[seed] >= 0:
            continue
        component[seed] = components
        queue = deque([seed])
        while queue:
            here = queue.popleft()
            for other, must_turn in neighbours.get(here, ()):
                wanted = flip[here] != must_turn
                if component[other] < 0:
                    component[other] = components
                    flip[other] = wanted
                    queue.append(other)
                elif flip[other] != wanted:
                    orientable = False   # the two paths to it disagree
        components += 1
    return flip, component, components, orientable


def bake_static_skins(model: gltf_reader.Gltf, verbose: bool = False) -> int:
    """Freeze a skin nothing animates into plain geometry on its own node.

    Splitting a skin into one rigid piece per bone is what makes it animatable,
    but it is not free: neighbouring pieces meet along a shared edge, and each
    one rounds that edge's vertices to the format's four-unit lattice in its own
    frame. Two frames, two roundings, and the shared edge becomes two edges a
    pixel apart -- a crack, which is only invisible while the model is small on
    screen.

    A skin whose bones never move buys nothing for that price. Baking it leaves
    an ordinary rigid mesh, which the capacity splitter then cuts into child
    joints at zero offset, all sharing one frame: the same vertex rounds the
    same way in every piece, so the seams are exact.
    """
    animated = {channel.node for animation in model.animations
                for channel in animation.channels}
    baked = 0

    for node in model.nodes:
        if node.mesh is None or node.skin is None:
            continue
        skin = model.skins[node.skin]
        if animated & set(skin.joints):
            continue

        world = node_world_matrices(model, {}, {}, 0.0)
        if node.index not in world or any(b not in world for b in skin.joints):
            continue
        to_local = mat_inverse_affine(world[node.index],
                                      f"{model.path.name}:{node.name}")
        to_world = [mat_mul(world[b], skin.inverse_bind[i])
                    for i, b in enumerate(skin.joints)]

        def place(matrix, point):
            return tuple(
                matrix[axis][0] * point[0] + matrix[axis][1] * point[1]
                + matrix[axis][2] * point[2] + matrix[axis][3]
                for axis in range(3)
            )

        def turn(matrix, vector):
            """A direction under an affine transform: the linear part only."""
            return tuple(
                matrix[axis][0] * vector[0] + matrix[axis][1] * vector[1]
                + matrix[axis][2] * vector[2]
                for axis in range(3)
            )

        for prim in model.meshes[node.mesh]:
            if prim.joints is None or prim.weights is None:
                continue
            turned = [] if prim.normals else None
            placed = []
            for index, point in enumerate(prim.positions):
                blended = [0.0, 0.0, 0.0]
                total = 0.0
                for bone, weight in zip(prim.joints[index], prim.weights[index]):
                    if weight <= 0.0 or bone >= len(to_world):
                        continue
                    total += weight
                    moved_point = place(to_world[bone], point)
                    for axis in range(3):
                        blended[axis] += weight * moved_point[axis]
                placed.append(point if total <= 0.0
                              else place(to_local, tuple(c / total for c in blended)))
                if turned is not None:
                    # The vertex moved, so its normal has to move with it or it
                    # stops describing the face it belongs to.
                    normal = prim.normals[index]
                    if total <= 0.0:
                        turned.append(normal)
                    else:
                        spun = [0.0, 0.0, 0.0]
                        for bone, weight in zip(prim.joints[index], prim.weights[index]):
                            if weight <= 0.0 or bone >= len(to_world):
                                continue
                            moved_normal = turn(to_world[bone], normal)
                            for axis in range(3):
                                spun[axis] += weight * moved_normal[axis]
                        turned.append(turn(to_local, tuple(c / total for c in spun)))
            prim.positions = placed
            if turned is not None:
                prim.normals = turned
            prim.joints = None
            prim.weights = None
        node.skin = None
        baked += 1

    if verbose and baked:
        print(f"    note: {baked} skin(s) are never animated; baked into rigid "
              "geometry so the pieces share one frame and their seams stay exact")
    return baked


def skirt_triangles(triangles, owners, placed) -> list[tuple[int, int]]:
    """Which triangles each bone should keep a copy of, just past its own edge.

    Pieces cut from one skin meet edge to edge with nothing to spare, so the
    seam is a butt joint: the moment the bones either side of it turn, it opens
    and the background shows through. Tomb Raider's own models never do this --
    a limb is a closed shape that sinks into its neighbour, and the two simply
    interpenetrate.

    The same shape is given to a cut skin here. Every triangle across a seam is
    copied into the neighbouring piece as well, baked into that piece's frame,
    so each side reaches a row past the cut. Turning the joint slides the two
    overlaps across each other instead of parting them.

    The cost is that those triangles are drawn twice, and at rest they coincide
    exactly, so nothing changes on screen until the seam actually moves.
    """
    weld: dict[tuple, int] = {}
    ids: list[tuple[int, int, int]] = []
    for prim, corners in triangles:
        vertex = []
        for corner in corners:
            point = placed[id(prim)][corner]
            key = tuple(round(c, 4) for c in point)
            vertex.append(weld.setdefault(key, len(weld)))
        ids.append(tuple(vertex))

    edges: dict[tuple[int, int], list[int]] = {}
    for index, vertex in enumerate(ids):
        for k in range(3):
            u, v = vertex[k], vertex[(k + 1) % 3]
            edges.setdefault((min(u, v), max(u, v)), []).append(index)

    out: set[tuple[int, int]] = set()
    for users in edges.values():
        for a in users:
            for b in users:
                if owners[a] != owners[b]:
                    out.add((owners[a], b))
    return sorted(out)


def bind_to_skeleton(model: gltf_reader.Gltf, verbose: bool = False,
                     skirt: bool = True) -> int:
    """Split a skinned mesh into one rigid mesh per bone.

    The engine has no skinning. It draws a rigid mesh per joint and moves the
    joints, which is how Tomb Raider's own models are built, so a skinned export
    is converted to that shape rather than rejected: every triangle is given to
    the bone carrying most of its weight, and its vertices are baked into that
    bone's rest frame. Smooth deformation across a joint is lost -- the seam
    becomes a hinge -- which is the same trade the original models make.

    Without this a skinned character arrives as one mesh on one joint: rigid,
    impossible to animate, and usually past the 255-vertex ceiling of a single
    mesh block, because the whole body sits inside it.

    The bone nodes already exist in the scene, so the split only has to move the
    geometry onto them; joint discovery then finds them the ordinary way, by
    looking for nodes that carry a mesh.
    """
    # Only meshes that need it are rebound. A skin with a single bone carries no
    # articulation -- Blender writes one as a placeholder -- and a still mesh
    # that already fits a block is already split the way its author intended,
    # usually better than by weight. Rebinding either would throw away a good
    # split to no purpose.
    #
    # An animated model is the case that always needs it: the take moves the
    # bones, so geometry left on the mesh node would sit still while the
    # skeleton it belongs to plays underneath.
    animated = {channel.node for animation in model.animations
                for channel in animation.channels}
    # Being too big for one block is no longer a reason to rebind: the capacity
    # splitter cuts a mesh into child joints at zero offset, which share a frame
    # and therefore round their shared vertices identically. Bone pieces do not,
    # so they are used only where they earn their seams -- when the bones move.
    skinned = [node for node in model.nodes
               if node.mesh is not None and node.skin is not None
               and len(model.skins[node.skin].joints) > 1
               and animated & set(model.skins[node.skin].joints)]
    if not skinned:
        return 0

    world = node_world_matrices(model, {}, {}, 0.0)
    added: dict[int, list[gltf_reader.Primitive]] = {}
    moved = 0
    copied = 0

    for node in skinned:
        skin = model.skins[node.skin]
        missing = [b for b in skin.joints if b not in world]
        if missing:
            raise BuildError(
                f"{model.path.name}: mesh {node.name!r} is bound to {len(missing)} bone(s) "
                "that are not in the scene graph"
            )
        # Bind pose to rest pose, per bone, exactly as a GPU skin would build it.
        to_world = [mat_mul(world[b], skin.inverse_bind[i])
                    for i, b in enumerate(skin.joints)]
        to_bone = [mat_inverse_affine(world[b], f"{model.path.name}:{model.nodes[b].name}")
                   for b in skin.joints]

        def skinned_point(prim, index):
            point = prim.positions[index]
            blended = [0.0, 0.0, 0.0]
            total = 0.0
            if prim.joints is not None and prim.weights is not None:
                for bone, weight in zip(prim.joints[index], prim.weights[index]):
                    if weight <= 0.0 or bone >= len(to_world):
                        continue
                    total += weight
                    matrix = to_world[bone]
                    for axis in range(3):
                        blended[axis] += weight * (
                            matrix[axis][0] * point[0] + matrix[axis][1] * point[1]
                            + matrix[axis][2] * point[2] + matrix[axis][3])
            if total <= 0.0:                     # unweighted: leave it where it lies
                matrix = world[node.index]
                return tuple(
                    matrix[axis][0] * point[0] + matrix[axis][1] * point[1]
                    + matrix[axis][2] * point[2] + matrix[axis][3]
                    for axis in range(3))
            return tuple(c / total for c in blended)

        def vertex_influences(prim, index):
            """Every bone the source names for a vertex, heaviest first.

            Kept whole rather than reduced to the heaviest: at the bind pose the
            two agree exactly -- every bone maps the vertex to the same place --
            so the difference only shows once the skeleton moves, which is
            precisely when it matters.
            """
            if prim.joints is None or prim.weights is None:
                return []
            pairs = [(bone, share)
                     for bone, share in zip(prim.joints[index], prim.weights[index])
                     if share > 0.001 and bone < len(to_world)]
            pairs.sort(key=lambda pair: (-pair[1], pair[0]))
            return pairs[:4]

        def vertex_owner(prim, index) -> int:
            """The bone a single vertex follows.

            One influence per vertex is what a DS or GBA era model carries, and
            what this reads: the corner belongs to a bone, not the triangle. A
            triangle whose corners sit on different bones is stretched between
            them, which is exactly how such a model bends.
            """
            best, weight = None, 0.0
            if prim.joints is not None and prim.weights is not None:
                for bone, share in zip(prim.joints[index], prim.weights[index]):
                    if share > weight and bone < len(to_world):
                        best, weight = bone, share
            return 0 if best is None else best

        def owner(prim, corners) -> int:
            """The bone holding most of a triangle's weight, summed over corners."""
            tally: dict[int, float] = {}
            if prim.joints is not None and prim.weights is not None:
                for corner in corners:
                    for bone, weight in zip(prim.joints[corner], prim.weights[corner]):
                        if weight > 0.0 and bone < len(to_world):
                            tally[bone] = tally.get(bone, 0.0) + weight
            if not tally:
                return 0
            # Ties go to the earlier bone so the split is reproducible.
            return min(tally.items(), key=lambda pair: (-pair[1], pair[0]))[0]

        def skinned_normal(prim, index):
            """The vertex's normal, moved by the same bones as the vertex."""
            normal = prim.normals[index]
            if prim.joints is None or prim.weights is None:
                return normal
            spun = [0.0, 0.0, 0.0]
            total = 0.0
            for bone, weight in zip(prim.joints[index], prim.weights[index]):
                if weight <= 0.0 or bone >= len(to_world):
                    continue
                total += weight
                matrix = to_world[bone]
                for axis in range(3):
                    spun[axis] += weight * (
                        matrix[axis][0] * normal[0] + matrix[axis][1] * normal[1]
                        + matrix[axis][2] * normal[2])
            return normal if total <= 0.0 else tuple(c / total for c in spun)

        placed = {id(prim): [skinned_point(prim, i) for i in range(len(prim.positions))]
                  for prim in model.meshes[node.mesh]}
        # Normals follow the geometry through the split: they are the only thing
        # that says which side of a face is the front once a shell is open.
        placed_normals = {id(prim): ([skinned_normal(prim, i)
                                      for i in range(len(prim.positions))]
                                     if prim.normals else None)
                          for prim in model.meshes[node.mesh]}
        triangles = [(prim, tuple(prim.indices[t:t + 3]))
                     for prim in model.meshes[node.mesh]
                     for t in range(0, len(prim.indices), 3)]
        owners = [owner(prim, corners) for prim, corners in triangles]

        # (bone, material) -> triangles, keeping the primitive split the engine needs.
        buckets: dict[tuple[int, int | None], dict] = {}

        def deposit(bone: int, prim, corners) -> None:
            key = (bone, prim.material)
            bucket = buckets.get(key)
            if bucket is None:
                bucket = {"positions": [], "uvs": [], "normals": [], "bones": [],
                          "blends": [], "indices": [], "slots": {},
                          "has_uv": prim.uvs is not None,
                          "has_normal": prim.normals is not None}
                buckets[key] = bucket
            for corner in corners:
                slot_key = (id(prim), corner)
                slot = bucket["slots"].get(slot_key)
                if slot is None:
                    slot = len(bucket["positions"])
                    bucket["slots"][slot_key] = slot
                    bucket["bones"].append(skin.joints[vertex_owner(prim, corner)])
                    influences = vertex_influences(prim, corner)
                    if len(influences) > 1:
                        total = sum(share for _, share in influences)
                        world_point = placed[id(prim)][corner]
                        bucket["blends"].append(tuple(
                            (skin.joints[bone], share / total,
                             tuple(to_bone[bone][axis][0] * world_point[0]
                                   + to_bone[bone][axis][1] * world_point[1]
                                   + to_bone[bone][axis][2] * world_point[2]
                                   + to_bone[bone][axis][3]
                                   for axis in range(3)))
                            for bone, share in influences))
                    else:
                        bucket["blends"].append(None)
                    bucket["positions"].append(placed[id(prim)][corner])
                    bucket["uvs"].append(prim.uvs[corner] if prim.uvs else (0.0, 0.0))
                    normals_of = placed_normals[id(prim)]
                    bucket["normals"].append(
                        normals_of[corner] if normals_of else (0.0, 0.0, 0.0))
                bucket["indices"].append(slot)

        for index, (prim, corners) in enumerate(triangles):
            deposit(owners[index], prim, corners)

        for bone, index in (skirt_triangles(triangles, owners, placed) if skirt else ()):
            prim, corners = triangles[index]
            deposit(bone, prim, corners)
            copied += 1

        bone_of_node = {node_index: bone for bone, node_index in enumerate(skin.joints)}
        for (bone, material), bucket in buckets.items():
            bone_node = skin.joints[bone]
            # Each vertex goes into the frame of the bone it follows, not the
            # frame of the block it happens to be stored in. That is what lets a
            # triangle stretch across a joint the way the source does, and it
            # also makes a vertex shared by two blocks quantise identically in
            # both, so the seam cannot split.
            local, local_normals = [], []
            for index, point in enumerate(bucket["positions"]):
                matrix = to_bone[bone_of_node[bucket["bones"][index]]]
                local.append(tuple(
                    matrix[axis][0] * point[0] + matrix[axis][1] * point[1]
                    + matrix[axis][2] * point[2] + matrix[axis][3]
                    for axis in range(3)))
                # A direction carries no translation, so only the linear part.
                normal = bucket["normals"][index]
                local_normals.append(tuple(
                    matrix[axis][0] * normal[0] + matrix[axis][1] * normal[1]
                    + matrix[axis][2] * normal[2]
                    for axis in range(3)))
            added.setdefault(bone_node, []).append(gltf_reader.Primitive(
                positions=local,
                uvs=bucket["uvs"] if bucket["has_uv"] else None,
                indices=bucket["indices"],
                material=material,
                normals=local_normals if bucket["has_normal"] else None,
                bones=list(bucket["bones"]),
                blends=list(bucket["blends"]),
            ))
            moved += len(bucket["indices"]) // 3

        node.mesh = None
        node.skin = None

    for bone_node, primitives in added.items():
        target = model.nodes[bone_node]
        if target.mesh is None:
            target.mesh = len(model.meshes)
            model.meshes.append(primitives)
        else:                                # two skins feeding one bone
            model.meshes[target.mesh].extend(primitives)

    if verbose:
        print(f"    skinned mesh bound to {len(added)} bones, {moved} triangles rigidly "
              "assigned (deformation across joints is not representable)")
        if copied:
            print(f"    note: {copied} triangle(s) copied across the seams so the pieces "
                  "overlap by a row; a butt joint opens as soon as the bones turn")
    return len(added)


def merge_excess_joints(model: gltf_reader.Gltf, verbose: bool = False) -> int:
    """Fold the cheapest leaf joints into their parents until the mask can hold them.

    The catalog addresses joints through a 32-bit visibility mask, so 32 is a
    real ceiling and not a guess. A rig with more than that can still be drawn,
    though: a leaf joint's mesh can be baked into its parent, which costs the
    leaf's independent movement and nothing else.

    Leaves are taken in order of what merging them loses -- first the ones no
    animation touches, where the result is identical, then the ones carrying the
    least geometry. Only leaves are merged, so no other joint's parent changes
    and no animation channel is left pointing at a node that is gone.
    """
    order: list[int] = []
    parent_of: dict[int, int | None] = {}

    def walk(index: int, parent: int | None) -> None:
        parent_of[index] = parent
        order.append(index)
        for child in model.nodes[index].children:
            walk(child, index)

    for root in model.roots:
        walk(root, None)

    carries = {i: model.nodes[i].mesh is not None for i in order}
    for index in reversed(order):
        if any(carries[c] for c in model.nodes[index].children):
            carries[index] = True
    joints = [i for i in order if carries[i]]
    if len(joints) <= MAX_JOINTS:
        return 0

    animated = {channel.node for animation in model.animations
                for channel in animation.channels}

    def faces_of(index: int) -> int:
        node = model.nodes[index]
        if node.mesh is None:
            return 0
        return sum(len(p.indices) // 3 for p in model.meshes[node.mesh])

    merged = 0
    lossy = 0
    while len(joints) > MAX_JOINTS:
        kept = set(joints)
        candidates = [
            i for i in joints
            if parent_of.get(i) in kept                    # never the root
            and not any(c in kept for c in model.nodes[i].children)
            and model.nodes[i].mesh is not None
        ]
        if not candidates:
            break
        # No animation first, then the least geometry: the cheapest thing to lose.
        candidates.sort(key=lambda i: (i in animated, faces_of(i), i))
        leaf = candidates[0]
        parent = parent_of[leaf]
        if leaf in animated:
            lossy += 1
        if not merge_joint_into(model, leaf, parent):
            break
        joints.remove(leaf)
        merged += 1

    if merged and verbose:
        note = (f", {lossy} of them animated (their own movement is lost)"
                if lossy else " (none of them animated, so nothing moves differently)")
        print(f"    note: {len(joints) + merged} joints is past the {MAX_JOINTS} the "
              f"visibility mask can address; {merged} leaf joint(s) merged into their "
              f"parents{note}")
    if len(joints) > MAX_JOINTS:
        raise BuildError(
            f"{model.path.name}: {len(joints)} joints after merging every leaf that "
            f"could be merged, and the catalog mask addresses {MAX_JOINTS}. Join some "
            "parts to their parent bone in the modeller."
        )
    return merged


def merge_joint_into(model: gltf_reader.Gltf, leaf: int, parent: int) -> bool:
    """Bake a leaf joint's geometry into its parent, in the parent's frame."""
    world = node_world_matrices(model, {}, {}, 0.0)
    if leaf not in world or parent not in world:
        return False
    try:
        into_parent = mat_mul(mat_inverse_affine(world[parent], model.nodes[parent].name),
                              world[leaf])
    except BuildError:
        return False

    moved = []
    for prim in model.meshes[model.nodes[leaf].mesh]:
        points = [tuple(
            into_parent[axis][0] * p[0] + into_parent[axis][1] * p[1]
            + into_parent[axis][2] * p[2] + into_parent[axis][3]
            for axis in range(3)) for p in prim.positions]
        moved.append(gltf_reader.Primitive(
            positions=points, uvs=prim.uvs, indices=list(prim.indices),
            material=prim.material))

    target = model.nodes[parent]
    if target.mesh is None:
        target.mesh = len(model.meshes)
        model.meshes.append(moved)
    else:
        model.meshes[target.mesh].extend(moved)

    model.nodes[leaf].mesh = None
    if leaf in target.children:
        target.children.remove(leaf)
    return True


def split_oversized_meshes(model: gltf_reader.Gltf, verbose: bool = False,
                           budget: int = SPLIT_BUDGET) -> int:
    """Break a mesh too big for one block into child joints at zero offset.

    A mesh block stores its vertex count in a uint8 and chains face indices as
    int8 deltas, so one joint holds a couple of hundred vertices and no more.
    Formats that carry no skeleton -- OBJ above all, but also any single-object
    export -- hand over a whole character as one mesh, well past that.

    The engine's own way out is the one taken here: more joints. Each overflow
    chunk becomes a child node with an identity transform, so it draws in
    exactly the same place while occupying its own mesh block. Chunks are cut
    along the widest axis so each stays compact, which keeps its bounding sphere
    tight and its index chain short. The cost is joint slots, of which there are
    32; the result is indistinguishable.
    """
    added = 0
    for node in list(model.nodes):
        if node.mesh is None:
            continue
        primitives = model.meshes[node.mesh]
        if welded_count(primitives) <= budget:
            continue

        triangles = [(index, tuple(prim.indices[t:t + 3]))
                     for index, prim in enumerate(primitives)
                     for t in range(0, len(prim.indices), 3)]

        def centroid(entry):
            index, corners = entry
            points = [primitives[index].positions[c] for c in corners]
            return tuple(sum(p[axis] for p in points) / 3.0 for axis in range(3))

        def parts(entries: list) -> list[list]:
            if len(entries) <= 1 or \
                    welded_count(chunk_primitives(primitives, entries)) <= budget:
                return [entries]
            centres = [centroid(entry) for entry in entries]
            spans = [(max(c[axis] for c in centres) - min(c[axis] for c in centres), axis)
                     for axis in range(3)]
            axis = max(spans)[1]
            ordered = sorted(range(len(entries)), key=lambda i: centres[i][axis])
            middle = len(ordered) // 2
            left = [entries[i] for i in ordered[:middle]]
            right = [entries[i] for i in ordered[middle:]]
            if not left or not right:              # every centroid coincides
                left, right = entries[:middle], entries[middle:]
            return parts(left) + parts(right)

        chunks = parts(triangles)
        if len(chunks) < 2:
            continue
        model.meshes[node.mesh] = chunk_primitives(primitives, chunks[0])
        for chunk in chunks[1:]:
            index = len(model.nodes)
            model.nodes.append(gltf_reader.Node(
                index=index, name=f"{node.name}_part{added + 1}", children=[],
                mesh=len(model.meshes), skin=None, translation=(0.0, 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0, 1.0), scale=(1.0, 1.0, 1.0)))
            model.meshes.append(chunk_primitives(primitives, chunk))
            node.children.append(index)
            added += 1
        if verbose:
            print(f"    mesh {node.name!r} exceeds one mesh block; split across "
                  f"{len(chunks)} joints at zero offset")
    return added


def welded_count(primitives) -> int:
    """How many vertices a mesh block would hold, after welding.

    An estimate, but a close one and always on the safe side: the encoder welds
    on quantised integer positions, which can only merge more than rounding
    here does.
    """
    seen = set()
    for prim in primitives:
        for point in prim.positions:
            seen.add(tuple(round(c, 5) for c in point))
    return len(seen)


def chunk_primitives(primitives, entries) -> list:
    """The primitives holding just these triangles, with their own vertex buffer."""
    buckets: dict[int, dict] = {}
    for prim_index, corners in entries:
        prim = primitives[prim_index]
        bucket = buckets.get(prim_index)
        if bucket is None:
            bucket = {"positions": [], "uvs": [], "normals": [], "bones": [],
                      "blends": [], "indices": [], "slots": {}}
            buckets[prim_index] = bucket
        for corner in corners:
            slot = bucket["slots"].get(corner)
            if slot is None:
                slot = len(bucket["positions"])
                bucket["slots"][corner] = slot
                bucket["positions"].append(prim.positions[corner])
                bucket["uvs"].append(prim.uvs[corner] if prim.uvs else (0.0, 0.0))
                # Carried through every cut, because they are what says which
                # side of a face is the front.
                bucket["normals"].append(
                    prim.normals[corner] if prim.normals else (0.0, 0.0, 0.0))
                bucket["bones"].append(prim.bones[corner] if prim.bones else 0)
                bucket["blends"].append(prim.blends[corner] if prim.blends else None)
            bucket["indices"].append(slot)
    out = []
    for prim_index, bucket in sorted(buckets.items()):
        prim = primitives[prim_index]
        out.append(gltf_reader.Primitive(
            positions=bucket["positions"],
            uvs=bucket["uvs"] if prim.uvs is not None else None,
            indices=bucket["indices"],
            material=prim.material,
            normals=bucket["normals"] if prim.normals is not None else None,
            bones=bucket["bones"] if prim.bones is not None else None,
            blends=bucket["blends"] if prim.blends is not None else None,
        ))
    return out


def discover_joints(model: gltf_reader.Gltf) -> list[Joint]:
    """Derive the engine's joint tree from whichever nodes carry geometry.

    The engine draws one rigid mesh per joint, so only nodes with geometry can be
    joints. Everything else is scaffolding: a rig wrapper above the model, or an
    armature beside it, whose bones the engine cannot use because it has no
    skinning. Those are pruned and their transforms fold into the joints below,
    which is why every local transform here comes from world matrices rather
    than from a node's own TRS.

    A node without geometry is still kept when geometry hangs below it: Tomb
    Raider models do that -- MUMMY has empty slots in the middle of its chain --
    and the slot has to survive to keep the hierarchy's shape.
    """
    nodes = model.nodes
    parent_of: dict[int, int | None] = {}
    order: list[int] = []

    def walk(index: int, parent: int | None) -> None:
        parent_of[index] = parent
        order.append(index)
        for child in nodes[index].children:
            walk(child, index)

    for root in model.roots:
        walk(root, None)

    carries = {index: nodes[index].mesh is not None for index in order}
    for index in reversed(order):                 # a child always precedes its parent here
        if any(carries[child] for child in nodes[index].children):
            carries[index] = True
    kept = [index for index in order if carries[index]]
    if not kept:
        raise BuildError(f"{model.path.name}: no node carries a mesh")

    # Strip wrappers above the model: empty nodes with a single kept child, whose
    # transform folds into the child through the world matrices below. A node
    # with several kept children is a real branch point and stays, even with no
    # geometry of its own: it is the joint the branches turn about, and on an
    # animated model it is usually the one the whole body hangs from. The engine
    # handles that -- drawNodesLerp sets the matrix up and only skips the mesh
    # when the visibility bit is clear.
    while nodes[kept[0]].mesh is None:
        children = [c for c in nodes[kept[0]].children if carries[c]]
        if len(children) != 1:
            break
        kept.pop(0)

    in_tree = set(kept)

    def joint_parent(index: int) -> int | None:
        walker = parent_of.get(index)
        while walker is not None and walker not in in_tree:
            walker = parent_of.get(walker)
        return walker

    slot_of = {node: slot for slot, node in enumerate(kept)}
    joints = [
        Joint(node,
              None if joint_parent(node) is None else slot_of[joint_parent(node)],
              0, nodes[node].name)
        for node in kept
    ]
    for slot, joint in enumerate(joints):
        if slot and joint.parent is None:
            joint.parent = 0

    # Stack flags: save the parent before every child but the last, restore it
    # before every child but the first. That is one pop per joint, which is all
    # the format's single POP bit allows.
    children_of: dict[int, list[int]] = {}
    for slot, joint in enumerate(joints):
        if joint.parent is not None:
            children_of.setdefault(joint.parent, []).append(slot)
    depth_of = [0] * len(joints)
    for slot, joint in enumerate(joints):
        if joint.parent is None:
            continue
        siblings = children_of[joint.parent]
        depth_of[slot] = depth_of[joint.parent] + (1 if len(siblings) > 1 else 0)
        if len(siblings) > 1:
            position = siblings.index(slot)
            flags = 0
            if position > 0:
                flags |= NODE_FLAG_POP
            if position < len(siblings) - 1:
                flags |= NODE_FLAG_PUSH
            joint.flags = flags

    if len(joints) > MAX_JOINTS:
        raise BuildError(
            f"{model.path.name}: {len(joints)} joints, the catalog mask allows {MAX_JOINTS}"
        )
    deepest = max(depth_of) if depth_of else 0
    if deepest > MAX_MATRIX_DEPTH:
        raise BuildError(
            f"{model.path.name}: branching nests {deepest} deep, the matrix stack allows "
            f"{MAX_MATRIX_DEPTH}"
        )
    return joints


def joint_locals(model, joints, scale, rotation, translation, time,
                 pre: Mat = IDENTITY) -> list[Mat]:
    """Each joint's transform relative to its parent joint, in engine space."""
    world = {index: orthonormalize(m)
             for index, m in node_world_matrices(model, rotation, translation, time, pre).items()}
    out: list[Mat] = []
    for slot, joint in enumerate(joints):
        if slot == 0:
            local = world[joint.node]
        else:
            local = mat_mul(mat_inverse_rigid(world[joints[joint.parent].node]),
                            world[joint.node])
        out.append(scaled_translation(conjugate(local), scale))
    return out


def evaluate_pose(model, joints, scale, rotation, translation, time,
                  pre: Mat = IDENTITY) -> list[Mat]:
    """Each joint's world transform in engine space, at one instant."""
    locals_ = joint_locals(model, joints, scale, rotation, translation, time, pre)
    world: list[Mat] = []
    for slot, joint in enumerate(joints):
        world.append(locals_[slot] if slot == 0
                     else mat_mul(world[joint.parent], locals_[slot]))
    return world


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

@dataclass
class Face:
    """A triangle before its material is resolved.

    The flag word embeds a texture record whose UV corners are ordered to match
    the vertices, so it can only be built once encode_mesh has settled which
    corner starts the triangle. Resolving it earlier silently mismatches the two
    whenever the delta chain rotates a face.
    """

    indices: tuple[int, int, int]
    material: int | None
    uvs: list[tuple[float, float]] | None


@dataclass
class JointMesh:
    vertices: list[tuple[int, int, int]]
    faces: list[Face]
    # The scene node each vertex follows, one per vertex, or an empty list when
    # the mesh is rigid and every vertex follows the joint it is stored on.
    bones: list[int] = field(default_factory=list)
    # For the vertices the source shares between bones: every influence, as
    # (node, weight, position in that node's rest frame). None elsewhere.
    blends: list = field(default_factory=list)
    # Triangles whose three corners quantised onto fewer than three points.
    # A handful is normal; a large share means the scale is too small to hold
    # the shape, and the caller says so rather than shipping a ruined model.
    collapsed: int = 0

    def aabb(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Local bounds in engine units; stored vertices are world units >> 2."""
        if not self.vertices:
            return (0, 0, 0), (0, 0, 0)
        lo = tuple(min(v[i] for v in self.vertices) * 4 for i in range(3))
        hi = tuple(max(v[i] for v in self.vertices) * 4 for i in range(3))
        return lo, hi


def normal_agreement(primitives) -> float:
    """How much of a mesh's area is wound to match the normals it was saved with.

    The winding order alone is a convention, and a file is free to break it. The
    normals are not a convention: they are the author's statement of which side
    of each face is the front. So they are the thing to read.

    Returns the share of triangle area whose winding agrees with its own
    normals, or -1.0 when the file carries none and therefore says nothing.
    """
    agree = disagree = 0.0
    seen = False
    for prim in primitives:
        if not getattr(prim, "normals", None):
            continue
        seen = True
        for triangle in range(0, len(prim.indices), 3):
            corners = prim.indices[triangle:triangle + 3]
            a, b, c = (prim.positions[i] for i in corners)
            u = [b[i] - a[i] for i in range(3)]
            v = [c[i] - a[i] for i in range(3)]
            face = (u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0])
            saved = [sum(prim.normals[i][axis] for i in corners) for axis in range(3)]
            area = (face[0] ** 2 + face[1] ** 2 + face[2] ** 2) ** 0.5 / 2.0
            if area == 0.0:
                continue
            if sum(face[axis] * saved[axis] for axis in range(3)) >= 0.0:
                agree += area
            else:
                disagree += area
    if not seen or agree + disagree == 0.0:
        return -1.0
    return agree / (agree + disagree)


def mesh_facing(model: gltf_reader.Gltf, node_index: int) -> tuple[bool, bool | None]:
    """Is this mesh closed, and if so does it need its triangles reversed?

    A closed shell wound the glTF way -- counter-clockwise seen from outside --
    encloses a positive volume, and the engine wants the opposite order, because
    checkBackface keeps a positive screen cross product under a downward Y.

    The volume only means something for a closed surface, so closedness is
    tested rather than guessed at from a ratio: every edge must be used exactly
    twice, once in each direction. An open shell -- a hair shell, a skirt, a face
    plate -- can enclose a large signed volume that is pure noise, and trusting
    it turns part of a figure inside out.

    An open shell is not the end of the road, though: the normals saved with it
    say which side is the front, and reading them is not guessing. Only a file
    that is both open and stripped of its normals leaves the question
    unanswered, and then nothing here invents an answer.

    Positions are welded by value first, because an exporter splits vertices at
    every UV seam and the raw indices then show holes that are not there.
    """
    mesh_index = model.nodes[node_index].mesh
    if mesh_index is None:
        return False, None

    def key(point):
        return tuple(round(c, 4) for c in point)

    directed: dict[tuple, int] = {}
    volume = 0.0
    for prim in model.meshes[mesh_index]:
        for triangle in range(0, len(prim.indices), 3):
            corners = [prim.positions[prim.indices[triangle + i]] for i in range(3)]
            a, b, c = corners
            volume += (a[0] * (b[1] * c[2] - b[2] * c[1])
                       - a[1] * (b[0] * c[2] - b[2] * c[0])
                       + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0
            welded = [key(p) for p in corners]
            for i in range(3):
                edge = (welded[i], welded[(i + 1) % 3])
                directed[edge] = directed.get(edge, 0) + 1

    closed = all(count == 1 and directed.get((b, a), 0) == 1
                 for (a, b), count in directed.items())
    if closed and volume != 0.0:
        return True, volume > 0.0

    # Not closed, so there is no volume to sign -- but the normals the author
    # saved answer the same question directly, and a file that carries them is
    # not guessed at. A closed shell wound the glTF way has both a positive
    # volume and a winding that matches its normals, so the two paths agree.
    agreement = normal_agreement(model.meshes[mesh_index])
    if agreement >= 0.0:
        return True, agreement >= 0.5
    return False, None


def local_points(model: gltf_reader.Gltf, joint: Joint,
                 world: dict[int, Mat] | None) -> list[list[tuple[float, float, float]]]:
    """Each primitive's vertices in the joint's own frame, before quantisation.

    Every vertex is taken to world space and then brought back into the joint's
    rigid frame. That covers three things at once: a skinned mesh, whose node
    transform the format says to ignore and whose rest pose must be baked
    because the engine has no skinning; a plain mesh under a transformed node;
    and any scale in the chain, which the engine cannot express per joint and
    which therefore ends up in the geometry.
    """
    node = model.nodes[joint.node]
    if node.mesh is None:
        return []
    primitives = model.meshes[node.mesh]
    if world is None:
        return [list(prim.positions) for prim in primitives]

    to_local = mat_inverse_rigid(orthonormalize(world[joint.node]))
    skin_matrices = None
    if node.skin is not None:
        source = model.skins[node.skin]
        skin_matrices = [mat_mul(world[source.joints[i]], source.inverse_bind[i])
                         for i in range(len(source.joints))]

    def apply(matrix, point):
        return tuple(
            matrix[axis][0] * point[0] + matrix[axis][1] * point[1]
            + matrix[axis][2] * point[2] + matrix[axis][3]
            for axis in range(3)
        )

    out = []
    for prim in primitives:
        points = []
        for index, point in enumerate(prim.positions):
            if skin_matrices and prim.joints is not None and prim.weights is not None:
                blended = [0.0, 0.0, 0.0]
                weighted = False
                for bone, weight in zip(prim.joints[index], prim.weights[index]):
                    if weight <= 0.0 or bone >= len(skin_matrices):
                        continue
                    weighted = True
                    moved = apply(skin_matrices[bone], point)
                    for axis in range(3):
                        blended[axis] += weight * moved[axis]
                world_point = tuple(blended) if weighted else apply(world[joint.node], point)
            else:
                world_point = apply(world[joint.node], point)
            points.append(apply(to_local, world_point))
        out.append(points)
    return out


def seam_anchors(model: gltf_reader.Gltf, joints: list["Joint"], scale: float,
                 pre: Mat, world: dict[int, Mat] | None) -> dict[int, dict]:
    """Pull the vertices two joints share onto one agreed point.

    Rigid binding cuts a skin along triangle edges, so the pieces meet edge to
    edge with nothing to spare. Each piece then rounds that shared edge to the
    format's four-unit lattice in its own frame, and two frames round it two
    ways: the edge becomes two edges, and the background shows between them.

    The lattices cannot be made to agree -- a lattice carried through a rotation
    is no longer a lattice -- but the disagreement can be halved. One joint is
    made the owner of each shared vertex and rounds it as it likes; every other
    joint is then handed the point the owner actually stored, and rounds that
    instead of its own reading. The owner's error becomes common to all of them
    rather than adding to theirs.

    Returns, per joint slot, the corners whose position should be replaced, in
    the joint's own frame and in the model's own units, so the caller quantises
    them exactly as it would any other vertex.
    """
    # The engine composes the pose from integer node offsets, so the anchors are
    # measured on the matrices it will really use, not on the exact ones.
    locals_ = joint_locals(model, joints, scale, {}, {}, 0.0, pre)
    rest: list[Mat] = []
    for slot, joint in enumerate(joints):
        position, _ = decompose(locals_[slot], f"{model.path.name}:{joint.name}")
        rows = [list(row) for row in locals_[slot]]
        for axis in range(3):
            rows[axis][3] = float(round(position[axis]))
        rounded = tuple(tuple(row) for row in rows)
        rest.append(rounded if slot == 0 else mat_mul(rest[joint.parent], rounded))

    def apply(matrix, point):
        return tuple(
            matrix[axis][0] * point[0] + matrix[axis][1] * point[1]
            + matrix[axis][2] * point[2] + matrix[axis][3]
            for axis in range(3)
        )

    shared: dict[tuple, list] = {}
    for slot, joint in enumerate(joints):
        for prim_index, points in enumerate(local_points(model, joint, world)):
            for corner, point in enumerate(points):
                engine = (point[0] * scale, -point[1] * scale, -point[2] * scale)
                key = tuple(round(c, 3) for c in apply(rest[slot], engine))
                shared.setdefault(key, []).append((slot, prim_index, corner, engine))

    out: dict[int, dict] = {}
    for members in shared.values():
        if len(members) < 2 or len({m[0] for m in members}) < 2:
            continue
        owner_slot, _, _, owner_engine = members[0]
        stored = apply(rest[owner_slot], tuple(4.0 * round(c / 4.0) for c in owner_engine))
        for slot, prim_index, corner, _ in members[1:]:
            back = apply(mat_inverse_rigid(rest[slot]), stored)
            out.setdefault(slot, {})[(prim_index, corner)] = (
                back[0] / scale, -back[1] / scale, -back[2] / scale)
    return out


def build_joint_mesh(model: gltf_reader.Gltf, joint: Joint, scale: float,
                     flip_winding: bool, world: dict[int, Mat] | None = None,
                     double_sided: bool = False,
                     anchors: dict | None = None) -> JointMesh:
    node = model.nodes[joint.node]
    if node.mesh is None:
        return JointMesh([], [], [], [])
    primitives = model.meshes[node.mesh]
    placed = local_points(model, joint, world)
    # Corners this joint shares with another one are taken from the seam anchor
    # rather than from its own reading, so both joints round the same point.
    for (prim_index, corner), point in (anchors or {}).items():
        placed[prim_index][corner] = point

    vertex_slots: dict[tuple, int] = {}
    vertices: list[tuple[int, int, int]] = []
    vertex_bones: list[int] = []
    vertex_blends: list = []
    faces: list[Face] = []
    collapsed = 0

    def vertex_slot(position, bone: int, blend=None) -> int:
        x, y, z = position
        # FLIP into engine space, then the >> 2 the mesh format stores.
        key = (
            int(round(x * scale / 4.0)),
            int(round(-y * scale / 4.0)),
            int(round(-z * scale / 4.0)),
        )
        for value in key:
            # -32768 is a perfectly good int16; only the positive end is short.
            if not -32768 <= value <= 32767:
                raise BuildError(
                    f"{model.path.name}: vertex {key} does not fit the int16 the "
                    f"vertex store uses, at scale {scale:.4g}"
                )
        # Two vertices at the same point but on different bones are two
        # vertices: they part company as soon as the bones do. The same goes
        # for two that share a point but not the set of bones pulling on them.
        weld = (key, bone, blend)
        slot = vertex_slots.get(weld)
        if slot is None:
            slot = len(vertices)
            vertex_slots[weld] = slot
            vertices.append(key)
            vertex_bones.append(bone)
            vertex_blends.append(blend)
        return slot

    for prim_index, prim in enumerate(primitives):
        label = f"{model.path.name}:{node.name}:prim{prim_index}"
        material = model.materials[prim.material] if prim.material is not None else None
        # The engine always culls back faces, so a shell with no consistent
        # outside has to be submitted both ways or half of it simply vanishes.
        # The decision is made per joint by the caller and is final here: it
        # used to be ANDed with the material's own double-sided flag, which only
        # glTF carries -- COLLADA, OBJ and FBX all report False -- so the whole
        # mechanism was dead for three formats out of four, and "double_sided":
        # true in pack.json did nothing at all.
        both_ways = bool(double_sided)
        for triangle in range(0, len(prim.indices), 3):
            source = prim.indices[triangle:triangle + 3]
            orders = [[source[0], source[2], source[1]]] if flip_winding else [list(source)]
            if both_ways:
                first = orders[0]
                orders.append([first[0], first[2], first[1]])
            for corners in orders:
                slots = tuple(
                    vertex_slot(placed[prim_index][c],
                                prim.bones[c] if prim.bones else joint.node,
                                prim.blends[c] if prim.blends else None)
                    for c in corners)
                if len(set(slots)) != 3:
                    collapsed += 1    # degenerate after welding, nothing to draw
                    continue
                uvs = None
                if prim.uvs is not None:
                    uvs = [prim.uvs[c] for c in corners]
                faces.append(Face(slots, prim.material, uvs))

    if len(vertices) > MAX_VERTS_PER_MESH:
        raise BuildError(
            f"{model.path.name}: joint {node.name!r} has {len(vertices)} welded vertices, "
            f"vCount is a uint8 and allows {MAX_VERTS_PER_MESH}"
        )
    return JointMesh(vertices, faces, vertex_bones, vertex_blends, collapsed)


def chain_span(indices, previous: int) -> tuple[int, int]:
    """The shortest int8 hop that can encode one face, and the rotation using it."""
    best = None
    for rotation in range(3):
        i0, i1, i2 = (indices[rotation], indices[(rotation + 1) % 3],
                      indices[(rotation + 2) % 3])
        span = max(abs(i0 - previous), abs(i1 - i0), abs(i2 - i1))
        if best is None or span < best[0]:
            best = (span, rotation)
    return best


def chain_faces(faces: list[Face], vertex_count: int):
    """Order the faces and number the vertices in one pass, keeping hops short.

    The format chains each face's corners as int8 deltas: the first corner is a
    step from the previous face's last corner, the other two step from the one
    before. So the encodable limit is not a vertex count but a reach -- no face
    may look further than 127 places back along the numbering.

    Ordering and numbering are therefore the same problem and are solved
    together. The walk starts at the lowest face and repeatedly takes whichever
    neighbouring face can be encoded with the shortest hop, numbering vertices
    as it first meets them. Choosing by the hop it actually produces, rather
    than by how many corners are shared, is what keeps dense meshes inside the
    range: a face that reaches back across the buffer is passed over while its
    neighbours are still cheap, and picked up later when the walk returns.

    Returns the renumbered faces in walk order, and the order to store the
    vertices in.
    """
    number_of: dict[int, int] = {}
    order: list[int] = []                      # original indices, in first-use order
    touching: dict[int, list[int]] = {}
    for position, face in enumerate(faces):
        for index in face.indices:
            touching.setdefault(index, []).append(position)

    remaining = set(range(len(faces)))
    frontier: set[int] = set()
    out: list[Face] = []
    previous = 0

    def numbering(face: Face) -> tuple[int, ...]:
        """What this face's corners would be numbered, were it taken now."""
        pending = len(order)
        result = []
        seen: dict[int, int] = {}
        for index in face.indices:
            if index in number_of:
                result.append(number_of[index])
            elif index in seen:
                result.append(seen[index])
            else:
                seen[index] = pending
                result.append(pending)
                pending += 1
        return tuple(result)

    while remaining:
        candidates = frontier & remaining
        if not candidates:
            candidates = remaining
        best = None
        for position in candidates:
            indices = numbering(faces[position])
            span, _ = chain_span(indices, previous)
            key = (span, sum(1 for i in faces[position].indices if i not in number_of),
                   min(indices), position)
            if best is None or key < best[0]:
                best = (key, position, indices)
        _, position, _ = best

        face = faces[position]
        for index in face.indices:
            if index not in number_of:
                number_of[index] = len(order)
                order.append(index)
        renumbered = tuple(number_of[i] for i in face.indices)
        span, rotation = chain_span(renumbered, previous)
        previous = renumbered[(rotation + 2) % 3]
        out.append(Face(renumbered, face.material, face.uvs))

        remaining.discard(position)
        frontier.discard(position)
        for index in face.indices:
            frontier.update(touching[index])
        frontier &= remaining

    for index in range(vertex_count):          # vertices no face refers to
        if index not in number_of:
            number_of[index] = len(order)
            order.append(index)
    return out, order


def encode_mesh(mesh: JointMesh, resolve_material, label: str) -> tuple[bytes, list[int]]:
    """Serialise one mesh block: header, triangles, then vertices.

    Returns the block and the permutation it applied, as old index per new slot.
    Anything held alongside the vertices -- which bone each follows, which ones
    several bones share -- has to be reordered by it or it describes the wrong
    vertices.
    """
    if mesh.vertices:
        xs = [v[0] for v in mesh.vertices]
        ys = [v[1] for v in mesh.vertices]
        zs = [v[2] for v in mesh.vertices]
        center = ((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2, (min(zs) + max(zs)) // 2)
        radius = max(
            int(math.ceil(math.sqrt(sum((v[i] - center[i]) ** 2 for i in range(3)))))
            for v in mesh.vertices
        )
    else:
        center, radius = (0, 0, 0), 0

    # Face order and vertex numbering are one problem -- see chain_faces -- and
    # are solved together. The cyclic rotation of each triangle is picked at
    # encode time below; rotation preserves winding.
    faces, order = chain_faces(mesh.faces, len(mesh.vertices))
    vertices = [mesh.vertices[index] for index in order]

    out = bytearray()
    out.extend(struct.pack(
        "<3hhHBBhhhh",
        center[0], center[1], center[2], min(radius, 32767), 0,
        len(vertices), 0,
        0,                  # quads: the converter emits triangles only
        len(faces),         # triangles
        0, 0,
    ))

    previous = 0
    for face in faces:
        span, rotation = chain_span(face.indices, previous)
        i0, i1, i2 = (face.indices[rotation], face.indices[(rotation + 1) % 3],
                      face.indices[(rotation + 2) % 3])
        deltas, last = (i0 - previous, i1 - i0, i2 - i1), i2
        if span > 127:
            raise ChainOverflow(
                f"{label}: face index delta {span} does not fit int8. A mesh whose "
                f"chain reaches too far is split across joints and rebuilt, down to "
                f"parts of {SAFE_VERTS_PER_MESH} vertices, where the largest possible "
                f"index difference is 127 and the chain cannot overflow. Reaching this "
                f"message means that floor did not hold, which is a bug in the "
                "converter rather than anything wrong with the model"
            )
        previous = last
        # The UV corners must follow the same rotation the vertices took.
        uvs = None
        if face.uvs is not None:
            uvs = [face.uvs[(rotation + i) % 3] for i in range(3)]
        flags = resolve_material(face.material, uvs, label)
        # indices[2] is the engine's "asr hack" slot; the third delta is last.
        out.extend(struct.pack("<4bHH", deltas[0], deltas[1], 0, deltas[2], flags, 0))

    for x, y, z in vertices:
        out.extend(struct.pack("<3h", x, y, z))
    return bytes(out), order


# --------------------------------------------------------------------------
# per-model conversion
# --------------------------------------------------------------------------

@dataclass
class ModelSpec:
    path: Path
    name: str
    slot: int
    frame_rate: int = DEFAULT_FRAME_RATE
    scale: float | str = "auto"
    flip_winding: bool | str = "auto"
    double_sided: bool | str = "auto"
    rotate: tuple[float, float, float] = (0.0, 0.0, 0.0)   # degrees, applied first


@dataclass
class BuiltModel:
    body: bytes
    spec: ModelSpec
    joints: int
    mask: int
    clips: int
    report: dict = field(default_factory=dict)
    # One byte per emitted vertex naming the joint it follows, block by block.
    # Empty when every vertex follows its own block, which is what the engine
    # does unaided.
    skin: bytes = b""
    # Variable-length records, in increasing vertex order, for the vertices the
    # source shares between bones.
    blends: bytes = b""
    blend_count: int = 0


def _decode_image(image: gltf_reader.Image):
    from io import BytesIO

    try:
        from PIL import Image as PilImage
    except ImportError as error:                       # pragma: no cover
        raise BuildError("Pillow is required to convert textured models") from error
    try:
        return PilImage.open(BytesIO(image.data)).convert("RGBA")
    except Exception as error:
        raise BuildError(f"cannot decode image {image.name!r}: {error}") from error


def build_model(spec: ModelSpec, glyphs: GlyphStrip, verbose: bool) -> BuiltModel:
    """Convert one model, splitting further if its index chain will not fit."""
    budget = SPLIT_BUDGET
    two_sided = spec.double_sided
    skirt = True
    while True:
        try:
            return build_model_at(spec, glyphs, verbose, budget, two_sided, skirt)
        except FaceBudgetExceeded:
            # Emitting open shells both ways doubles their faces. Dropping back
            # to one-sided is what a modeller would be told to do by hand, so it
            # is done here instead of refusing the model.
            if two_sided is not False:
                two_sided = False
                if verbose:
                    print("    note: two-sided shells would pass the renderer's "
                          "per-frame face budget; rebuilding one-sided")
            elif skirt:
                # Overlapping the seams is what keeps them shut when the bones
                # turn, so it is given up only once there is nothing else left.
                skirt = False
                if verbose:
                    print("    note: overlapping the seams would pass the renderer's "
                          "per-frame face budget; rebuilding with butt joints, which "
                          "will show a gap at a bent joint")
            else:
                raise
        except ChainOverflow:
            # The reach only shows once the mesh is built, so this is measured
            # rather than predicted. Under 128 vertices the largest possible
            # index difference is 127, so the retry cannot fail the same way.
            if budget <= SAFE_VERTS_PER_MESH:
                raise
            budget = SAFE_VERTS_PER_MESH
            if verbose:
                print(f"    note: a mesh reaches too far back for the int8 face "
                      f"chain; rebuilding with parts of at most {budget} vertices")


def build_model_at(spec: ModelSpec, glyphs: GlyphStrip, verbose: bool,
                   split_budget: int, double_sided=None,
                   skirt: bool = True) -> BuiltModel:
    model = load_model(spec.path, verbose, split_budget, skirt)
    joints = discover_joints(model)
    pre = rotation_matrix(spec.rotate)
    bind_world = node_world_matrices(model, {}, {}, 0.0, pre)
    node_of = {joint.node for joint in joints}

    # ---- scale -------------------------------------------------------------
    # Rest bounds in engine space, used both to pick a scale and to sanity-check
    # which way is up. They must be measured through the joint chain: a mesh's
    # own vertices are local, so a pre-rotation would not show in them.
    rest_world = evaluate_pose(model, joints, 1.0, {}, {}, 0.0, pre)
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    for slot, joint in enumerate(joints):
        matrix = rest_world[slot]
        # Measured before quantisation: at scale 1 a model authored in metres
        # rounds to a single point, and the auto-scale below would then be
        # picked from noise.
        for points in local_points(model, joint, bind_world):
            for point in points:
                x, y, z = point[0], -point[1], -point[2]   # FLIP into engine space
                for axis in range(3):
                    row = matrix[axis]
                    value = row[0] * x + row[1] * y + row[2] * z + row[3]
                    lo[axis] = min(lo[axis], value)
                    hi[axis] = max(hi[axis], value)
    spans = [hi[i] - lo[i] if hi[i] > -1e17 else 0.0 for i in range(3)]
    if spec.scale == "auto":
        extent = max(spans) or 1.0
        # Anything smaller than a sector is scaled up to one. The vertex store
        # keeps a quarter unit, so a model left at a tenth of a sector throws
        # away most of the precision it could have had; only a model already at
        # least that big is left alone, since scaling it down would lose shape.
        scale = AUTO_SCALE_TARGET / extent if extent < AUTO_SCALE_TARGET else 1.0
    else:
        scale = float(spec.scale)

    # Two separate int16s bound the scale, and they bite at different sizes.
    # Mesh vertices are stored local to their joint in quarter units, so that
    # one allows four times as much; joint offsets and the per-frame bounding
    # boxes are stored in whole engine units from the model's own origin, and
    # for anything sizeable that is the tighter of the two. Both are measured
    # here, before a single mesh is built, so the message can name the number to
    # write down instead of the first vertex that happened to go over. The first
    # offender is rarely the worst one, and a limit read off it fails again a
    # little lower down.
    reach = 0.0
    for joint in joints:
        for points in local_points(model, joint, bind_world):
            for point in points:
                reach = max(reach, abs(point[0]), abs(point[1]), abs(point[2]))
    origin = max((max(abs(lo[i]), abs(hi[i])) for i in range(3)
                  if hi[i] > -1e17), default=0.0)
    ceilings = []
    if reach:
        ceilings.append((131068.0 / reach, "the vertex store"))
    if origin:
        ceilings.append((32767.0 / origin, "the joint offsets and frame bounds"))
    if ceilings:
        allowed, binding = min(ceilings)
        if scale > allowed:
            # Rounded down, and a whisker under, so the printed value is one
            # that builds rather than one sitting exactly on the edge.
            usable = math.floor(allowed * 0.999 * 100) / 100
            raise BuildError(
                f"{spec.path.name}: scale {scale:.4g} is too large for this "
                f"model. {binding.capitalize()} hold int16, and this model "
                f"passes that at any scale above {allowed:.4g}. Give it "
                f"\"scale\": {usable:g} or less on its own entry in pack.json; "
                f"a model's own entry overrides the pack-wide default, so the "
                f"rest of the pack keeps theirs."
            )

    # Reported rather than judged: a quadruped is honestly longer than it is
    # tall, so no threshold separates that from a model exported Z-up. The
    # author reads the numbers and reaches for "rotate" if they look wrong.
    # Given in engine units, where a Tomb Raider sector is 1024, because the
    # authored units may be metres, centimetres or anything else.
    if verbose:
        sized = [span * scale for span in spans]
        print(f"    rest size X {sized[0]:.0f}  Y {sized[1]:.0f}  Z {sized[2]:.0f}"
              f" engine units, a sector being 1024"
              f"   (Y is up; use \"rotate\" in pack.json if this is on its side)")

    # One verdict per mesh; an explicit setting overrides them all. Meshes whose
    # facing cannot be measured follow the majority of those that can, so a model
    # stays self-consistent instead of being half inside out.
    if spec.flip_winding == "auto":
        # Read, never guessed: a closed shell is settled by its signed volume,
        # an open one by the normals it was saved with. A mesh that is both open
        # and stripped of its normals says nothing at all, and nothing here
        # invents an answer -- it follows the meshes that did say, or failing
        # that the format's counter-clockwise convention, and says so out loud.
        verdicts = [mesh_facing(model, joint.node) for joint in joints]
        votes = [flip for known, flip in verdicts if known]
        fallback = (sum(votes) * 2 > len(votes)) if votes else True
        windings = [flip if known else fallback for known, flip in verdicts]
        if verbose:
            silent = [joints[s].name for s, (known, _) in enumerate(verdicts) if not known]
            if silent:
                source = ("the meshes that do say" if votes
                          else "the format's counter-clockwise convention")
                print(f"    note: {len(silent)} of {len(joints)} meshes are open shells "
                      f"saved without normals, so their file says nothing about which side "
                      f"is out; their facing follows {source}. Set \"flip_winding\" in "
                      f"pack.json if a part shows its inside.")
    else:
        windings = [bool(spec.flip_winding)] * len(joints)

    # `double_sided` overrides the spec: the retry above sets it to False when
    # emitting both faces would pass the renderer's per-frame budget.
    wanted = spec.double_sided if double_sided is None else double_sided
    if wanted == "auto":
        # Only where it is needed: a closed shell has a real outside and a
        # measurable orientation, so drawing it twice would just double the face
        # count. An open one has neither, and half of it would vanish into the
        # backface cull -- which is what "faces broken into pieces" looks like.
        #
        # The material's own double-sided flag deliberately plays no part: only
        # glTF carries one, and it used to veto this decision, which left the
        # whole mechanism dead for COLLADA, OBJ and FBX.
        # A repaired mesh already agrees with itself, so its open pieces -- the
        # chunks rigid binding cuts out of it -- need no second copy. Only a
        # surface that could not be repaired is drawn both ways.
        trusted = bool(model.json.get("winding_trusted"))
        two_sided = [not trusted and not mesh_facing(model, joint.node)[0]
                     for joint in joints]
    else:
        two_sided = [bool(wanted)] * len(joints)

    # ---- materials: colours, images, atlas ---------------------------------
    used_materials = set()
    for joint in joints:
        mesh_index = model.nodes[joint.node].mesh
        if mesh_index is None:
            continue
        for prim in model.meshes[mesh_index]:
            used_materials.add(prim.material)

    flat_colors: dict[int, tuple[int, int, int]] = {}
    image_of: dict[int, int] = {}
    for material_index in sorted(m for m in used_materials if m is not None):
        material = model.materials[material_index]
        if material.base_color_texture is None:
            rgb = snap5(tuple(round(c * 255) for c in material.base_color[:3]))
            flat_colors[material_index] = rgb
        else:
            image_of[material_index] = material.base_color_texture
    if None in used_materials:
        flat_colors[-1] = snap5((200, 200, 200))       # untextured, unmaterialed

    if verbose:
        borrowed = [model.materials[i].name
                    for i in sorted(m for m in used_materials if m is not None)
                    if getattr(model.materials[i], "from_emissive", False)]
        if borrowed:
            print(f"    note: {len(borrowed)} material(s) keep their image in the "
                  "emissive slot rather than the base colour, which is how an unlit "
                  "model is usually wired; the target has no lighting, so that image "
                  f"is what gets drawn ({', '.join(repr(n) for n in borrowed[:3])})")
        # A model whose every surface is flat black builds, loads, and shows
        # nothing at all against the viewer's black background. Only worth
        # saying when it is the whole model: a dark trim colour beside a
        # textured body is just a dark trim colour.
        if not image_of and flat_colors and all(max(rgb) <= 24 for rgb in flat_colors.values()):
            print("    note: every material of this model is an untextured colour "
                  "close to black, so it will be invisible against the viewer's "
                  "black background. In Blender, wire the image into Base Color "
                  "or Emission, or give the material a lighter colour.")

    decoded = {index: _decode_image(model.images[index]) for index in set(image_of.values())}

    # How far outside 0..1 does each image actually get used?
    uv_span: dict[int, tuple[float, float, float, float]] = {}
    for joint in joints:
        mesh_index = model.nodes[joint.node].mesh
        if mesh_index is None:
            continue
        for prim in model.meshes[mesh_index]:
            image = image_of.get(prim.material)
            if image is None or not prim.uvs:
                continue
            us = [uv[0] for uv in prim.uvs]
            vs = [uv[1] for uv in prim.uvs]
            have = uv_span.get(image)
            box = (min(us), max(us), min(vs), max(vs))
            uv_span[image] = box if have is None else (
                min(have[0], box[0]), max(have[1], box[1]),
                min(have[2], box[2]), max(have[3], box[3]),
            )

    reserved: list[tuple[int, int, int]] = [(0, 0, 0)]
    seen = {(0, 0, 0)}
    for rgb in glyphs.colors:
        if rgb not in seen:
            seen.add(rgb)
            reserved.append(rgb)
    glyph_palette_index = {}
    for source, slot in glyphs.index_map.items():
        glyph_palette_index[source] = reserved.index(glyphs.colors[slot])
    for rgb in flat_colors.values():
        if rgb not in seen:
            seen.add(rgb)
            reserved.append(rgb)
    if len(reserved) > 256:
        raise BuildError(
            f"{spec.path.name}: {len(reserved)} reserved colours (glyphs plus flat materials) "
            "exceed the 256-entry palette"
        )

    budget = 256 - len(reserved)
    palette = list(reserved)
    if decoded and budget > 0:
        from PIL import Image as PilImage

        total_h = sum(img.height for img in decoded.values())
        max_w = max(img.width for img in decoded.values())
        strip = PilImage.new("RGB", (max_w, total_h), (0, 0, 0))
        y = 0
        for img in decoded.values():
            strip.paste(img.convert("RGB"), (0, y))
            y += img.height
        quantized = strip.quantize(colors=min(budget, 256), method=PilImage.Quantize.MEDIANCUT)
        raw = quantized.getpalette() or []
        for i in range(0, len(raw), 3):
            rgb = snap5((raw[i], raw[i + 1], raw[i + 2]))
            if rgb not in seen and len(palette) < 256:
                seen.add(rgb)
                palette.append(rgb)
    while len(palette) < 256:
        palette.append((0, 0, 0))

    color_cache: dict[tuple[int, int, int], int] = {}

    def palette_index(rgb: tuple[int, int, int]) -> int:
        key = snap5(rgb)
        hit = color_cache.get(key)
        if hit is None:
            hit = nearest_index(palette, key)
            color_cache[key] = hit
        return hit

    atlas = Atlas(reserved_rows=GLYPH_REGION_H)
    for y, row in enumerate(glyphs.pixels):
        atlas.pages[0][y] = [glyph_palette_index.get(index, 0) if index else 0 for index in row]

    placements: dict[int, Placement] = {}
    for image_index, img in decoded.items():
        width, height = img.size
        span = uv_span.get(image_index, (0.0, 1.0, 0.0, 1.0))
        u0 = math.floor(min(span[0], 0.0))
        u1 = max(math.ceil(span[1]), u0 + 1)
        v0 = math.floor(min(span[2], 0.0))
        v1 = max(math.ceil(span[3]), v0 + 1)
        repeats_u, repeats_v = u1 - u0, v1 - v0
        if width * repeats_u > TILE_DIM or height * repeats_v > TILE_DIM:
            raise BuildError(
                f"{spec.path.name}: {model.images[image_index].name!r} is used across "
                f"{repeats_u}x{repeats_v} repeats of a {width}x{height} image, which needs "
                f"{width * repeats_u}x{height * repeats_v} texels and will not fit a "
                f"{TILE_DIM}x{TILE_DIM} page. Keep the UVs inside 0..1, or use a smaller image."
            )
        region = atlas.place(width * repeats_u, height * repeats_v)
        pixels = img.load()
        base = [[0 if pixels[x, y][3] < 128 else palette_index(pixels[x, y][:3])
                 for x in range(width)] for y in range(height)]
        rows = [base[y % height] * repeats_u for y in range(height * repeats_v)]
        atlas.blit(region, rows)
        placements[image_index] = Placement(region, u0, v0, width, height)

    # ---- object textures ---------------------------------------------------
    textures: list[bytes] = []
    texture_slots: dict[bytes, int] = {}

    def samples_colour_key(region: Region, corners) -> bool:
        """Does this face touch a transparent texel?

        Index 0 is the colour key, and the atlas already carries it wherever the
        source image was transparent. Reading it back is more reliable than
        trusting the material's alphaMode: an exporter that bakes transparency
        into the image while leaving the material OPAQUE -- which is what the
        Tomb Raider extraction does -- would otherwise turn every grille and
        every railing into a solid slab.
        """
        page = atlas.pages[region.page]
        x0 = min(c[0] for c in corners)
        x1 = max(c[0] for c in corners)
        y0 = min(c[1] for c in corners)
        y1 = max(c[1] for c in corners)
        for y in range(y0, y1 + 1):
            row = page[y]
            if any(index == 0 for index in row[x0:x1 + 1]):
                return True
        return False

    def texture_record(placement: Placement, uvs):
        region = placement.region
        corners = []
        for u, v in uvs:
            # Nearest-neighbour sampling: the texel a UV lands in is floor(uv * size),
            # the same rule a GPU applies. Scaling by size - 1 instead drifts by a
            # texel across the second half of the image, which on a packed page
            # bleeds the neighbouring texture into the face. The offset by u0/v0
            # moves the UV into the baked block of repeats.
            x = math.floor((u - placement.u0) * placement.width)
            y = math.floor((v - placement.v0) * placement.height)
            x = min(max(x, 0), region.w - 1)
            y = min(max(y, 0), region.h - 1)
            corners.append((region.x + x, region.y + y))
        while len(corners) < 4:
            corners.append(corners[-1])
        for axis in (0, 1):
            span = max(c[axis] for c in corners) - min(c[axis] for c in corners)
            if span > MAX_UV_SPAN:
                raise BuildError(
                    f"a face spans {span} texels on {'UV'[axis]}; the engine clamps at "
                    f"{MAX_UV_SPAN}. Split the face or shrink its UV island."
                )
        packed = [((x & 0xFF) << 24) | ((y & 0xFF) << 8) for x, y in corners]
        record = struct.pack(
            "<III",
            region.page * TILE_SIZE,
            packed[0] | (packed[1] >> 8),
            packed[2] | (packed[3] >> 8),
        )
        slot = texture_slots.get(record)
        if slot is None:
            slot = len(textures)
            texture_slots[record] = slot
            textures.append(record)
        return slot, samples_colour_key(region, corners)

    def resolve_material(material_index, uvs, label: str) -> int:
        if material_index is None or material_index in flat_colors:
            key = -1 if material_index is None else material_index
            return (FACE_TYPE_F << FACE_TYPE_SHIFT) | palette_index(flat_colors[key])
        material = model.materials[material_index]
        if uvs is None:
            raise BuildError(f"{label}: material {material.name!r} is textured but the mesh has no UVs")
        placement = placements[image_of[material_index]]
        slot, keyed = texture_record(placement, uvs)
        keyed = keyed or material.alpha_mode in ("MASK", "BLEND")
        kind = FACE_TYPE_FTA if keyed else FACE_TYPE_FT
        return (kind << FACE_TYPE_SHIFT) | slot

    # ---- meshes ------------------------------------------------------------
    # A zero in the mesh-offset table means "this joint has no mesh", so no real
    # block may sit at offset zero or the two become indistinguishable. Tomb
    # Raider's own data does put its first mesh there and relies on joint zero
    # always being present; a custom model's root may legitimately be an empty
    # armature joint, so four bytes are spent to keep the meaning unambiguous.
    anchors = seam_anchors(model, joints, scale, pre, bind_world)

    node_slot = {joint.node: slot for slot, joint in enumerate(joints)}

    # One byte per emitted vertex naming the joint it follows, laid out block by
    # block in the order the mesh table lists them. The engine never reads it --
    # its mesh blocks are unchanged -- but the viewer does, to pose the model a
    # vertex at a time instead of a block at a time.
    skin_blob = bytearray()
    skin_needed = False

    # And, for the few vertices a source shares between bones, every influence
    # it names. At the bind pose a blend and its heaviest bone agree exactly, so
    # this changes nothing until the skeleton moves -- which is the whole point.
    blend_blob = bytearray()
    blend_count = 0
    vertex_base = 0

    mesh_data = bytearray(4)
    mesh_offsets: list[int] = []
    joint_bounds: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    visible_mask = 0
    total_faces = 0
    collapsed_faces = 0
    for slot, joint in enumerate(joints):
        mesh = build_joint_mesh(model, joint, scale, windings[slot], bind_world,
                                two_sided[slot], anchors.get(slot))
        joint_bounds.append(mesh.aabb())
        total_faces += len(mesh.faces)
        collapsed_faces += mesh.collapsed
        if not mesh.faces:
            # A joint with no geometry keeps its place in the hierarchy but is
            # marked absent the way TR1 does it: a zero mesh offset plus a clear
            # bit in the catalog's visibility mask. That holds for the root as
            # much as for any other joint -- an armature root carries the pose,
            # not necessarily a mesh.
            mesh_offsets.append(0)
            continue
        visible_mask |= 1 << slot
        while len(mesh_data) % 4:
            mesh_data.append(0)
        mesh_offsets.append(len(mesh_data))
        block, order = encode_mesh(
            mesh, resolve_material, f"{spec.path.name}:{model.nodes[joint.node].name}")
        mesh_data.extend(block)

        # The encoder renumbers the vertices to keep its index chain inside an
        # int8, so everything that runs parallel to them is renumbered with it.
        bones_of = mesh.bones or [joint.node] * len(mesh.vertices)
        for old in order:
            follows = node_slot.get(bones_of[old], slot)
            if follows != slot:
                skin_needed = True
            skin_blob.append(follows)

        blends_of = mesh.blends or []
        # Collected first and appended in vertex order: the viewer walks the
        # records and the vertices together in one pass, which only works if
        # both are ascending, and the renumbering above does not preserve that.
        block_records: list[tuple[int, bytes]] = []
        for index, old in enumerate(order):
            blend = blends_of[old] if old < len(blends_of) else None
            if not blend:
                continue
            shares = [share for _, share, _ in blend]
            # Quantised to sum to exactly 256, so the target blends with a shift
            # instead of a division, and loses nothing to the scale. Each keeps
            # at least one 256th, which also guarantees no single weight reaches
            # 256 and overflows its byte.
            bytes_ = [max(1, min(255, int(round(share * 256.0)))) for share in shares]
            drift = 256 - sum(bytes_)
            heaviest = sorted(range(len(bytes_)), key=lambda i: -bytes_[i])
            for i in heaviest:                   # give the drift to the heaviest
                step = max(-(bytes_[i] - 1), min(255 - bytes_[i], drift))
                bytes_[i] += step
                drift -= step
                if drift == 0:
                    break
            if drift or min(bytes_) < 1:
                continue                        # degenerate weights; keep the bone
            record = bytearray(struct.pack("<HBB", vertex_base + index, len(blend), 0))
            for (node_index, _, point), weight in zip(blend, bytes_):
                x = int(round(point[0] * scale / 4.0))
                y = int(round(-point[1] * scale / 4.0))
                z = int(round(-point[2] * scale / 4.0))
                if not all(-32768 <= v <= 32767 for v in (x, y, z)):
                    record = None
                    break
                record.extend(struct.pack("<BBhhh",
                                          node_slot.get(node_index, slot), weight, x, y, z))
            if record is None:
                continue
            block_records.append((index, bytes(record)))
            skin_needed = True

        for _, record in sorted(block_records):
            blend_blob.extend(record)
            blend_count += 1
        vertex_base += len(order)
    while len(mesh_data) % 4:
        mesh_data.append(0)
    if not visible_mask:
        raise BuildError(f"{spec.path.name}: no joint has drawable geometry")
    if not skin_needed:
        # Every vertex follows the block it sits in, which is what the engine
        # already does on its own; the table would say nothing.
        skin_blob = bytearray()
        blend_blob = bytearray()
        blend_count = 0
    elif verbose:
        print(f"    note: {len(skin_blob)} vertices carry the bone they follow, so a "
              "triangle spanning two bones stretches between them instead of being "
              "frozen onto one")
        if blend_count:
            print(f"    note: {blend_count} of them are shared between bones and keep "
                  "every influence the file names, blended at draw time")

    if len(textures) > 1536:
        raise BuildError(f"{spec.path.name}: {len(textures)} texture records, the engine allows 1536")
    if total_faces > MAX_FACES_PER_FRAME:
        raise FaceBudgetExceeded(
            f"{spec.path.name}: {total_faces} faces, and the renderer draws at most "
            f"{MAX_FACES_PER_FRAME} per frame -- the rest would silently vanish. "
            "Simplify the mesh in the modeller."
        )
    if verbose and total_faces > MAX_FACES_PER_FRAME * 0.8:
        print(f"    note: {total_faces} faces, within {MAX_FACES_PER_FRAME} but close to it; "
              "anything past that budget is dropped at draw time.")

    # A scale too small for the model rounds neighbouring vertices onto the same
    # quarter-unit and the triangles between them stop existing. It is silent
    # otherwise: the build succeeds and the viewer shows a ruined model, which
    # is a hard thing to attribute to a number in pack.json.
    if collapsed_faces:
        share = 100.0 * collapsed_faces / (total_faces + collapsed_faces)
        if share >= 2.0:
            where = "\"scale\" for this model in pack.json" if spec.scale != "auto" else "the model"
            print(f"    WARNING: {collapsed_faces} of {total_faces + collapsed_faces} "
                  f"triangles ({share:.0f}%) collapsed to a line or a point when the "
                  f"vertices were rounded at scale {scale:.4g}. The shape is being lost. "
                  f"Raise {where}: at this scale the whole model spans "
                  f"{max(span * scale for span in spans) / 4.0:.0f} stored steps.")
        elif verbose:
            print(f"    note: {collapsed_faces} triangle(s) collapsed when rounded at "
                  f"scale {scale:.4g}, {share:.1f}% of the model")

    # ---- nodes -------------------------------------------------------------
    # The node table holds each joint's rest offset from its parent joint.
    bind = joint_locals(model, joints, scale, {}, {}, 0.0, pre)
    nodes_blob = bytearray()
    for slot, joint in enumerate(joints[1:], start=1):
        position, _ = decompose(bind[slot], f"{spec.path.name}:{joint.name}")
        nodes_blob.extend(struct.pack(
            "<3hH",
            int(round(position[0])), int(round(position[1])), int(round(position[2])),
            joint.flags,
        ))

    # ---- animations --------------------------------------------------------
    animations, frames, clips = build_animations(
        model, joints, scale, spec, joint_bounds, pre, verbose
    )

    # ---- assemble ----------------------------------------------------------
    body = write_pkd_body(PkdParts(
        palette=b"".join(struct.pack("<H", to_bgr555(c)) for c in palette),
        lightmap=build_lightmap(palette),
        tiles=atlas.to_bytes(),
        mesh_data=bytes(mesh_data),
        mesh_offsets=mesh_offsets,
        animations=animations,
        nodes=bytes(nodes_blob),
        frames=frames,
        models=struct.pack("<BbHHH", spec.slot, len(joints), 0, 0, 0),
        textures=textures,
        glyph_sprites=glyphs.records,
    ))

    report = body_stats(body)
    report.update({
        "joints": len(joints),
        "faces": total_faces,
        "clips": clips,
        # Exact, not rounded: the checker rebuilds its reference geometry with
        # this value, and a scale of 1.0009775 rounded to 1.001 welds vertices
        # differently. Rounding belongs in the display, not in the record.
        "scale": scale,
        "collapsed_faces": collapsed_faces,
        "split_budget": split_budget,
        "seam_overlap": skirt,
        "skinned_vertices": len(skin_blob),
        "blended_vertices": blend_count,
        "reserved_colors": len(reserved),
        "atlas_pages": len(atlas.pages),
        "drawable_joints": bin(visible_mask).count("1"),
        "flip_winding": windings,
        "double_sided": two_sided,
        "winding_measured": sum(1 for closed, _ in
                                (mesh_facing(model, j.node) for j in joints) if closed),
    })
    return BuiltModel(body, spec, len(joints), visible_mask, clips, report,
                      bytes(skin_blob), bytes(blend_blob), blend_count)


def sample_rotation(sampler, time):
    """Evaluate a rotation sampler, slerping between the surrounding keys."""
    times = sampler.times
    if not times:
        return None
    if sampler.interpolation == "STEP" or len(times) == 1:
        index = 0
        for i, t in enumerate(times):
            if t <= time:
                index = i
        return tuple(sampler.values[index])
    if time <= times[0]:
        return tuple(sampler.values[0])
    if time >= times[-1]:
        return tuple(sampler.values[-1])
    hi = 1
    while hi < len(times) and times[hi] < time:
        hi += 1
    lo = hi - 1
    span = times[hi] - times[lo] or 1.0
    t = (time - times[lo]) / span
    stride = len(sampler.values) // len(times)
    if stride == 3:                      # CUBICSPLINE: keep the value keys
        return quat_slerp(tuple(sampler.values[lo * 3 + 1]),
                          tuple(sampler.values[hi * 3 + 1]), t)
    return quat_slerp(tuple(sampler.values[lo]), tuple(sampler.values[hi]), t)


def sample_vec3(sampler, time, fallback):
    value = sample_rotation(sampler, time) if sampler else None
    if value is None:
        return fallback
    return value[:3]


def track_channels(model, animation, node_slot, spec, verbose):
    """Split one glTF animation into the rotation and root-translation it can use."""
    rotation: dict[int, gltf_reader.Sampler] = {}
    translation: dict[int, gltf_reader.Sampler] = {}
    ignored = set()
    stretched = set()
    for channel in animation.channels:
        if channel.node not in node_slot:
            continue
        sampler = animation.samplers[channel.sampler]
        if channel.path == "rotation":
            rotation[channel.node] = sampler
        elif channel.path == "translation":
            if node_slot[channel.node] == 0:
                translation[channel.node] = sampler
            elif len({tuple(v) for v in sampler.values}) > 1:
                # Only joint 0 has a per-frame position; the rest live in the
                # static node table.
                ignored.add(channel.node)
        elif channel.path == "scale" and len({tuple(v) for v in sampler.values}) > 1:
            # Dropped rather than refused, exactly like an animated position.
            # The engine has no per-joint scale, so the squash cannot be kept
            # either way; refusing the model would throw away the rotation as
            # well, which is nearly all of the movement. A rig with a little
            # squash on one wrist is not a model anyone wants turned away.
            stretched.add(channel.node)
    if verbose:
        if ignored:
            names = ", ".join(sorted(model.nodes[n].name for n in ignored))
            print(f"    note: {animation.name!r} animates the position of {names}; "
                  "the engine keeps joint positions static, so only rotation is kept")
        if stretched:
            names = ", ".join(sorted(model.nodes[n].name for n in stretched))
            print(f"    note: {animation.name!r} scales {names}; the engine has no "
                  "per-joint scale, so the stretch is dropped and the rest of the "
                  "motion is kept")
    duration = 0.0
    for sampler in list(rotation.values()) + list(translation.values()):
        if sampler.times:
            duration = max(duration, sampler.times[-1])
    return rotation, translation, duration


def build_animations(model, joints, scale, spec, joint_bounds, pre, verbose):
    """Resample every glTF track onto the engine's fixed keyframe interval."""
    node_slot = {joint.node: slot for slot, joint in enumerate(joints)}
    rate = max(1, int(spec.frame_rate))

    tracks = []
    for animation in model.animations:
        rotation, translation, duration = track_channels(
            model, animation, node_slot, spec, verbose
        )
        tracks.append((animation, rotation, translation, duration))

    if not tracks:
        tracks = [(None, {}, {}, 0.0)]

    anim_records = bytearray()
    frame_blob = bytearray()
    # TR1 numbers frames on a single rising tick cursor across a model's clips.
    # Starting at 1 also keeps getViewerFrames from clamping the final segment,
    # which it does whenever frameBegin is zero.
    cursor = 1

    for animation, rotation, translation, duration in tracks:
        keyframes = max(1, int(math.ceil(duration * TICKS_PER_SECOND / rate)) + 1)
        frame_offset = len(frame_blob)
        for key in range(keyframes):
            time = key * rate / TICKS_PER_SECOND

            locals_ = joint_locals(model, joints, scale, rotation, translation, time, pre)
            position, angles = decompose(locals_[0], f"{spec.path.name}:root")
            packed = [pack_angles(angles)]
            world = [locals_[0]]
            for slot, joint in enumerate(joints[1:], start=1):
                _, joint_angles = decompose(locals_[slot],
                                            f"{spec.path.name}:{joint.name}")
                packed.append(pack_angles(joint_angles))
                world.append(mat_mul(world[joint.parent], locals_[slot]))

            # The viewer frames the camera on this box, and rootNeutralBounds
            # subtracts frame.pos from it, so it is stored in the same space as
            # the root translation.
            lo = [32767] * 3
            hi = [-32768] * 3
            for slot, matrix_j in enumerate(world):
                (ax, ay, az), (bx, by, bz) = joint_bounds[slot]
                for cx in (ax, bx):
                    for cy in (ay, by):
                        for cz in (az, bz):
                            for axis in range(3):
                                row = matrix_j[axis]
                                value = row[0] * cx + row[1] * cy + row[2] * cz + row[3]
                                lo[axis] = min(lo[axis], value)
                                hi[axis] = max(hi[axis], value)

            def clamp16(value: float) -> int:
                return int(min(max(round(value), -32768), 32767))

            frame_blob.extend(struct.pack(
                "<6h",
                clamp16(lo[0]), clamp16(hi[0]),
                clamp16(lo[1]), clamp16(hi[1]),
                clamp16(lo[2]), clamp16(hi[2]),
            ))
            frame_blob.extend(struct.pack(
                "<3hH",
                clamp16(position[0]), clamp16(position[1]), clamp16(position[2]),
                len(joints),
            ))
            for value in packed:
                frame_blob.extend(struct.pack("<I", value))

        frame_end = cursor + (keyframes - 1) * rate
        anim_records.extend(struct.pack(
            "<IBBHiiHHHHHHHH",
            frame_offset, rate, 0, 0, 0, 0,
            cursor, frame_end, len(anim_records) // 32, cursor,
            0, 0, 0, 0,
        ))
        cursor = frame_end + 1

    return bytes(anim_records), bytes(frame_blob), len(tracks)


# --------------------------------------------------------------------------
# pack assembly
# --------------------------------------------------------------------------

def sanitize_name(raw: str) -> str:
    text = "".join(c if c.isalnum() else "_" for c in raw.upper())
    text = text.strip("_") or "MODEL"
    return text[:NAME_ENTRY_SIZE - 1]


def collect_specs(folder: Path, defaults: dict) -> list[ModelSpec]:
    config_path = folder / "pack.json"
    config = {}
    if config_path.is_file():
        # utf-8-sig, not utf-8: Notepad and PowerShell both write a byte order
        # mark, and a pack.json edited by hand would otherwise stop the build
        # with a decoder traceback rather than a sentence.
        try:
            config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as error:
            raise BuildError(
                f"{config_path} is not valid JSON: {error}. A trailing comma after the "
                "last entry is the usual cause."
            ) from error
    merged = dict(defaults)
    merged.update(config.get("defaults", {}))
    entries = config.get("models")

    if entries is None:
        # One level of subfolders is searched too: a COLLADA file keeps its
        # textures beside it as separate files, so such a model travels as a
        # folder rather than as one self-contained file.
        found = [p for p in folder.iterdir() if p.suffix.lower() in MODEL_SUFFIXES]
        for child in folder.iterdir():
            if child.is_dir():
                found.extend(p for p in child.iterdir()
                             if p.suffix.lower() in MODEL_SUFFIXES)
        files = sorted(found, key=lambda p: str(p.relative_to(folder)).lower())
        entries = [{"file": str(p.relative_to(folder)).replace("\\", "/")}
                   for p in files]

    specs: list[ModelSpec] = []
    used_slots: set[int] = set()
    for order, entry in enumerate(entries):
        path = (folder / entry["file"]).resolve()
        if not path.is_file():
            raise BuildError(f"model not found: {path}")
        slot = entry.get("slot")
        if slot is None:
            slot = order
        if not 0 <= slot < 191:
            raise BuildError(f"{path.name}: slot {slot} is outside 0..190")
        if slot in used_slots:
            raise BuildError(f"{path.name}: slot {slot} is already taken")
        used_slots.add(slot)
        specs.append(ModelSpec(
            path=path,
            name=sanitize_name(entry.get("name", path.stem)),
            slot=slot,
            frame_rate=int(entry.get("frame_rate", merged.get("frame_rate", DEFAULT_FRAME_RATE))),
            scale=entry.get("scale", merged.get("scale", "auto")),
            flip_winding=entry.get("flip_winding", merged.get("flip_winding", "auto")),
            double_sided=entry.get("double_sided", merged.get("double_sided", "auto")),
            rotate=tuple(entry.get("rotate", merged.get("rotate", (0.0, 0.0, 0.0)))),
        ))
    if not specs:
        raise BuildError(
            f"no model found in {folder}; expected one of "
            + ", ".join(MODEL_SUFFIXES)
        )
    return specs


def assemble_pack(models: list[BuiltModel]) -> tuple[bytes, dict]:
    models = sorted(models, key=lambda m: m.spec.slot)
    source_count = len(models)
    model_count = len(models)

    skinned = any(built.skin for built in models)
    flags = PACK_FLAG_NAMES | (PACK_FLAG_SKIN if skinned else 0)

    source_table = PACK_HEADER_SIZE_V4
    model_table = source_table + source_count * SOURCE_ENTRY_SIZE
    name_table = model_table + model_count * MODEL_ENTRY_SIZE
    skin_table = name_table + model_count * NAME_ENTRY_SIZE
    bodies_at = (skin_table + (model_count * SKIN_ENTRY_SIZE if skinned else 0) + 3) & ~3

    output = bytearray(bodies_at)
    offsets = []
    for built in models:
        while len(output) % 4:
            output.append(0)
        offsets.append(len(output))
        output.extend(built.body)

    # The skin tables sit after the bodies: they are read once when a model is
    # selected, never during a frame, so they do not need to be near anything.
    skin_at, blend_at = [], []
    for built in models:
        while len(output) % 4:
            output.append(0)
        skin_at.append(len(output))
        output.extend(built.skin)
        while len(output) % 4:
            output.append(0)
        blend_at.append(len(output))
        output.extend(built.blends)

    struct.pack_into(
        "<4sIIHHIIII", output, 0, PACK_MAGIC, PACK_VERSION, len(output),
        source_count, model_count, source_table, model_table,
        flags, name_table,
    )
    struct.pack_into("<I", output, 32, skin_table if skinned else 0)
    if skinned:
        for index, built in enumerate(models):
            struct.pack_into("<IIII", output, skin_table + index * SKIN_ENTRY_SIZE,
                             skin_at[index] if built.skin else 0, len(built.skin),
                             blend_at[index] if built.blends else 0, built.blend_count)
    for index, (built, offset) in enumerate(zip(models, offsets)):
        entry = source_table + index * SOURCE_ENTRY_SIZE
        label = built.spec.name.encode("ascii")[:8]
        output[entry:entry + 8] = label + b"\0" * (8 - len(label))
        struct.pack_into("<II", output, entry + 8, offset, len(built.body))

        row = model_table + index * MODEL_ENTRY_SIZE
        struct.pack_into("<BBBBHHI", output, row,
                         built.spec.slot, index, built.joints, 0, built.clips, 0, built.mask)

        name = built.spec.name.encode("ascii")
        slot = name_table + index * NAME_ENTRY_SIZE
        output[slot:slot + NAME_ENTRY_SIZE] = name + b"\0" * (NAME_ENTRY_SIZE - len(name))

    report = {
        "format": "AVP1",
        "version": PACK_VERSION,
        "flags": flags,
        "producer": "build_custom_pack.py",
        "output_bytes": len(output),
        "models": {built.spec.name: built.report for built in models},
    }
    return bytes(output), report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, type=Path,
                        help="folder holding the custom .glb, .gltf or .dae models")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--glyphs", required=True, type=Path,
                        help="PKD to borrow the viewer glyph strip from (TITLE.PKD)")
    parser.add_argument("--frame-rate", type=int, default=DEFAULT_FRAME_RATE,
                        help="ticks between resampled keyframes (default 2, i.e. 15 Hz)")
    parser.add_argument("--scale", default="auto",
                        help="world scale, or 'auto' to fit one TR sector")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    try:
        glyphs = load_glyph_strip(args.glyphs.resolve())
        specs = collect_specs(
            args.models.resolve(),
            {"frame_rate": args.frame_rate, "scale": args.scale},
        )
        built = []
        skipped: list[tuple[str, str]] = []
        for spec in specs:
            if verbose:
                print(f"  {spec.path.name} -> slot {spec.slot} {spec.name!r}")
            try:
                model = build_model(spec, glyphs, verbose)
            except Exception as error:
                # A folder is a working collection, not one artefact: a file
                # just dropped in to see what happens should not take the nine
                # that already work down with it. Caught broadly on purpose --
                # a truncated or mislabelled file can fail anywhere in a parser,
                # as a JSON, struct, XML or unicode error, and none of those
                # deserve a traceback in place of the ROM. The type is kept in
                # the message so a genuine defect is still recognisable.
                reason = str(error) or error.__class__.__name__
                if not isinstance(error, (BuildError, gltf_reader.GltfError)):
                    reason = f"{error.__class__.__name__}: {reason}"
                skipped.append((spec.path.name, reason))
                print(f"    SKIPPED: {reason}", file=sys.stderr)
                continue
            built.append(model)
            if verbose:
                r = model.report
                print(f"    {r['joints']} joints, {r['faces']} faces, {r['clips']} clips, "
                      f"{r['textures']} textures, {r['tiles']} pages, "
                      f"scale {r['scale']:.4g}, {r['bytes']} bytes")
        if not built:
            raise BuildError(
                f"none of the {len(specs)} model(s) in {args.models} could be converted; "
                "the reasons are listed above"
            )
        pack, report = assemble_pack(built)
        report["skipped"] = [{"file": name, "reason": why} for name, why in skipped]
    except (BuildError, gltf_reader.GltfError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pack)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{args.output.name}: {len(pack)} bytes, {len(built)} custom models")
    if skipped:
        print(f"{len(skipped)} model(s) skipped, the ROM was built without them:")
        for name, why in skipped:
            print(f"  - {name}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
