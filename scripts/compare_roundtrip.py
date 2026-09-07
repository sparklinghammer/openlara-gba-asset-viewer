#!/usr/bin/env python3
"""Render the same model through both producers and measure the difference.

The Tomb Raider path and the glTF path start from the same original geometry, so
a model converted from an extracted GLB should draw the same silhouette as the
one the TR1 producer emits. Rendering both offline with the engine's own
pipeline turns "looks about right on an emulator" into a number.

Silhouette agreement is the meaningful figure: it exercises the hierarchy, the
stack flags, the angle packing and the backface test all at once. Colour is
expected to differ, because each custom body quantises its own 256-entry palette
independently of Tomb Raider's.
"""

from __future__ import annotations

import argparse
import math
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_custom_pack as B
from check_custom_body import decode_mesh, find_model, model_extent, model_record
from preview_custom_body import FRAME_HEIGHT, FRAME_WIDTH, render

SOURCE_ENTRY_SIZE = 16
MODEL_ENTRY_SIZE = 12


def read_item_types(common_h: Path) -> list[str]:
    text = common_h.read_text(encoding="utf-8")
    block = text.split("#define ITEM_TYPES(E)", 1)[1].split("enum ItemType", 1)[0]
    return re.findall(r"\bE\(\s*([A-Z0-9_]+)\s*\)", block)


def pack_bodies(pack: bytes) -> list[bytes]:
    source_count = struct.unpack_from("<H", pack, 12)[0]
    source_table = struct.unpack_from("<I", pack, 16)[0]
    bodies = []
    for index in range(source_count):
        entry = source_table + index * SOURCE_ENTRY_SIZE
        offset, size = struct.unpack_from("<II", pack, entry + 8)
        bodies.append(pack[offset:offset + size])
    return bodies


def find_in_pack(pack: bytes, item_type: int):
    """Return (body, model_index) for the source that owns this item type."""
    model_count = struct.unpack_from("<H", pack, 14)[0]
    model_table = struct.unpack_from("<I", pack, 20)[0]
    bodies = pack_bodies(pack)
    for index in range(model_count):
        row = struct.unpack_from("<BBBBHHI", pack, model_table + index * MODEL_ENTRY_SIZE)
        if row[0] == item_type:
            body = bodies[row[1]]
            return body, find_model(body, item_type)
    return None, None


def shape_signature(body: bytes, model_index: int):
    """Per-joint vertex count and local bounds, in stored units.

    Both producers deduplicate a type across sixteen levels, but by different
    rules, so the same name can name two different props -- one level's DOOR_2
    is a grille, another's is a carved slab. Comparing renders is only meaningful
    once the two sides are known to hold the same shape.
    """
    offsets = struct.unpack_from("<35I", body, 32)
    counts = struct.unpack_from("<14H", body, 4)
    _, joints, mesh_start, _, _ = model_record(body, model_index)
    mesh_offsets = struct.unpack_from(f"<{counts[3]}I", body, offsets[6])
    signature = []
    for joint in range(joints):
        index = mesh_start + joint
        if joint > 0 and mesh_offsets[index] == 0:
            signature.append(None)
            continue
        vertices, _, _ = decode_mesh(body, index)
        if not vertices:
            signature.append((0, (0, 0, 0), (0, 0, 0)))
            continue
        lo = tuple(min(v[i] for v in vertices) for i in range(3))
        hi = tuple(max(v[i] for v in vertices) for i in range(3))
        signature.append((len(vertices), lo, hi))
    return signature


def same_shape(a, b, tolerance: int = 3) -> bool:
    """Do the two bodies hold the same prop?

    Vertex counts are compared loosely and bounds within a few stored units,
    because the two packers quantise differently: TR1 truncates with an
    arithmetic shift, this converter rounds to nearest. That shifts a vertex by
    up to one stored unit and can merge a different pair of near-coincident
    corners, so an exact match is not available even for identical input.
    """
    if len(a) != len(b):
        return False
    for left, right in zip(a, b):
        if (left is None) != (right is None):
            return False
        if left is None:
            continue
        if abs(left[0] - right[0]) > max(2, round(0.1 * left[0])):
            return False
        for corner_a, corner_b in ((left[1], right[1]), (left[2], right[2])):
            if any(abs(x - y) > tolerance for x, y in zip(corner_a, corner_b)):
                return False
    return True


def frame_box(body: bytes, model_index: int):
    """Centre and largest span of a model's first keyframe, in engine units."""
    offsets = struct.unpack_from("<35I", body, 32)
    _, joints, _, _, anim_index = model_record(body, model_index)
    frame_at = offsets[12] + struct.unpack_from(
        "<I", body, offsets[7] + anim_index * 32)[0]
    lo_x, hi_x, lo_y, hi_y, lo_z, hi_z = struct.unpack_from("<6h", body, frame_at)
    centre = ((lo_x + hi_x) / 2, (lo_y + hi_y) / 2, (lo_z + hi_z) / 2)
    span = max(hi_x - lo_x, hi_y - lo_y, hi_z - lo_z, 1)
    return centre, span


def silhouette(image):
    """Which pixels the renderer actually painted, as a flat mask."""
    background = bytes(3)
    data = image.convert("RGB").tobytes()
    return [data[i:i + 3] != background for i in range(0, len(data), 3)]


def compare(a, b) -> dict:
    mask_a, mask_b = silhouette(a), silhouette(b)
    total = FRAME_WIDTH * FRAME_HEIGHT
    covered = sum(1 for x in mask_a if x)
    union = sum(1 for x, y in zip(mask_a, mask_b) if x or y)
    both = sum(1 for x, y in zip(mask_a, mask_b) if x and y)
    only_a = sum(1 for x, y in zip(mask_a, mask_b) if x and not y)
    only_b = sum(1 for x, y in zip(mask_a, mask_b) if y and not x)
    return {
        "covered_pixels": covered,
        "iou": (both / union) if union else 1.0,
        "only_reference": only_a,
        "only_custom": only_b,
        "disagreement": ((only_a + only_b) / union) if union else 0.0,
    }


def side_by_side(a, b, zoom: int):
    from PIL import Image

    mask_a, mask_b = silhouette(a), silhouette(b)
    diff = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0))
    pixels = diff.load()
    for i, (x, y) in enumerate(zip(mask_a, mask_b)):
        px, py = i % FRAME_WIDTH, i // FRAME_WIDTH
        if x and y:
            pixels[px, py] = (40, 60, 40)
        elif x:
            pixels[px, py] = (220, 70, 60)      # only the Tomb Raider path drew here
        elif y:
            pixels[px, py] = (70, 130, 230)     # only the glTF path drew here

    sheet = Image.new("RGB", (FRAME_WIDTH * 3, FRAME_HEIGHT), (0, 0, 0))
    sheet.paste(a, (0, 0))
    sheet.paste(b, (FRAME_WIDTH, 0))
    sheet.paste(diff, (FRAME_WIDTH * 2, 0))
    if zoom > 1:
        sheet = sheet.resize((sheet.width * zoom, sheet.height * zoom),
                             Image.Resampling.NEAREST)
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True, type=Path, help="reference TR1 AVP1 pack")
    parser.add_argument("--models", required=True, type=Path, help="extracted GLB folder")
    parser.add_argument("--common-h", required=True, type=Path)
    parser.add_argument("--glyphs", required=True, type=Path)
    parser.add_argument("--only", action="append", default=[],
                        help="restrict to these model names (repeatable)")
    parser.add_argument("--sheets", type=Path, help="folder to write comparison sheets into")
    parser.add_argument("--yaw", type=float, default=200.0)
    parser.add_argument("--pitch", type=float, default=18.0)
    parser.add_argument("--zoom", type=int, default=2)
    parser.add_argument("--neutral", action="store_true",
                        help="drop rotations and compare rest shapes; note that the glTF "
                             "path carries its coordinate flip in joint 0's rotation, so "
                             "this is only comparable between two Tomb Raider bodies")
    parser.add_argument("--iou-floor", type=float, default=0.90,
                        help="minimum silhouette overlap to count as a match")
    args = parser.parse_args()

    item_types = read_item_types(args.common_h.resolve())
    pack = args.pack.resolve().read_bytes()
    glyphs = B.load_glyph_strip(args.glyphs.resolve())

    wanted = {n.lower() for n in args.only}
    paths = sorted(args.models.resolve().glob("*.glb"))
    if wanted:
        paths = [p for p in paths if p.stem.lower() in wanted]

    rows = []
    missing = skipped = 0
    for path in paths:
        name = path.stem.upper()
        if name not in item_types:
            missing += 1
            continue
        item_type = item_types.index(name)
        reference_body, reference_index = find_in_pack(pack, item_type)
        if reference_body is None:
            missing += 1
            continue
        try:
            built = B.build_model(
                B.ModelSpec(path=path, name=name, slot=0), glyphs, verbose=False)
        except Exception as error:
            rows.append((name, None, str(error)[:60]))
            skipped += 1
            continue

        # Frame both from the reference model, so any difference is the model's
        # and not the camera's.
        centre, span = frame_box(reference_body, reference_index)
        if args.neutral:
            centre = (0.0, 0.0, 0.0)
        distance = max(span * 1.35, 320.0)
        # Several angles, because a flat prop seen edge-on is a one-pixel line
        # and any rounding difference then reads as a total mismatch.
        views = []
        try:
            for offset in (0.0, 55.0, 125.0):
                yaw = math.radians(args.yaw + offset)
                pitch = math.radians(args.pitch + (12.0 if offset else 0.0))
                left = render(reference_body, 0, 0, yaw, pitch, distance,
                              reference_index, centre=centre, neutral=args.neutral)
                right = render(built.body, 0, 0, yaw, pitch, distance, 0,
                               centre=centre, neutral=args.neutral)
                views.append((compare(left, right), left, right))
        except SystemExit as error:
            rows.append((name, None, f"render: {error}"))
            skipped += 1
            continue

        stats = max(views, key=lambda v: v[0]["covered_pixels"])[0]
        stats["iou"] = sum(v[0]["iou"] for v in views) / len(views)
        left, right = views[0][1], views[0][2]
        stats["same_shape"] = same_shape(
            shape_signature(reference_body, reference_index),
            shape_signature(built.body, 0),
        )
        rows.append((name, stats, None))
        if args.sheets:
            args.sheets.mkdir(parents=True, exist_ok=True)
            side_by_side(left, right, args.zoom).save(args.sheets / f"{path.stem}.png")

    comparable = [r for r in rows if r[1] and r[1]["same_shape"]]
    other_variant = [r for r in rows if r[1] and not r[1]["same_shape"]]
    failed = [r for r in rows if r[1] is None]
    good = [r for r in comparable if r[1]["iou"] >= args.iou_floor]
    poor = [r for r in comparable if r[1]["iou"] < args.iou_floor]

    print(f"round trip over {len(rows)} models ({missing} absent from one path)")
    print(f"  same shape on both sides       : {len(comparable)}")
    print(f"    silhouette overlap >= {args.iou_floor:.0%}  : {len(good)}")
    print(f"    below the floor              : {len(poor)}")
    if good:
        mean = sum(r[1]["iou"] for r in good) / len(good)
        worst = min(good, key=lambda r: r[1]["iou"])
        print(f"    mean overlap                 : {mean:.4f}"
              f"   worst {worst[0]} at {worst[1]['iou']:.3f}")
    print(f"  each path kept a different variant: {len(other_variant)}")
    print(f"  could not be converted           : {len(failed)}")
    for name, stats, _ in sorted(poor, key=lambda r: r[1]["iou"])[:15]:
        print(f"    MISMATCH {name:<20} overlap {stats['iou']:.3f}  "
              f"only-TR1 {stats['only_reference']:5d}  only-glTF {stats['only_custom']:5d}")
    for name, _, error in failed[:10]:
        print(f"    UNCONVERTED {name:<17} {error}")
    return 0 if not poor else 1


if __name__ == "__main__":
    raise SystemExit(main())
