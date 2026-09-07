#!/usr/bin/env python3
"""Static and artifact validation for the C/C++ GBA asset viewer."""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path


PLAYABLE = [
    "GYM", "LEVEL1", "LEVEL2", "LEVEL3A", "LEVEL3B", "LEVEL4",
    "LEVEL5", "LEVEL6", "LEVEL7A", "LEVEL7B", "LEVEL8A", "LEVEL8B",
    "LEVEL8C", "LEVEL10A", "LEVEL10B", "LEVEL10C",
]

SUPPORTED_PACK_VERSIONS = (2, 3, 4)
NAME_ENTRY_SIZE = 24
SKIN_ENTRY_SIZE = 16        # bone table, then the blend records
PACK_FLAG_TR1 = 1 << 0
PACK_FLAG_NAMES = 1 << 1
# One byte per vertex naming the joint it follows, so the viewer can pose a
# model a vertex at a time. v4 appends the table's offset to the header.
PACK_FLAG_SKIN = 1 << 2


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def check_pack(data: bytes, label: str) -> tuple[int, int]:
    if len(data) < 24:
        fail(f"asset pack is truncated: {label}")
    magic, version, total, source_count, model_count, source_table, model_table = \
        struct.unpack_from("<4sIIHHII", data, 0)
    if magic != b"AVP1" or version not in SUPPORTED_PACK_VERSIONS or total != len(data):
        fail(f"invalid asset-pack header: {label}")
    if source_table + source_count * 16 > total or model_table + model_count * 12 > total:
        fail(f"asset-pack tables exceed file: {label}")

    # A v2 pack predates the flag word and is always Tomb Raider data, which is
    # what the runtime assumes for it too.
    flags = PACK_FLAG_TR1
    if version >= 3:
        if len(data) < 32:
            fail(f"v3 asset-pack header is truncated: {label}")
        flags, name_table = struct.unpack_from("<II", data, 24)
        if flags & ~(PACK_FLAG_TR1 | PACK_FLAG_NAMES | PACK_FLAG_SKIN):
            fail(f"unknown asset-pack flags 0x{flags:X}: {label}")
    if version >= 4:
        if len(data) < 36:
            fail(f"v4 asset-pack header is truncated: {label}")
        skin_table, = struct.unpack_from("<I", data, 32)
        if bool(flags & PACK_FLAG_SKIN) != bool(skin_table):
            fail(f"the skin flag and the skin table disagree: {label}")
        if skin_table:
            if skin_table > len(data) or source_count * SKIN_ENTRY_SIZE > len(data) - skin_table:
                fail(f"skin table does not fit the pack: {label}")
            for index in range(source_count):
                offset, size, blends, records = struct.unpack_from(
                    "<IIII", data, skin_table + index * SKIN_ENTRY_SIZE)
                if offset == 0 and size == 0:
                    continue                      # this model needs no posing
                if offset > len(data) or size > len(data) - offset:
                    fail(f"skin table entry {index} is out of range: {label}")
                if not records:
                    continue
                if blends == 0 or blends > len(data):
                    fail(f"blend records for entry {index} are out of range: {label}")
                # The viewer walks the records and the vertices together in one
                # pass, so a record out of order would silently pose the wrong
                # vertex. Cheap to check here, invisible on hardware.
                walk, previous = blends, -1
                for record in range(records):
                    if walk + 4 > len(data):
                        fail(f"blend record {record} of entry {index} is truncated: {label}")
                    vertex, count = struct.unpack_from("<HB", data, walk)
                    if not 2 <= count <= 4:
                        fail(f"blend record {record} of entry {index} names "
                             f"{count} bones: {label}")
                    if vertex <= previous:
                        fail(f"blend records of entry {index} are out of order "
                             f"at {record}: {label}")
                    previous = vertex
                    walk += 4 + count * 8
                if walk > len(data):
                    fail(f"blend records of entry {index} run past the pack: {label}")
        if bool(flags & PACK_FLAG_NAMES) != bool(name_table):
            fail(f"name-table flag and offset disagree: {label}")
        if flags & PACK_FLAG_NAMES:
            if name_table < 32 or name_table + model_count * NAME_ENTRY_SIZE > total:
                fail(f"name table exceeds file: {label}")
            for index in range(model_count):
                slot = data[name_table + index * NAME_ENTRY_SIZE:
                            name_table + (index + 1) * NAME_ENTRY_SIZE]
                if b"\0" not in slot:
                    fail(f"model name {index} is not NUL-terminated: {label}")
                name = slot.split(b"\0", 1)[0]
                if not name or not name.decode("ascii", "strict").strip():
                    fail(f"empty model name at index {index}: {label}")

    bodies: list[bytes] = []
    names: set[str] = set()
    for index in range(source_count):
        entry = source_table + index * 16
        name = data[entry:entry + 8].split(b"\0", 1)[0].decode("ascii", "strict")
        offset, size = struct.unpack_from("<II", data, entry + 8)
        if not name or name in names or size < 172 or offset + size > total:
            fail(f"invalid source {index} in {label}")
        names.add(name)
        body = data[offset:offset + size]
        if body[:4] != b"GBA ":
            fail(f"source {name} is not a PKD in {label}")
        counts = struct.unpack_from("<14H", body, 4)
        offsets = struct.unpack_from("<35I", body, 32)
        if any(value > len(body) for value in offsets) or list(offsets) != sorted(offsets):
            fail(f"invalid PKD offsets for {name} in {label}")
        for count_index in (1, 4, 6, 7, 10, 11, 12, 13):
            if counts[count_index] != 0:
                fail(f"gameplay count {count_index} retained in {name} ({label})")
        if counts[5] != 1 or counts[9] != 110:
            fail(f"viewer glyph set is incomplete in {name} ({label})")

        tile_count, model_rows, mesh_count = counts[0], counts[2], counts[3]
        texture_count, sprite_count = counts[8], counts[9]
        for table_offset, count, stride, kind in (
            (offsets[15], texture_count, 12, "texture"),
            (offsets[16], sprite_count, 16, "sprite"),
        ):
            if table_offset + count * stride > len(body):
                fail(f"truncated {kind} table in {name} ({label})")
            for row in range(count):
                raw = struct.unpack_from("<I", body, table_offset + row * stride)[0]
                if ((raw >> 16) & 0x3FFF) >= tile_count:
                    fail(f"invalid {kind} tile index in {name} ({label})")

        if offsets[6] + mesh_count * 4 > offsets[7]:
            fail(f"truncated mesh-offset table in {name} ({label})")
        mesh_offsets = struct.unpack_from(f"<{mesh_count}I", body, offsets[6]) if mesh_count else ()
        unique_meshes = sorted(set(mesh_offsets))
        if not flags & PACK_FLAG_TR1:
            # In a custom pack a zero offset marks a joint with no mesh -- the
            # producer pads its mesh blob so no real block can sit there -- so
            # there is nothing to validate at zero.
            unique_meshes = [value for value in unique_meshes if value]
        for mesh_offset in unique_meshes:
            start = offsets[5] + mesh_offset
            next_offset = next((value for value in unique_meshes if value > mesh_offset), offsets[6] - offsets[5])
            end = offsets[5] + next_offset
            if start + 20 > end or end > offsets[6]:
                fail(f"invalid mesh block in {name} ({label})")
            quads, triangles = struct.unpack_from("<hh", body, start + 12)
            if quads < 0 or triangles < 0 or start + 20 + (quads + triangles) * 8 > end:
                fail(f"invalid mesh faces in {name} ({label})")
            for face in range(quads + triangles):
                # Named apart from the pack's own `flags`, which is still needed
                # below: shadowing it here made every body after the first read
                # a face word as though it were the pack header.
                face_flags = struct.unpack_from("<H", body, start + 24 + face * 8)[0]
                if ((face_flags >> 14) & 0x0F) in {2, 3, 4, 5} and                         (face_flags & 0x3FFF) >= texture_count:
                    fail(f"invalid mesh texture in {name} ({label})")

        if offsets[13] + model_rows * 8 > offsets[14]:
            fail(f"truncated model table in {name} ({label})")
        bodies.append(body)

    seen: set[int] = set()
    previous_type = -1
    assigned: list[set[int]] = [set() for _ in bodies]
    for index in range(model_count):
        item_type, source, mesh_count, reserved, clips, reserved2, mesh_mask = struct.unpack_from(
            "<BBBBHHI", data, model_table + index * 12
        )
        if item_type in seen or item_type <= previous_type or source >= source_count:
            fail(f"duplicate, unsorted or invalid model catalog in {label}")
        allowed_mask = 0xFFFFFFFF if mesh_count >= 32 else (1 << mesh_count) - 1
        if not mesh_count or mesh_count > 32 or not clips or not mesh_mask or \
                mesh_mask & ~allowed_mask or reserved or reserved2:
            fail(f"invalid model metadata for type {item_type} in {label}")
        catalog = pkd_catalog(bodies[source], bool(flags & PACK_FLAG_TR1))
        if catalog.get(item_type) != (clips, mesh_count, mesh_mask):
            fail(f"catalog/body mismatch for type {item_type} in {label}")
        seen.add(item_type)
        assigned[source].add(item_type)
        previous_type = item_type

    for source, body in enumerate(bodies):
        if set(pkd_catalog(body)) != assigned[source]:
            fail(f"unreferenced or missing model rows in source {source} ({label})")
    return source_count, model_count


def check_rom(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if len(data) < 192:
        fail(f"ROM too small: {path}")
    title = data[0xA0:0xAC].rstrip(b"\0").decode("ascii", "replace")
    code = data[0xAC:0xB0].decode("ascii", "replace")
    if title != "ASSET VIEWER" or code != "OLAV":
        fail(f"bad ROM identity in {path}: {title!r}/{code!r}")
    checksum = (-(sum(data[0xA0:0xBD]) + 0x19)) & 0xFF
    if data[0xBD] != checksum:
        fail(f"bad GBA header checksum in {path}")
    if len(data) > 32 * 1024 * 1024:
        fail(f"ROM exceeds 32 MiB: {path} ({len(data)} bytes)")
    pack_offset = -1
    search_from = 0
    while True:
        candidate = data.find(b"AVP1", search_from)
        if candidate < 0:
            break
        if candidate + 12 <= len(data):
            version, pack_size = struct.unpack_from("<II", data, candidate + 4)
            if version in SUPPORTED_PACK_VERSIONS and 24 <= pack_size <= len(data) - candidate:
                pack_offset = candidate
                break
        search_from = candidate + 1
    if pack_offset < 0:
        fail(f"embedded asset pack missing or truncated: {path}")
    pack_size = struct.unpack_from("<I", data, pack_offset + 8)[0]
    return check_pack(data[pack_offset:pack_offset + pack_size], str(path))


def pkd_catalog(data: bytes, mesh_at_zero: bool = True) -> dict[int, tuple[int, int, int]]:
    """Rebuild each model's (mesh count, animation index, visibility mask).

    `mesh_at_zero` says whether a mesh-offset of zero at table index zero means
    a real block. Tomb Raider's data puts its first mesh there, so for a TR1
    pack it does; the custom producer pads its mesh blob precisely so that it
    does not, which is what lets a custom model have an empty root joint.
    """
    if len(data) < 88:
        fail("PKD is truncated")
    models_count, meshes_count = struct.unpack_from("<HH", data, 8)
    mesh_offsets_offset = struct.unpack_from("<I", data, 56)[0]
    anims_offset, states_offset = struct.unpack_from("<II", data, 60)
    models_offset = struct.unpack_from("<I", data, 84)[0]
    if states_offset < anims_offset or (states_offset - anims_offset) % 32:
        fail("invalid PKD animation table")
    if models_offset + models_count * 8 > len(data):
        fail("invalid PKD model table")

    animation_count = (states_offset - anims_offset) // 32
    if mesh_offsets_offset + meshes_count * 4 > len(data):
        fail("invalid PKD mesh-offset table")
    mesh_offsets = struct.unpack_from(f"<{meshes_count}I", data, mesh_offsets_offset)
    models: dict[int, tuple[int, int, int]] = {}
    for index in range(models_count):
        item_type, mesh_count, mesh_start, _, anim_index = struct.unpack_from(
            "<BbHHH", data, models_offset + index * 8
        )
        if mesh_count > 0 and item_type < 191 and anim_index < animation_count:
            if mesh_count > 32 or mesh_start + mesh_count > meshes_count:
                fail(f"invalid PKD mesh range for type {item_type}")
            mesh_mask = 0
            for slot, mesh_index in enumerate(range(mesh_start, mesh_start + mesh_count)):
                if mesh_offsets[mesh_index] != 0 or (mesh_index == 0 and mesh_at_zero):
                    mesh_mask |= 1 << slot
            models[item_type] = (mesh_count, anim_index, mesh_mask)

    starts = sorted({anim_index for _, anim_index, _ in models.values()})
    result: dict[int, tuple[int, int, int]] = {}
    for item_type, (mesh_count, anim_index, mesh_mask) in models.items():
        end = next((value for value in starts if value > anim_index), animation_count)
        if end > anim_index and mesh_mask:
            result[item_type] = (end - anim_index, mesh_count, mesh_mask)
    return result


def check_unified_catalog(sources: list[tuple[str, bytes]]) -> tuple[int, int]:
    winners: dict[int, tuple[int, int, int, str]] = {}
    raw_count = 0
    for source_index, (name, data) in enumerate(sources):
        models = pkd_catalog(data)
        raw_count += len(models)
        for item_type, (clips, meshes, _) in models.items():
            candidate = (clips, meshes, -source_index, name)
            previous = winners.get(item_type)
            if previous is None or candidate[:3] > previous[:3]:
                winners[item_type] = candidate
    if not winners or len(winners) != len(set(winners)):
        fail("unified catalog is empty or contains duplicate ItemTypes")
    return raw_count, len(winners)


def check_wad(path: Path) -> list[tuple[str, bytes]]:
    data = path.read_bytes()
    if len(data) < 24:
        fail("WAD is truncated")
    magic, version, total, count = struct.unpack_from("<4I", data, 0)
    if magic != 0x31575254 or version != 1 or total != len(data) or count != 16:
        fail(f"invalid WAD header: magic={magic:#x} version={version} total={total} count={count}")
    names = []
    sources = []
    for index in range(count):
        off = 24 + index * 16
        tag = data[off:off + 8].split(b"\0", 1)[0].decode("ascii")
        body_off, body_size = struct.unpack_from("<II", data, off + 8)
        if body_size < 172 or body_off + body_size > len(data):
            fail(f"invalid WAD body for {tag}")
        names.append(tag)
        sources.append((tag, data[body_off:body_off + body_size]))
    if names != PLAYABLE:
        fail(f"unexpected WAD level order: {names}")
    return sources


def load_settings(bundle: Path) -> dict:
    path = bundle / "settings.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        fail(f"settings.json is not valid JSON: {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parents[1],
                        help="the asset viewer folder; defaults to the one holding this script")
    parser.add_argument("--openlara", type=Path,
                        help="donor checkout; defaults to the path in settings.json")
    parser.add_argument("--wad", type=Path)
    parser.add_argument("--rom", action="append", type=Path, default=[],
                        help="repeatable; defaults to every .gba in the bundle's roms folder")
    parser.add_argument("--pack", action="append", type=Path, default=[],
                        help="repeatable; defaults to the pack the last build staged")
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    settings = load_settings(bundle)

    repo = args.openlara or settings.get("openlara")
    if not repo:
        fail("no OpenLara checkout given: set \"openlara\" in settings.json or pass --openlara")
    repo = Path(repo).resolve()
    if not (repo / "src/platform/gba/Makefile").is_file():
        fail(f"not an OpenLara checkout: {repo}")

    # The viewer's own sources live here, not in the donor.
    overlay = bundle / "overlay/src/platform/gba"
    viewer = overlay / "asset_viewer.cpp"
    # main.cpp, common.cpp and the Makefile carry the viewer's hooks, so the
    # bundle owns them too; the donor keeps its pristine copies.
    donor_main = overlay / "main.cpp"
    donor_makefile = overlay / "Makefile"
    donor_common = bundle / "overlay/src/fixed/common.cpp"
    for path in [viewer, overlay / "asset_viewer.h", overlay / "asset_viewer_runtime.h",
                 donor_main, donor_makefile, donor_common,
                 repo / "src/fixed/common.h", repo / "src/platform/gba/Makefile"]:
        if not path.is_file():
            fail(f"missing {path}")

    # What matters now is that the tree the ROM was compiled from still carries
    # the overlay unchanged; that is what ties these sources to that binary.
    work = bundle / str(settings.get("work", "build")) / "src/platform/gba"
    if work.is_dir():
        for name in ("asset_viewer.cpp", "asset_viewer.h", "asset_viewer_runtime.h",
                     "main.cpp", "Makefile"):
            built = work / name
            if not built.is_file():
                fail(f"the work tree is missing {name}; rebuild")
            if built.read_text(encoding="utf-8").replace("\r\n", "\n") != \
                    (overlay / name).read_text(encoding="utf-8").replace("\r\n", "\n"):
                fail(f"the work tree was built from a different {name}; rebuild")

    if not args.rom:
        roms = bundle / str(settings.get("roms", "roms"))
        args.rom = sorted(roms.glob("*.gba")) if roms.is_dir() else []
    if not args.pack:
        staged = work / ".asset_viewer_pack_data/ASSET_VIEWER.PAK"
        if staged.is_file():
            args.pack = [staged]

    cpp = viewer.read_text(encoding="utf-8")
    main_cpp = donor_main.read_text(encoding="utf-8")
    makefile = donor_makefile.read_text(encoding="utf-8")
    common_cpp = donor_common.read_text(encoding="utf-8")
    required = [
        (cpp, "screenOffsetToWorld", "projected centering"),
        (cpp, "state.fitRadius * 5", "rotation-safe bounding-sphere fit"),
        (cpp, "scaleCurrentBasis(state.fitScale)", "large-model fallback scale"),
        (cpp, "MODEL_TARGET_X = FRAME_WIDTH / 2", "horizontally centered target"),
        (cpp, "state.yaw = ANGLE_180", "front-facing initial model rotation"),
        (cpp, "Keep every view field untouched", "animation view-state preservation"),
        (cpp, "rootNeutralBounds", "root-neutral fit bounds"),
        (cpp, "projectionDepth / 16", "exact Mode 4 inverse projection"),
        (cpp, "state.showHelp = !state.showHelp", "Select help toggle"),
        (cpp, "buildUnifiedCatalog", "unified model catalog"),
        (cpp, "state.modelSources[type]", "automatic source load"),
        (cpp, "state.modelMasks[type]", "per-model mesh sentinel mask"),
        (cpp, "prepareLaraVariant", "armed Lara mesh composition"),
        (cpp, "INFO_PANEL_TOP = 140", "two-line bottom information panel"),
        (cpp, "drawTextClipped(4, 149", "first compact detail line"),
        (cpp, "drawTextClipped(4, 158", "second compact detail line"),
        (cpp, "INFO_STATUS_X = 190", "fixed playback-status column"),
        (cpp, "drawTextCompact(INFO_STATUS_X, 158", "anchored playback status"),
        (cpp, "HELP_PANEL_TOP, FRAME_WIDTH, FRAME_HEIGHT - HELP_PANEL_TOP, 0, 1", "opaque help overlay depth"),
        (cpp, "if (!AssetViewer::state.showHelp)", "model occlusion behind help overlay"),
        (cpp, "state.modelCount > 0 ? state.modelName", "selected model title"),
        (cpp, "MODEL_NAMES[type]", "ITEM_TYPES title fallback"),
        (cpp, "PACK_FLAG_NAMES", "pack-supplied model names"),
        (cpp, "packFlags() & PACK_FLAG_TR1", "TR1-only runtime fixups gated by pack flag"),
        (cpp, "HELP_PANEL_TOP = 31", "full-height controls overlay"),
        (cpp, "B RESET  SELECT CLOSE", "visible menu close control"),
        (cpp, "WIRE_COLOR = 12", "wireframe colour from the reserved glyph palette"),
        (cpp, "wireNodes(&item, frameA, frameB, frameDelta, frameRate)",
         "wireframe node walk"),
        (cpp, "state.settings[state.menuCursor] = !state.settings[state.menuCursor]",
         "A toggles the highlighted setting"),
        (cpp, "drawSettingRow(45, SETTING_WIREFRAME", "wireframe settings row"),
        (cpp, "YAW_HOLD_TICKS = 8", "tap/hold yaw threshold"),
        (cpp, "AUTO_ROTATION_STEP = ANGLE_1", "automatic yaw speed"),
        (cpp, "state.autoYawDirection == state.yawPressDirection", "automatic yaw toggle"),
        (cpp, "state.yawHoldTicks >= YAW_HOLD_TICKS", "manual yaw long press"),
        (cpp, "adjustVertical(-1, frames)", "vertical move up control"),
        (cpp, "adjustVertical(1, frames)", "vertical move down control"),
        (cpp, "X_CLAMP(state.verticalOffset, -144, 144)", "tripled vertical movement range"),
        (cpp, "MODEL_TARGET_Y + state.verticalOffset", "vertical render offset"),
        (cpp, "ASSET_VIEWER_PAK", "deduplicated build-time pack"),
        (cpp, "getViewerFrames", "viewer-only frame lookup"),
        (cpp, "drawNodesLerp", "native hierarchy renderer"),
        (main_cpp, "assetViewerRender", "main dispatch"),
        (main_cpp, "asset_viewer_runtime.h", "viewer-only runtime"),
        (makefile, "ASSET_VIEWER_FULL", "Makefile feature"),
        (makefile, "%.PAK.o", "asset-pack embedding rule"),
        (makefile, "--gc-sections", "viewer dead-code elimination"),
        (common_cpp, "Gameplay screens", "game-data exclusion"),
    ]
    for text, marker, label in required:
        if marker not in text:
            fail(f"missing {label}: {marker}")

    select_animation = cpp.split("static void selectAnimation", 1)[1].split(
        "static void adjustVertical", 1
    )[0]
    if "resetClip(" in select_animation or "recomputeFit(" in select_animation:
        fail("animation selection still resets fitted view state")
    for field in (
        "yaw", "pitch", "roll", "zoom", "verticalOffset", "fitCenter",
        "fitRadius", "fitDistance", "fitScale", "autoYawDirection",
    ):
        if f"state.{field} =" in select_animation:
            fail(f"animation selection changes view field: {field}")

    # The settings row is drawn by drawSettingRow at its own x, so it does
    # not appear here; these are the centred lines only.
    expected_help = [
        (36, "SETTINGS"),
        (54, "UP/DN PICK  A TOGGLE"),
        (63, "CONTROLS"),
        (72, "TAP LEFT/RIGHT AUTO Y"),
        (81, "HOLD LEFT/RIGHT ROTATE Y"),
        (90, "DPAD UP/DN ROTATE X"),
        (99, "SEL+DPAD ROTATE Z ZOOM"),
        (108, "SEL+L/R BUTTONS MOVE Y"),
        (117, "L/R PREV/NEXT MODEL"),
        (126, "START NEXT SEL+START PREV"),
        (135, "A PLAY/PAUSE SEL+A STEP"),
        (144, "B RESET  SELECT CLOSE"),
    ]
    actual_help = [
        (int(y), label)
        for y, label in re.findall(
            r'drawTextCompact\(0,\s*(\d+),\s*"([^"]+)"', cpp
        )
        if 31 < int(y) <= 144
    ]
    if actual_help != expected_help or expected_help[-1][0] + 8 >= 160:
        fail(f"controls overlay layout mismatch: {actual_help}")

    width_match = re.search(
        r"CHAR_WIDTH\[GLYPH_COUNT\]\s*=\s*\{([^}]*)\}", cpp, re.S
    )
    map_match = re.search(r"CHAR_MAP\[102\]\s*=\s*\{([^}]*)\}", cpp, re.S)
    if not width_match or not map_match:
        fail("compact font metrics are missing")
    char_width = [int(value) for value in re.findall(r"\d+", width_match.group(1))]
    char_map = [int(value) for value in re.findall(r"\d+", map_match.group(1))]

    def compact_width(label: str) -> int:
        width = 0
        for char in label:
            if char in " _":
                width += 3
            elif 32 <= ord(char) < 32 + len(char_map):
                width += (char_width[char_map[ord(char) - 32]] + 2) // 2
        return width

    for _, label in expected_help:
        width = compact_width(label)
        if width > 236:
            fail(f"controls overlay line is too wide ({width}px): {label}")
    for label in ("PLAYING", "PAUSED"):
        if 190 + compact_width(label) > 236:
            fail(f"anchored playback status exceeds info panel: {label}")

    forbidden = [
        ("selectSource(", "manual source selector"),
        ('line.append("SRC ")', "visible source label"),
        ("drawModelList(", "neighbour model list"),
    ]
    for marker, label in forbidden:
        if marker in cpp:
            fail(f"obsolete {label} remains: {marker}")

    common_h = (repo / "src/fixed/common.h").read_text(encoding="utf-8")
    block = common_h.split("#define ITEM_TYPES(E)", 1)[1].split("enum ItemType", 1)[0]
    names = re.findall(r"\bE\(\s*([A-Z0-9_]+)\s*\)", block)
    if len(names) != 191 or names[0] != "LARA" or names[-1] != "GLYPHS":
        fail(f"ITEM_TYPES mismatch: count={len(names)} first={names[:1]} last={names[-1:]}")

    catalog_stats = None
    if args.wad:
        wad_sources = check_wad(args.wad.resolve())
        title = repo / "src/platform/gba/data/TITLE.PKD"
        if not title.is_file():
            fail(f"missing {title}")
        catalog_stats = check_unified_catalog([("TITLE", title.read_bytes()), *wad_sources])
    pack_stats = []
    for pack in args.pack:
        resolved = pack.resolve()
        pack_stats.append((resolved, check_pack(resolved.read_bytes(), str(resolved))))
    rom_stats = []
    for rom in args.rom:
        resolved = rom.resolve()
        rom_stats.append((resolved, check_rom(resolved)))

    print("PASS: C/C++ asset viewer static validation")
    print("PASS: 191 ITEM_TYPES names (LARA..GLYPHS)")
    print("PASS: black controls overlay y=31..159, 13 unclipped lines")
    if args.wad:
        print("PASS: TR1.WAD header and 16 playable levels")
        print(
            "PASS: unified catalog "
            f"{catalog_stats[0]} raw candidates -> {catalog_stats[1]} unique ItemTypes"
        )
    for path, stats in pack_stats:
        print(f"PASS: pack {path} ({stats[0]} sources, {stats[1]} unique models)")
    for path, stats in rom_stats:
        print(f"PASS: ROM {path} ({stats[0]} sources, {stats[1]} unique models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
