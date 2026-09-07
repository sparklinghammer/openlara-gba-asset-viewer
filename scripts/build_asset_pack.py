#!/usr/bin/env python3
"""Build a deduplicated, viewer-only OpenLara GBA asset pack from PKD/WAD data."""

from __future__ import annotations

import argparse
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

from pkd_writer import PkdParts, body_stats, write_pkd_body


PKD_HEADER_SIZE = 172
PKD_MAGIC = b"GBA "
WAD_MAGIC = b"TRW1"
PACK_MAGIC = b"AVP1"
PACK_VERSION = 3
PACK_HEADER_SIZE = 32
SOURCE_ENTRY_SIZE = 16
MODEL_ENTRY_SIZE = 12
NAME_ENTRY_SIZE = 24
PACK_FLAG_TR1 = 1 << 0
PACK_FLAG_NAMES = 1 << 1
GLYPHS_TYPE = 190
GLYPH_COUNT = 110
ITEM_TYPE_COUNT = 191
FACE_TEXTURE = 0x3FFF
TEXTURED_FACE_TYPES = {2, 3, 4, 5}


def align(buffer: bytearray, boundary: int = 4) -> int:
    while len(buffer) % boundary:
        buffer.append(0)
    return len(buffer)


def put_u16(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<H", buffer, offset, value)


def put_u32(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<I", buffer, offset, value)


@dataclass(frozen=True)
class Model:
    item_type: int
    mesh_count: int
    mesh_start: int
    node_index: int
    anim_start: int


class Pkd:
    def __init__(self, name: str, data: bytes):
        if len(data) < PKD_HEADER_SIZE or data[:4] != PKD_MAGIC:
            raise ValueError(f"{name}: invalid PKD")
        self.name = name
        self.data = data
        self.counts = struct.unpack_from("<14H", data, 4)
        self.offsets = struct.unpack_from("<35I", data, 32)
        self.anim_count = (self.offsets[8] - self.offsets[7]) // 32
        if self.offsets[8] < self.offsets[7] or (self.offsets[8] - self.offsets[7]) % 32:
            raise ValueError(f"{name}: invalid animation table")

        models_offset = self.offsets[13]
        self.models: dict[int, Model] = {}
        for index in range(self.counts[2]):
            item_type, count, start, node, anim = struct.unpack_from(
                "<BbHHH", data, models_offset + index * 8
            )
            if count > 0 and anim < self.anim_count:
                self.models[item_type] = Model(item_type, count, start, node, anim)

        starts = sorted({model.anim_start for model in self.models.values()})
        self.anim_ranges: dict[int, tuple[int, int]] = {}
        for item_type, model in self.models.items():
            end = next((value for value in starts if value > model.anim_start), self.anim_count)
            if end > model.anim_start:
                self.anim_ranges[item_type] = (model.anim_start, end)

    def section(self, index: int, next_index: int) -> bytes:
        start = self.offsets[index]
        end = self.offsets[next_index]
        if not (0 <= start <= end <= len(self.data)):
            raise ValueError(f"{self.name}: invalid section {index}")
        return self.data[start:end]

    def mesh_offsets(self) -> list[int]:
        offset = self.offsets[6]
        return list(struct.unpack_from(f"<{self.counts[3]}I", self.data, offset))

    def mesh_block(self, relative_offset: int) -> bytes:
        offsets = sorted(set(self.mesh_offsets()))
        try:
            index = offsets.index(relative_offset)
        except ValueError as error:
            raise ValueError(f"{self.name}: missing mesh offset {relative_offset}") from error
        end = offsets[index + 1] if index + 1 < len(offsets) else self.offsets[6] - self.offsets[5]
        start_abs = self.offsets[5] + relative_offset
        end_abs = self.offsets[5] + end
        if not (start_abs < end_abs <= self.offsets[6]):
            raise ValueError(f"{self.name}: invalid mesh block")
        return self.data[start_abs:end_abs]

    def model_mesh_mask(self, model: Model) -> int:
        """Return the drawable slots, preserving TR1's zero-offset sentinel."""
        if model.mesh_count > 32:
            raise ValueError(f"{self.name}: model {model.item_type} exceeds the 32-bit mesh mask")
        offsets = self.mesh_offsets()
        mask = 0
        for slot, mesh_index in enumerate(range(model.mesh_start, model.mesh_start + model.mesh_count)):
            if mesh_index >= len(offsets):
                raise ValueError(f"{self.name}: model {model.item_type} has an invalid mesh range")
            # Offset zero is a real mesh only for the first entry in the source.
            # Every later zero is the PHD sentinel for an absent mesh slot.
            if mesh_index == 0 or offsets[mesh_index] != 0:
                mask |= 1 << slot
        return mask

    def animation(self, index: int) -> bytes:
        if not 0 <= index < self.anim_count:
            raise ValueError(f"{self.name}: animation {index} out of range")
        start = self.offsets[7] + index * 32
        return self.data[start:start + 32]

    def animation_frame_block(self, animation_index: int) -> tuple[int, bytes]:
        frame_offset = struct.unpack_from("<I", self.animation(animation_index), 0)[0]
        all_offsets = sorted({
            struct.unpack_from("<I", self.data, self.offsets[7] + index * 32)[0]
            for index in range(self.anim_count)
        })
        next_offset = next(
            (value for value in all_offsets if value > frame_offset),
            self.offsets[13] - self.offsets[12],
        )
        start = self.offsets[12] + frame_offset
        end = self.offsets[12] + next_offset
        if not (self.offsets[12] <= start < end <= self.offsets[13]):
            raise ValueError(f"{self.name}: invalid frame block for animation {animation_index}")
        return frame_offset, self.data[start:end]

    def glyph_sprites(self) -> list[bytes]:
        sequences = self.offsets[17]
        glyph_start = None
        for index in range(self.counts[5]):
            item_type, _, count, start = struct.unpack_from("<HHhH", self.data, sequences + index * 8)
            if item_type == GLYPHS_TYPE and abs(count) >= GLYPH_COUNT:
                glyph_start = start
                break
        if glyph_start is None or glyph_start + GLYPH_COUNT > self.counts[9]:
            raise ValueError(f"{self.name}: GLYPHS sprite sequence is missing")
        sprites = self.offsets[16]
        return [
            self.data[sprites + (glyph_start + index) * 16:sprites + (glyph_start + index + 1) * 16]
            for index in range(GLYPH_COUNT)
        ]


def parse_wad(path: Path) -> list[Pkd]:
    data = path.read_bytes()
    if len(data) < 24 or data[:4] != WAD_MAGIC:
        raise ValueError(f"{path}: invalid TR1.WAD")
    version, total, count = struct.unpack_from("<III", data, 4)
    if version != 1 or total != len(data) or count > 32:
        raise ValueError(f"{path}: invalid WAD header")
    result = []
    for index in range(count):
        entry = 24 + index * SOURCE_ENTRY_SIZE
        name = data[entry:entry + 8].split(b"\0", 1)[0].decode("ascii")
        offset, size = struct.unpack_from("<II", data, entry + 8)
        if size < PKD_HEADER_SIZE or offset + size > len(data):
            raise ValueError(f"{path}: invalid body {name}")
        result.append(Pkd(name, data[offset:offset + size]))
    return result


def parse_source(value: str) -> Pkd:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or len(name.encode("ascii", "strict")) > 8:
        raise argparse.ArgumentTypeError("source name must contain 1..8 ASCII characters")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"PKD not found: {path}")
    return Pkd(name, path.read_bytes())


def read_item_types(path: Path) -> list[str]:
    """Read the donor's public ITEM_TYPES(E) macro, the single source of model names."""
    text = path.read_text(encoding="utf-8")
    if "#define ITEM_TYPES(E)" not in text or "enum ItemType" not in text:
        raise ValueError(f"{path}: ITEM_TYPES(E) macro not found")
    block = text.split("#define ITEM_TYPES(E)", 1)[1].split("enum ItemType", 1)[0]
    names = re.findall(r"\bE\(\s*([A-Z0-9_]+)\s*\)", block)
    if len(names) != ITEM_TYPE_COUNT or names[0] != "LARA" or names[-1] != "GLYPHS":
        raise ValueError(
            f"{path}: unexpected ITEM_TYPES contents "
            f"(count={len(names)} first={names[:1]} last={names[-1:]})"
        )
    for name in names:
        if len(name.encode("ascii")) > NAME_ENTRY_SIZE - 1:
            raise ValueError(f"{path}: model name exceeds {NAME_ENTRY_SIZE - 1} characters: {name}")
    return names


def choose_models(sources: list[Pkd]) -> dict[int, tuple[int, Model, int]]:
    winners: dict[int, tuple[int, Model, int]] = {}
    for source_index, source in enumerate(sources):
        for item_type, model in source.models.items():
            anim_range = source.anim_ranges.get(item_type)
            if not anim_range or source.model_mesh_mask(model) == 0:
                continue
            clips = anim_range[1] - anim_range[0]
            previous = winners.get(item_type)
            if previous is None:
                winners[item_type] = (source_index, model, clips)
                continue
            previous_source, previous_model, previous_clips = previous
            candidate_key = (clips, model.mesh_count, -source_index)
            previous_key = (previous_clips, previous_model.mesh_count, -previous_source)
            if candidate_key > previous_key:
                winners[item_type] = (source_index, model, clips)
    return winners


def collect_textures(source: Pkd, models: list[Model]) -> set[int]:
    mesh_offsets = source.mesh_offsets()
    result: set[int] = set()
    for model in models:
        for mesh_index in range(model.mesh_start, model.mesh_start + model.mesh_count):
            if mesh_index >= len(mesh_offsets):
                raise ValueError(f"{source.name}: model {model.item_type} has an invalid mesh range")
            old_offset = mesh_offsets[mesh_index]
            if mesh_index > 0 and old_offset == 0:
                continue
            block = source.mesh_block(old_offset)
            if len(block) < 20:
                raise ValueError(f"{source.name}: truncated mesh")
            quad_count, triangle_count = struct.unpack_from("<hh", block, 12)
            if quad_count < 0 or triangle_count < 0:
                raise ValueError(f"{source.name}: negative mesh face count")
            for face in range(quad_count + triangle_count):
                flags = struct.unpack_from("<H", block, 20 + face * 8 + 4)[0]
                if (flags >> 14) & 0x0F in TEXTURED_FACE_TYPES:
                    texture = flags & FACE_TEXTURE
                    if texture >= source.counts[8]:
                        raise ValueError(f"{source.name}: texture index {texture} out of range")
                    result.add(texture)
    return result


def remap_mesh(block: bytes, texture_map: dict[int, int], source_name: str) -> bytes:
    output = bytearray(block)
    quad_count, triangle_count = struct.unpack_from("<hh", output, 12)
    for face in range(quad_count + triangle_count):
        flag_offset = 20 + face * 8 + 4
        flags = struct.unpack_from("<H", output, flag_offset)[0]
        if (flags >> 14) & 0x0F in TEXTURED_FACE_TYPES:
            old_texture = flags & FACE_TEXTURE
            if old_texture not in texture_map:
                raise ValueError(f"{source_name}: missing remap for texture {old_texture}")
            struct.pack_into("<H", output, flag_offset, (flags & ~FACE_TEXTURE) | texture_map[old_texture])
    return bytes(output)


def remap_tile_record(record: bytes, tile_map: dict[int, int], source_name: str) -> bytes:
    output = bytearray(record)
    raw = struct.unpack_from("<I", output, 0)[0]
    old_tile = (raw >> 16) & 0x3FFF
    if old_tile not in tile_map:
        raise ValueError(f"{source_name}: missing remap for tile {old_tile}")
    struct.pack_into("<I", output, 0, (raw & 0xC000FFFF) | (tile_map[old_tile] << 16))
    return bytes(output)


def build_minimal_pkd(source: Pkd, selected_models: list[Model]) -> tuple[bytes, dict[str, int]]:
    selected_models = sorted(selected_models, key=lambda model: model.item_type)
    texture_ids = collect_textures(source, selected_models)
    texture_map = {old: new for new, old in enumerate(sorted(texture_ids))}

    textures = []
    texture_table = source.offsets[15]
    used_tiles: set[int] = set()
    for old_texture in sorted(texture_ids):
        record = source.data[texture_table + old_texture * 12:texture_table + (old_texture + 1) * 12]
        used_tiles.add((struct.unpack_from("<I", record, 0)[0] >> 16) & 0x3FFF)
        textures.append(record)

    glyph_sprites = source.glyph_sprites()
    for record in glyph_sprites:
        used_tiles.add((struct.unpack_from("<I", record, 0)[0] >> 16) & 0x3FFF)
    tile_map = {old: new for new, old in enumerate(sorted(used_tiles))}
    textures = [remap_tile_record(record, tile_map, source.name) for record in textures]
    glyph_sprites = [remap_tile_record(record, tile_map, source.name) for record in glyph_sprites]

    source_mesh_offsets = source.mesh_offsets()
    mesh_data = bytearray()
    mesh_offsets: list[int] = []
    mesh_remap: dict[int, int] = {}
    nodes = bytearray()
    node_remap: dict[tuple[int, int], int] = {}

    model_work: list[dict[str, int]] = []
    for model in selected_models:
        new_mesh_start = len(mesh_offsets)
        for mesh_index in range(model.mesh_start, model.mesh_start + model.mesh_count):
            old_offset = source_mesh_offsets[mesh_index]
            if mesh_index > 0 and old_offset == 0:
                mesh_offsets.append(0)
                continue
            if old_offset not in mesh_remap:
                align(mesh_data)
                mesh_remap[old_offset] = len(mesh_data)
                mesh_data.extend(remap_mesh(source.mesh_block(old_offset), texture_map, source.name))
            mesh_offsets.append(mesh_remap[old_offset])

        node_count = max(model.mesh_count - 1, 0)
        node_key = (model.node_index, node_count)
        if node_key not in node_remap:
            node_remap[node_key] = len(nodes) // 8
            begin = source.offsets[11] + model.node_index * 8
            end = begin + node_count * 8
            if end > source.offsets[12]:
                raise ValueError(f"{source.name}: invalid nodes for model {model.item_type}")
            nodes.extend(source.data[begin:end])
        model_work.append({
            "type": model.item_type,
            "count": model.mesh_count,
            "start": new_mesh_start,
            "node": node_remap[node_key],
            "old_anim_start": source.anim_ranges[model.item_type][0],
            "old_anim_end": source.anim_ranges[model.item_type][1],
        })

    animations = bytearray()
    frames = bytearray()
    frame_remap: dict[int, int] = {}
    range_remap: dict[tuple[int, int], int] = {}
    for model in model_work:
        old_range = (model["old_anim_start"], model["old_anim_end"])
        if old_range not in range_remap:
            range_remap[old_range] = len(animations) // 32
            for old_animation in range(*old_range):
                record = bytearray(source.animation(old_animation))
                old_frame_offset, frame_block = source.animation_frame_block(old_animation)
                if old_frame_offset not in frame_remap:
                    align(frames, 2)
                    frame_remap[old_frame_offset] = len(frames)
                    frames.extend(frame_block)
                new_animation = len(animations) // 32
                put_u32(record, 0, frame_remap[old_frame_offset])
                put_u16(record, 20, new_animation)
                put_u16(record, 22, struct.unpack_from("<H", record, 16)[0])
                put_u16(record, 24, 0)
                put_u16(record, 26, 0)
                put_u16(record, 28, 0)
                put_u16(record, 30, 0)
                animations.extend(record)
        model["anim"] = range_remap[old_range]

    models = bytearray()
    for model in model_work:
        models.extend(struct.pack(
            "<BbHHH",
            model["type"], model["count"], model["start"], model["node"], model["anim"],
        ))

    tiles_base = source.offsets[2]
    tile_pages = []
    for old_tile in sorted(used_tiles):
        if old_tile >= source.counts[0]:
            raise ValueError(f"{source.name}: tile {old_tile} out of range")
        tile_pages.append(
            source.data[tiles_base + old_tile * 65536:tiles_base + (old_tile + 1) * 65536]
        )

    body = write_pkd_body(PkdParts(
        palette=source.section(0, 1)[:512],
        lightmap=source.section(1, 2)[:8192],
        tiles=tile_pages,
        mesh_data=bytes(mesh_data),
        mesh_offsets=mesh_offsets,
        animations=bytes(animations),
        nodes=bytes(nodes),
        frames=bytes(frames),
        models=bytes(models),
        textures=textures,
        glyph_sprites=glyph_sprites,
    ))
    return body, body_stats(body)


def build_pack(sources: list[Pkd], item_names: list[str] | None = None) -> tuple[bytes, dict[str, object]]:
    winners = choose_models(sources)
    grouped: list[list[Model]] = [[] for _ in sources]
    for source_index, model, _ in winners.values():
        grouped[source_index].append(model)

    kept_sources: list[tuple[int, Pkd, bytes, dict[str, int]]] = []
    source_remap: dict[int, int] = {}
    for old_index, (source, models) in enumerate(zip(sources, grouped)):
        if not models:
            continue
        body, stats = build_minimal_pkd(source, models)
        source_remap[old_index] = len(kept_sources)
        kept_sources.append((old_index, source, body, stats))

    flags = PACK_FLAG_TR1
    if item_names is not None:
        flags |= PACK_FLAG_NAMES

    source_table_offset = PACK_HEADER_SIZE
    model_table_offset = source_table_offset + len(kept_sources) * SOURCE_ENTRY_SIZE
    name_table_offset = model_table_offset + len(winners) * MODEL_ENTRY_SIZE
    tables_end = name_table_offset
    if item_names is not None:
        tables_end += len(winners) * NAME_ENTRY_SIZE
    bodies_offset = (tables_end + 3) & ~3
    output = bytearray(bodies_offset)
    body_offsets: list[int] = []
    for _, _, body, _ in kept_sources:
        align(output)
        body_offsets.append(len(output))
        output.extend(body)

    struct.pack_into(
        "<4sIIHHIIII", output, 0, PACK_MAGIC, PACK_VERSION, len(output),
        len(kept_sources), len(winners), source_table_offset, model_table_offset,
        flags, name_table_offset if item_names is not None else 0,
    )
    for index, ((_, source, body, _), body_offset) in enumerate(zip(kept_sources, body_offsets)):
        entry = source_table_offset + index * SOURCE_ENTRY_SIZE
        name = source.name.encode("ascii")[:8]
        output[entry:entry + 8] = name + b"\0" * (8 - len(name))
        struct.pack_into("<II", output, entry + 8, body_offset, len(body))

    for index, item_type in enumerate(sorted(winners)):
        old_source, model, clips = winners[item_type]
        entry = model_table_offset + index * MODEL_ENTRY_SIZE
        struct.pack_into(
            "<BBBBHHI", output, entry,
            item_type, source_remap[old_source], model.mesh_count, 0, clips, 0,
            sources[old_source].model_mesh_mask(model),
        )
        if item_names is not None:
            name = item_names[item_type].encode("ascii")
            slot = name_table_offset + index * NAME_ENTRY_SIZE
            output[slot:slot + NAME_ENTRY_SIZE] = name + b"\0" * (NAME_ENTRY_SIZE - len(name))

    raw_candidates = sum(len(source.anim_ranges) for source in sources)
    report: dict[str, object] = {
        "format": "AVP1",
        "version": PACK_VERSION,
        "flags": flags,
        "model_names_embedded": item_names is not None,
        "input_bytes": sum(len(source.data) for source in sources),
        "output_bytes": len(output),
        "saved_bytes": sum(len(source.data) for source in sources) - len(output),
        "raw_model_candidates": raw_candidates,
        "unique_models": len(winners),
        "duplicate_or_non_renderable_model_rows_removed": raw_candidates - len(winners),
        "input_sources": len(sources),
        "kept_sources": len(kept_sources),
        "sources": {
            source.name: stats for _, source, _, stats in kept_sources
        },
    }
    return bytes(output), report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--title", type=Path)
    parser.add_argument("--wad", type=Path)
    parser.add_argument(
        "--names",
        type=Path,
        help="donor src/fixed/common.h; embeds the ITEM_TYPES names in the pack",
    )
    args = parser.parse_args()

    item_names = None
    if args.names:
        names_path = args.names.resolve()
        if not names_path.is_file():
            parser.error(f"common.h not found: {names_path}")
        item_names = read_item_types(names_path)

    sources = [parse_source(value) for value in args.source]
    if args.wad:
        if not args.title:
            parser.error("--wad requires --title")
        title = args.title.resolve()
        if not title.is_file():
            parser.error(f"TITLE.PKD not found: {title}")
        sources.append(Pkd("TITLE", title.read_bytes()))
        sources.extend(parse_wad(args.wad.resolve()))
    if not sources:
        parser.error("provide --source or --title/--wad")
    if len(sources) > 255:
        parser.error("too many sources")

    pack, report = build_pack(sources, item_names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pack)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"ASSET_VIEWER.PAK: {len(pack)} bytes")
    print(
        f"models: {report['raw_model_candidates']} candidates -> "
        f"{report['unique_models']} unique"
    )
    print(f"saved: {report['saved_bytes']} bytes from viewer source data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
