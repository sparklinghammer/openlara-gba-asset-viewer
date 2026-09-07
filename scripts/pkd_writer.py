#!/usr/bin/env python3
"""Assemble a viewer-only GBA PKD body.

Shared by both producers: `build_asset_pack.py` converts Tomb Raider level data,
`build_custom_pack.py` converts glTF. Neither the engine nor the loader can tell
them apart, because both emit exactly this layout.

The body is the structure `read_PKD` memcpy's into `Level`: a 172-byte header
holding 14 counters and 35 section offsets, then the sections in offset order.
Every gameplay table is present but empty; the viewer never reads them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


PKD_MAGIC = b"GBA "
PKD_HEADER_SIZE = 172
PALETTE_SIZE = 512          # 256 entries, BGR555
LIGHTMAP_SIZE = 256 * 32    # 32 shade rows remapping 256 palette indices
TILE_SIZE = 65536           # one 256x256 page of 8-bit indices
MESH_HEADER_SIZE = 20
MODEL_RECORD_SIZE = 8
ANIM_RECORD_SIZE = 32
NODE_RECORD_SIZE = 8
TEXTURE_RECORD_SIZE = 12
SPRITE_RECORD_SIZE = 16
GLYPHS_TYPE = 190
GLYPH_COUNT = 110


def align(buffer: bytearray, boundary: int = 4) -> int:
    while len(buffer) % boundary:
        buffer.append(0)
    return len(buffer)


@dataclass
class PkdParts:
    """Everything a viewer-only body needs, already in GBA byte order."""

    palette: bytes = b""                       # PALETTE_SIZE
    lightmap: bytes = b""                      # LIGHTMAP_SIZE
    tiles: list[bytes] = field(default_factory=list)          # TILE_SIZE each
    mesh_data: bytes = b""
    mesh_offsets: list[int] = field(default_factory=list)     # into mesh_data
    animations: bytes = b""                    # ANIM_RECORD_SIZE each
    nodes: bytes = b""                         # NODE_RECORD_SIZE each
    frames: bytes = b""
    models: bytes = b""                        # MODEL_RECORD_SIZE each
    textures: list[bytes] = field(default_factory=list)       # TEXTURE_RECORD_SIZE
    glyph_sprites: list[bytes] = field(default_factory=list)  # SPRITE_RECORD_SIZE

    def validate(self) -> None:
        if len(self.palette) != PALETTE_SIZE:
            raise ValueError(f"palette must be {PALETTE_SIZE} bytes, got {len(self.palette)}")
        if len(self.lightmap) != LIGHTMAP_SIZE:
            raise ValueError(f"lightmap must be {LIGHTMAP_SIZE} bytes, got {len(self.lightmap)}")
        for index, tile in enumerate(self.tiles):
            if len(tile) != TILE_SIZE:
                raise ValueError(f"tile {index} must be {TILE_SIZE} bytes, got {len(tile)}")
        if len(self.models) % MODEL_RECORD_SIZE:
            raise ValueError("model table is not a whole number of records")
        if len(self.animations) % ANIM_RECORD_SIZE:
            raise ValueError("animation table is not a whole number of records")
        if len(self.nodes) % NODE_RECORD_SIZE:
            raise ValueError("node table is not a whole number of records")
        for index, record in enumerate(self.textures):
            if len(record) != TEXTURE_RECORD_SIZE:
                raise ValueError(f"texture {index} must be {TEXTURE_RECORD_SIZE} bytes")
        # The viewer draws every string through the GLYPHS sprite sequence, so a
        # body without a complete glyph strip renders no text at all.
        if len(self.glyph_sprites) != GLYPH_COUNT:
            raise ValueError(f"expected {GLYPH_COUNT} glyph sprites, got {len(self.glyph_sprites)}")
        for index, record in enumerate(self.glyph_sprites):
            if len(record) != SPRITE_RECORD_SIZE:
                raise ValueError(f"glyph sprite {index} must be {SPRITE_RECORD_SIZE} bytes")
        for index, offset in enumerate(self.mesh_offsets):
            if offset < 0 or offset > len(self.mesh_data):
                raise ValueError(f"mesh offset {index} is outside mesh data")


def write_pkd_body(parts: PkdParts) -> bytes:
    """Serialise the parts into a loadable PKD body."""
    import struct

    parts.validate()

    output = bytearray(PKD_HEADER_SIZE)
    offsets = [0] * 35

    offsets[0] = align(output)
    output.extend(parts.palette)
    offsets[1] = align(output)
    output.extend(parts.lightmap)
    offsets[2] = align(output)
    for tile in parts.tiles:
        output.extend(tile)

    offsets[3] = align(output)     # rooms
    offsets[4] = len(output)       # floors
    offsets[5] = align(output)
    output.extend(parts.mesh_data)
    offsets[6] = align(output)
    for value in parts.mesh_offsets:
        output.extend(struct.pack("<I", value))
    offsets[7] = align(output)
    output.extend(parts.animations)
    offsets[8] = align(output)     # animStates: also bounds the anim table
    offsets[9] = len(output)       # animRanges
    offsets[10] = len(output)      # animCommands
    offsets[11] = align(output)
    output.extend(parts.nodes)
    offsets[12] = align(output)
    output.extend(parts.frames)
    offsets[13] = align(output)
    output.extend(parts.models)
    offsets[14] = align(output)    # staticMeshes
    offsets[15] = len(output)
    for record in parts.textures:
        output.extend(record)
    offsets[16] = align(output)
    for record in parts.glyph_sprites:
        output.extend(record)
    offsets[17] = align(output)
    output.extend(struct.pack("<HHhH", GLYPHS_TYPE, 0, -GLYPH_COUNT, 0))

    for index in range(18, 28):
        offsets[index] = align(output)
    offsets[28] = align(output)    # animTexData: a leading zero range count
    output.extend(b"\0\0")
    for index in range(29, 35):
        offsets[index] = align(output)

    counts = [
        len(parts.tiles),                       # tiles
        0,                                      # rooms
        len(parts.models) // MODEL_RECORD_SIZE, # models
        len(parts.mesh_offsets),                # meshes
        0,                                      # staticMeshes
        1,                                      # spriteSequences: GLYPHS
        0,                                      # soundSources
        0,                                      # boxes
        len(parts.textures),                    # textures
        len(parts.glyph_sprites),               # sprites
        0,                                      # items
        0,                                      # cameras
        0,                                      # cameraFrames
        0,                                      # soundOffsets
    ]
    output[0:4] = PKD_MAGIC
    struct.pack_into("<14H", output, 4, *counts)
    struct.pack_into("<35I", output, 32, *offsets)
    return bytes(output)


def body_stats(body: bytes) -> dict[str, int]:
    import struct

    counts = struct.unpack_from("<14H", body, 4)
    offsets = struct.unpack_from("<35I", body, 32)
    return {
        "bytes": len(body),
        "models": counts[2],
        "meshes": counts[3],
        "animations": (offsets[8] - offsets[7]) // ANIM_RECORD_SIZE,
        "textures": counts[8],
        "tiles": counts[0],
    }
