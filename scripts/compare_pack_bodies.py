#!/usr/bin/env python3
"""Prove that two asset packs carry byte-identical PKD bodies and model catalogs.

Container revisions (AVP1 v2 -> v3) must change only the header and the tables.
Any difference in a body or in a catalog row is a regression, not a format change.
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from pathlib import Path


PACK_MAGIC = b"AVP1"
SOURCE_ENTRY_SIZE = 16
MODEL_ENTRY_SIZE = 12
NAME_ENTRY_SIZE = 24
PACK_FLAG_NAMES = 1 << 1
HEADER_SIZE = {2: 24, 3: 32}


def read_pack(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    if len(data) < 24 or data[:4] != PACK_MAGIC:
        raise SystemExit(f"{path}: not an AVP1 pack")
    version, total, source_count, model_count, source_table, model_table = \
        struct.unpack_from("<IIHHII", data, 4)
    if version not in HEADER_SIZE:
        raise SystemExit(f"{path}: unsupported AVP1 version {version}")
    if total != len(data):
        raise SystemExit(f"{path}: declared size {total} != actual {len(data)}")

    flags = 0
    names: list[str] = []
    if version >= 3:
        if len(data) < 32:
            raise SystemExit(f"{path}: v3 header is truncated")
        flags, name_table = struct.unpack_from("<II", data, 24)
        if flags & PACK_FLAG_NAMES:
            end = name_table + model_count * NAME_ENTRY_SIZE
            if name_table == 0 or end > total:
                raise SystemExit(f"{path}: name table is out of range")
            for index in range(model_count):
                slot = data[name_table + index * NAME_ENTRY_SIZE:
                            name_table + (index + 1) * NAME_ENTRY_SIZE]
                if b"\0" not in slot:
                    raise SystemExit(f"{path}: name {index} is not NUL-terminated")
                names.append(slot.split(b"\0", 1)[0].decode("ascii"))

    sources = []
    for index in range(source_count):
        entry = source_table + index * SOURCE_ENTRY_SIZE
        name = data[entry:entry + 8].split(b"\0", 1)[0].decode("ascii")
        offset, size = struct.unpack_from("<II", data, entry + 8)
        if offset + size > total:
            raise SystemExit(f"{path}: source {name} is out of range")
        sources.append((name, data[offset:offset + size]))

    models = [
        struct.unpack_from("<BBBBHHI", data, model_table + index * MODEL_ENTRY_SIZE)
        for index in range(model_count)
    ]
    return {
        "path": path, "version": version, "flags": flags,
        "sources": sources, "models": models, "names": names, "bytes": len(data),
    }


def digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()[:16]


def read_item_types(path: Path) -> list[str]:
    import re

    text = path.read_text(encoding="utf-8")
    block = text.split("#define ITEM_TYPES(E)", 1)[1].split("enum ItemType", 1)[0]
    return re.findall(r"\bE\(\s*([A-Z0-9_]+)\s*\)", block)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--item-types",
        type=Path,
        help="donor src/fixed/common.h; asserts every embedded name equals "
             "MODEL_NAMES[type], i.e. the viewer title text is unchanged",
    )
    args = parser.parse_args()

    ref = read_pack(args.reference.resolve())
    cnd = read_pack(args.candidate.resolve())

    print(f"reference: v{ref['version']} flags=0x{ref['flags']:02X} "
          f"{ref['bytes']} bytes, {len(ref['sources'])} sources, {len(ref['models'])} models")
    print(f"candidate: v{cnd['version']} flags=0x{cnd['flags']:02X} "
          f"{cnd['bytes']} bytes, {len(cnd['sources'])} sources, {len(cnd['models'])} models")

    problems: list[str] = []

    ref_sources = {name: body for name, body in ref["sources"]}
    cnd_sources = {name: body for name, body in cnd["sources"]}
    if list(ref_sources) != list(cnd_sources):
        problems.append(f"source order differs: {list(ref_sources)} vs {list(cnd_sources)}")
    else:
        for name in ref_sources:
            a, b = ref_sources[name], cnd_sources[name]
            if a == b:
                print(f"  body {name:<8} identical  {len(a):>9} bytes  {digest(a)}")
            else:
                problems.append(
                    f"body {name} differs: {len(a)} vs {len(b)} bytes, "
                    f"{digest(a)} vs {digest(b)}"
                )

    if ref["models"] != cnd["models"]:
        differing = [
            (i, r, c) for i, (r, c) in enumerate(zip(ref["models"], cnd["models"])) if r != c
        ]
        problems.append(f"model catalog differs in {len(differing)} rows, first: {differing[:1]}")
    else:
        print(f"  model catalog identical ({len(ref['models'])} rows)")

    if cnd["names"]:
        print(f"  candidate carries {len(cnd['names'])} names, "
              f"e.g. {cnd['names'][:3]} ... {cnd['names'][-1:]}")
        if len(cnd["names"]) != len(cnd["models"]):
            problems.append("name table length does not match the model count")

        if args.item_types:
            item_types = read_item_types(args.item_types.resolve())
            mismatched = [
                (row[0], item_types[row[0]], name)
                for row, name in zip(cnd["models"], cnd["names"])
                if row[0] >= len(item_types) or item_types[row[0]] != name
            ]
            if mismatched:
                problems.append(
                    f"{len(mismatched)} embedded names differ from ITEM_TYPES, "
                    f"first: {mismatched[0]}"
                )
            else:
                print(f"  all {len(cnd['names'])} names equal MODEL_NAMES[type]: "
                      "viewer title text is unchanged")
    elif args.item_types:
        problems.append("--item-types given but the candidate carries no name table")

    if problems:
        for line in problems:
            print(f"FAIL: {line}", file=sys.stderr)
        return 1

    print("PASS: bodies and catalog are identical; only the container changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
