#!/usr/bin/env python3
"""Wavefront OBJ reader that hands back the same structures as the glTF reader.

Like the COLLADA reader, this produces `gltf_reader`'s Node, Primitive,
Material and Image objects, so the converter downstream never learns which
format a model came from.

OBJ is the plainest of the three and carries the least, which is what makes it
useful: no scene graph beyond `o`/`g` names, no skeleton, no animation, no
units and no declared up axis. What it does carry -- positions, one UV set,
faces and a material library -- it carries unambiguously, so an OBJ beside a
COLLADA export of the same model is a way to check the harder reader.

Three format details are dealt with here:

  * Each corner indexes position, UV and normal separately, so a shared buffer
    has to be rebuilt, one output vertex per distinct combination.
  * Indices are 1-based and may be negative, meaning "counted back from here".
  * The V axis runs up from the bottom-left, as in COLLADA; every image file
    and glTF run down from the top-left.
"""

from __future__ import annotations

from pathlib import Path

from gltf_reader import Gltf, GltfError, Image, Material, Node, Primitive

# Texture map statements may carry options before the filename
# (`map_Kd -s 1 1 1 -o 0 0 0 texture.png`). Options that take arguments have to
# be skipped by name, since a filename can otherwise look like an option value.
MAP_OPTION_ARGS = {
    "-blendu": 1, "-blendv": 1, "-boost": 1, "-mm": 2, "-o": 3, "-s": 3,
    "-t": 3, "-texres": 1, "-clamp": 1, "-bm": 1, "-imfchan": 1, "-type": 1,
}

IMAGE_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".bmp": "image/bmp", ".tga": "image/x-tga", ".gif": "image/gif",
}


class ObjError(GltfError):
    pass


def _split_lines(path: Path) -> list[str]:
    """OBJ and MTL are plain text, but exporters disagree about the encoding.

    A stray byte in a comment or a material name should not cost the whole
    model, so decoding falls back rather than raising.
    """
    raw = path.read_bytes()
    for encoding in ("utf-8", "cp1252"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    out: list[str] = []
    pending = ""
    for line in text.splitlines():
        line = pending + line.rstrip()
        pending = ""
        if line.endswith("\\"):          # a backslash continues onto the next line
            pending = line[:-1]
            continue
        out.append(line)
    if pending:
        out.append(pending)
    return out


def _map_filename(tokens: list[str]) -> str | None:
    """The filename in a `map_*` statement, past any leading options."""
    cursor = 0
    while cursor < len(tokens):
        token = tokens[cursor]
        if not token.startswith("-"):
            return " ".join(tokens[cursor:])       # names may hold spaces
        cursor += 1 + MAP_OPTION_ARGS.get(token, 1)
    return None


def _read_mtl(path: Path, base: Path):
    """Materials and the images they name, in declaration order."""
    materials: list[Material] = []
    images: list[Image] = []
    index_of: dict[str, int] = {}
    image_of: dict[str, int] = {}
    current: dict | None = None

    def flush() -> None:
        if current is None:
            return
        index_of[current["name"]] = len(materials)
        materials.append(Material(
            index=len(materials),
            name=current["name"],
            base_color=current["color"],
            base_color_texture=current["texture"],
            # A material with a transparency map or a `d` below one is asking
            # for the colour-key face type, but the converter decides that from
            # the atlas texels it actually sees, which is more reliable than a
            # flag an exporter may have written by habit.
            alpha_mode="OPAQUE",
            alpha_cutoff=0.5,
            # OBJ has no double-sided flag, so the converter's closedness test
            # decides, exactly as it does for COLLADA.
            double_sided=False,
        ))

    def image_index(name: str) -> int:
        key = name.replace("\\", "/")
        if key in image_of:
            return image_of[key]
        target = Path(key)
        if not target.is_absolute():
            target = base / target
        if not target.is_file():                    # exporters write stale paths
            stray = base / target.name
            if stray.is_file():
                target = stray
            else:
                raise ObjError(
                    f"material library {path.name} points at a missing texture: {key}. "
                    "OBJ keeps its images beside the file rather than inside it, so "
                    "the whole folder has to travel with the model."
                )
        image_of[key] = len(images)
        images.append(Image(index=len(images), name=target.stem,
                            data=target.read_bytes(),
                            mime_type=IMAGE_MIME.get(target.suffix.lower(), "")))
        return image_of[key]

    for line in _split_lines(path):
        tokens = line.split()
        if not tokens or tokens[0].startswith("#"):
            continue
        key, values = tokens[0].lower(), tokens[1:]
        if key == "newmtl":
            flush()
            current = {"name": " ".join(values) or f"material_{len(materials)}",
                       "color": (1.0, 1.0, 1.0, 1.0), "texture": None}
        elif current is None:
            continue
        elif key == "kd" and len(values) >= 3:
            r, g, b = (float(v) for v in values[:3])
            current["color"] = (r, g, b, current["color"][3])
        elif key == "d" and values:
            r, g, b, _ = current["color"]
            current["color"] = (r, g, b, float(values[0]))
        elif key == "tr" and values:                # the inverse of `d`
            r, g, b, _ = current["color"]
            current["color"] = (r, g, b, 1.0 - float(values[0]))
        elif key == "map_kd":
            name = _map_filename(values)
            if name:
                current["texture"] = image_index(name)
    flush()
    return materials, images, index_of


def load(path: Path) -> Gltf:
    path = Path(path).resolve()
    lines = _split_lines(path)

    positions: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    normals: list[tuple[float, float, float]] = []
    materials: list[Material] = []
    images: list[Image] = []
    material_index: dict[str, int] = {}

    # One group per (object, material) pair: the engine draws one material at a
    # time, and an object with several is several meshes as far as it cares.
    groups: dict[tuple[str, int | None], dict] = {}
    order: list[tuple[str, int | None]] = []
    current_object = path.stem
    current_material: int | None = None
    unknown: set[str] = set()

    def group() -> dict:
        key = (current_object, current_material)
        entry = groups.get(key)
        if entry is None:
            entry = {"positions": [], "uvs": [], "normals": [], "indices": [],
                     "slots": {}, "has_uv": False, "has_normal": False}
            groups[key] = entry
            order.append(key)
        return entry

    def resolve(reference: str, total: int, what: str) -> int:
        value = int(reference)
        if value > 0:
            index = value - 1
        elif value < 0:
            index = total + value          # negative counts back from here
        else:
            raise ObjError(f"{path.name}: index 0 is not valid in an OBJ {what}")
        if not 0 <= index < total:
            raise ObjError(f"{path.name}: {what} index {value} is out of range")
        return index

    for number, line in enumerate(lines, 1):
        tokens = line.split()
        if not tokens or tokens[0].startswith("#"):
            continue
        key, values = tokens[0], tokens[1:]

        if key == "v":
            if len(values) < 3:
                raise ObjError(f"{path.name}:{number}: a vertex needs three numbers")
            positions.append(tuple(float(v) for v in values[:3]))
        elif key == "vn":
            # Kept only to tell which side of a face is the front: an open shell
            # encloses no volume, so nothing else in the file answers that.
            normals.append(tuple(float(v) for v in values[:3]))
        elif key == "vt":
            u = float(values[0]) if values else 0.0
            v = float(values[1]) if len(values) > 1 else 0.0
            # OBJ's V axis runs up from the bottom-left, as COLLADA's does.
            uvs.append((u, 1.0 - v))
        elif key in ("o", "g"):
            current_object = " ".join(values) or path.stem
        elif key == "usemtl":
            name = " ".join(values)
            if name in material_index:
                current_material = material_index[name]
            else:
                current_material = None
                if name and name not in unknown:
                    unknown.add(name)
        elif key == "mtllib":
            for name in values:
                library = path.parent / name
                if not library.is_file():
                    raise ObjError(
                        f"{path.name} names a material library that is missing: {name}"
                    )
                more_materials, more_images, more_index = _read_mtl(library, path.parent)
                shift, image_shift = len(materials), len(images)
                for material in more_materials:
                    material.index += shift
                    if material.base_color_texture is not None:
                        material.base_color_texture += image_shift
                    materials.append(material)
                for image in more_images:
                    image.index += image_shift
                    images.append(image)
                material_index.update({n: i + shift for n, i in more_index.items()})
        elif key == "f":
            if len(values) < 3:
                raise ObjError(f"{path.name}:{number}: a face needs three corners")
            entry = group()
            corners: list[int] = []
            for corner in values:
                parts = corner.split("/")
                pi = resolve(parts[0], len(positions), "vertex")
                ui = -1
                if len(parts) > 1 and parts[1]:
                    ui = resolve(parts[1], len(uvs), "texture coordinate")
                ni = -1
                if len(parts) > 2 and parts[2]:
                    ni = resolve(parts[2], len(normals), "normal")
                slot = entry["slots"].get((pi, ui))
                if slot is None:
                    slot = len(entry["positions"])
                    entry["slots"][(pi, ui)] = slot
                    entry["positions"].append(positions[pi])
                    entry["uvs"].append(uvs[ui] if ui >= 0 else (0.0, 0.0))
                    # One normal per output vertex, from the first corner that
                    # made it; splitting vertices by normal as well would
                    # multiply them for nothing, the target having no lighting.
                    entry["normals"].append(normals[ni] if ni >= 0 else (0.0, 0.0, 0.0))
                    if ui >= 0:
                        entry["has_uv"] = True
                    if ni >= 0:
                        entry["has_normal"] = True
                corners.append(slot)
            for fan in range(1, len(corners) - 1):      # quads and ngons as fans
                entry["indices"].extend([corners[0], corners[fan], corners[fan + 1]])

    if not positions:
        raise ObjError(f"{path.name}: no vertices")
    if unknown:
        names = ", ".join(sorted(unknown)[:4])
        raise ObjError(
            f"{path.name}: faces use materials the library does not define ({names}). "
            "The .mtl beside the model is probably not the one it was exported with."
        )

    nodes: list[Node] = []
    meshes: list[list[Primitive]] = []
    roots: list[int] = []
    by_object: dict[str, list[Primitive]] = {}
    for name, material in order:
        entry = groups[(name, material)]
        if not entry["indices"]:
            continue
        by_object.setdefault(name, []).append(Primitive(
            positions=entry["positions"],
            uvs=entry["uvs"] if entry["has_uv"] else None,
            indices=entry["indices"],
            material=material,
            normals=entry["normals"] if entry["has_normal"] else None,
        ))

    for name, primitives in by_object.items():
        index = len(nodes)
        nodes.append(Node(index=index, name=name, children=[], mesh=len(meshes),
                          skin=None, translation=(0.0, 0.0, 0.0),
                          rotation=(0.0, 0.0, 0.0, 1.0), scale=(1.0, 1.0, 1.0)))
        meshes.append(primitives)
        roots.append(index)

    if not nodes:
        raise ObjError(f"{path.name}: no faces")

    # OBJ states neither units nor an up axis. Y-up is the convention every
    # exporter follows, so nothing is rotated here; a model that arrives on its
    # side is corrected with `rotate` in pack.json, as with any other format.
    return Gltf(path=path, json={}, nodes=nodes, skins=[], meshes=meshes,
                materials=materials, images=images, animations=[], roots=roots)
