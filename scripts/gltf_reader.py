#!/usr/bin/env python3
"""Minimal glTF 2.0 / GLB reader, standard library only.

Covers exactly what the GBA target can consume: node hierarchies, triangle
primitives with positions and one UV set, PBR base colour and base colour
texture, and TRS animation channels. Anything outside that is reported as a
clear error rather than silently dropped, because a silent drop on this target
means a model that looks wrong on hardware with no diagnostic.
"""

from __future__ import annotations

import base64
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path


GLB_MAGIC = b"glTF"
CHUNK_JSON = b"JSON"
CHUNK_BIN = b"BIN\0"

COMPONENT = {
    5120: ("b", 1),   # BYTE
    5121: ("B", 1),   # UNSIGNED_BYTE
    5122: ("h", 2),   # SHORT
    5123: ("H", 2),   # UNSIGNED_SHORT
    5125: ("I", 4),   # UNSIGNED_INT
    5126: ("f", 4),   # FLOAT
}

TYPE_COUNT = {
    "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
    "MAT2": 4, "MAT3": 9, "MAT4": 16,
}

NORMALIZE_DIVISOR = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}

MODE_TRIANGLES = 4


class GltfError(Exception):
    pass


@dataclass
class Primitive:
    positions: list[tuple[float, float, float]]
    uvs: list[tuple[float, float]] | None
    indices: list[int]
    material: int | None
    joints: list[tuple[int, ...]] | None = None     # JOINTS_0
    weights: list[tuple[float, ...]] | None = None  # WEIGHTS_0
    # Kept only to tell which side of a face is the front. An open shell -- a
    # car panel, a hair card -- encloses no volume, so nothing in the geometry
    # says which way is out; the normals the author saved do.
    normals: list[tuple[float, float, float]] | None = None
    # The scene node whose frame each position is expressed in, once a skin has
    # been resolved. One per vertex, because that is what the source says: a
    # triangle may have each of its corners on a different bone, and the corner
    # is what follows the bone, not the triangle.
    bones: list[int] | None = None
    # For a vertex the source shares between several bones: every influence it
    # names, as (node, weight, position in that node's rest frame). None where
    # one bone carries the vertex outright, which is the usual case.
    blends: list[tuple | None] | None = None


@dataclass
class Node:
    index: int
    name: str
    children: list[int]
    mesh: int | None
    skin: int | None
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]   # x, y, z, w
    scale: tuple[float, float, float]


@dataclass
class Sampler:
    times: list[float]
    values: list[tuple[float, ...]]
    interpolation: str


@dataclass
class Channel:
    node: int
    path: str          # translation | rotation | scale | weights
    sampler: int


@dataclass
class Animation:
    name: str
    channels: list[Channel]
    samplers: list[Sampler]


@dataclass
class Material:
    index: int
    name: str
    base_color: tuple[float, float, float, float]
    base_color_texture: int | None       # index into `images`
    alpha_mode: str
    alpha_cutoff: float
    double_sided: bool
    # True when the colour above came from the material's emissive slot rather
    # than its base colour, which is worth saying out loud in the build log.
    from_emissive: bool = False


@dataclass
class Image:
    index: int
    name: str
    data: bytes
    mime_type: str


@dataclass
class Skin:
    joints: list[int]                                   # node indices
    inverse_bind: list[list[list[float]]]               # one 4x4 per joint


@dataclass
class Gltf:
    path: Path
    json: dict
    nodes: list[Node]
    skins: list[Skin]
    meshes: list[list[Primitive]]
    materials: list[Material]
    images: list[Image]
    animations: list[Animation]
    roots: list[int] = field(default_factory=list)


def _read_glb(data: bytes) -> tuple[dict, bytes]:
    if len(data) < 12 or data[:4] != GLB_MAGIC:
        raise GltfError("not a GLB container")
    _, version, total = struct.unpack_from("<4sII", data, 0)
    if version != 2:
        raise GltfError(f"unsupported GLB version {version}")
    if total > len(data):
        raise GltfError("GLB declares more bytes than the file holds")

    doc: dict | None = None
    binary = b""
    offset = 12
    while offset + 8 <= total:
        length, kind = struct.unpack_from("<I4s", data, offset)
        body = data[offset + 8:offset + 8 + length]
        if len(body) != length:
            raise GltfError("truncated GLB chunk")
        if kind == CHUNK_JSON and doc is None:
            doc = json.loads(body.decode("utf-8"))
        elif kind == CHUNK_BIN and not binary:
            binary = bytes(body)
        offset += 8 + length
        offset += (-offset) % 4
    if doc is None:
        raise GltfError("GLB has no JSON chunk")
    return doc, binary


def _resolve_buffers(doc: dict, base: Path, glb_binary: bytes) -> list[bytes]:
    buffers: list[bytes] = []
    for index, buffer in enumerate(doc.get("buffers", [])):
        uri = buffer.get("uri")
        if uri is None:
            if not glb_binary:
                raise GltfError(f"buffer {index} has no URI and there is no GLB chunk")
            buffers.append(glb_binary)
        elif uri.startswith("data:"):
            header, _, payload = uri.partition(",")
            if not header.endswith(";base64"):
                raise GltfError(f"buffer {index} uses an unsupported data URI encoding")
            buffers.append(base64.b64decode(payload))
        else:
            from urllib.parse import unquote

            target = (base / unquote(uri)).resolve()
            if not target.is_file():
                raise GltfError(f"buffer {index} points at a missing file: {target}")
            buffers.append(target.read_bytes())
    return buffers


def _read_accessor(doc: dict, buffers: list[bytes], index: int) -> list[tuple]:
    accessor = doc["accessors"][index]
    if "sparse" in accessor:
        raise GltfError(
            f"accessor {index} is sparse; re-export without sparse accessors"
        )
    component = accessor["componentType"]
    if component not in COMPONENT:
        raise GltfError(f"accessor {index} has unknown componentType {component}")
    fmt, size = COMPONENT[component]
    arity = TYPE_COUNT[accessor["type"]]
    count = accessor["count"]

    if "bufferView" not in accessor:
        return [tuple([0] * arity)] * count

    view = doc["bufferViews"][accessor["bufferView"]]
    blob = buffers[view.get("buffer", 0)]
    base = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride") or (size * arity)
    element = struct.Struct("<" + fmt * arity)

    values: list[tuple] = []
    for i in range(count):
        start = base + i * stride
        chunk = blob[start:start + element.size]
        if len(chunk) != element.size:
            raise GltfError(f"accessor {index} reads past the end of its buffer")
        values.append(element.unpack(chunk))

    if accessor.get("normalized") and component in NORMALIZE_DIVISOR:
        divisor = NORMALIZE_DIVISOR[component]
        if component in (5120, 5122):
            values = [tuple(max(v / divisor, -1.0) for v in row) for row in values]
        else:
            values = [tuple(v / divisor for v in row) for row in values]
    return values


def _read_nodes(doc: dict) -> list[Node]:
    nodes: list[Node] = []
    for index, raw in enumerate(doc.get("nodes", [])):
        if "matrix" in raw:
            raise GltfError(
                f"node {index} ({raw.get('name', '?')}) uses a raw matrix; "
                "re-export with TRS transforms (Blender does this by default)"
            )
        nodes.append(Node(
            index=index,
            name=raw.get("name", f"node_{index}"),
            children=list(raw.get("children", [])),
            mesh=raw.get("mesh"),
            skin=raw.get("skin"),
            translation=tuple(raw.get("translation", (0.0, 0.0, 0.0))),
            rotation=tuple(raw.get("rotation", (0.0, 0.0, 0.0, 1.0))),
            scale=tuple(raw.get("scale", (1.0, 1.0, 1.0))),
        ))
    return nodes


def _read_meshes(doc: dict, buffers: list[bytes]) -> list[list[Primitive]]:
    meshes: list[list[Primitive]] = []
    for mesh_index, raw in enumerate(doc.get("meshes", [])):
        primitives: list[Primitive] = []
        for prim_index, prim in enumerate(raw.get("primitives", [])):
            mode = prim.get("mode", MODE_TRIANGLES)
            if mode != MODE_TRIANGLES:
                raise GltfError(
                    f"mesh {mesh_index} primitive {prim_index} uses draw mode {mode}; "
                    "only triangles (mode 4) are supported"
                )
            attributes = prim.get("attributes", {})
            if "POSITION" not in attributes:
                raise GltfError(f"mesh {mesh_index} primitive {prim_index} has no POSITION")
            positions = [tuple(v) for v in _read_accessor(doc, buffers, attributes["POSITION"])]
            normals = None
            if "NORMAL" in attributes:
                normals = [tuple(v[:3]) for v in _read_accessor(doc, buffers, attributes["NORMAL"])]
                if len(normals) != len(positions):
                    normals = None
            uvs = None
            if "TEXCOORD_0" in attributes:
                uvs = [tuple(v) for v in _read_accessor(doc, buffers, attributes["TEXCOORD_0"])]
                if len(uvs) != len(positions):
                    raise GltfError(
                        f"mesh {mesh_index} primitive {prim_index}: UV count does not match positions"
                    )
            if "indices" in prim:
                indices = [v[0] for v in _read_accessor(doc, buffers, prim["indices"])]
            else:
                indices = list(range(len(positions)))
            if len(indices) % 3:
                raise GltfError(
                    f"mesh {mesh_index} primitive {prim_index}: index count is not a multiple of 3"
                )
            joints = weights = None
            if "JOINTS_0" in attributes and "WEIGHTS_0" in attributes:
                joints = [tuple(int(v) for v in row)
                          for row in _read_accessor(doc, buffers, attributes["JOINTS_0"])]
                weights = [tuple(float(v) for v in row)
                           for row in _read_accessor(doc, buffers, attributes["WEIGHTS_0"])]
            primitives.append(
                Primitive(positions, uvs, indices, prim.get("material"), joints, weights,
                          normals))
        meshes.append(primitives)
    return meshes


def _read_materials(doc: dict) -> list[Material]:
    textures = doc.get("textures", [])
    materials: list[Material] = []
    for index, raw in enumerate(doc.get("materials", [])):
        pbr = raw.get("pbrMetallicRoughness", {})

        def image_of(slot, where):
            if slot.get("texCoord", 0) != 0:
                raise GltfError(
                    f"material {index} samples TEXCOORD_{slot['texCoord']} for its "
                    f"{where}; only set 0 is supported"
                )
            source = textures[slot["index"]].get("source")
            if source is None:
                raise GltfError(
                    f"material {index} references a {where} with no image source")
            return source

        base_color = tuple(pbr.get("baseColorFactor", (1.0, 1.0, 1.0, 1.0)))
        emissive = tuple(raw.get("emissiveFactor", (0.0, 0.0, 0.0)))
        texture_index = None
        from_emissive = False

        if "baseColorTexture" in pbr:
            texture_index = image_of(pbr["baseColorTexture"], "base colour texture")
        elif "emissiveTexture" in raw:
            # An unlit model is usually authored by wiring the image into
            # Emission rather than Base Color, and Blender's exporter writes
            # exactly that, leaving the base colour black. The target has no
            # lighting at all, so emission is simply what the surface looks
            # like; reading it is the only sane thing to do with it, and not
            # reading it leaves a black model on a black background.
            texture_index = image_of(raw["emissiveTexture"], "emissive texture")
            from_emissive = True

        # Same reasoning without a texture: a black base colour beside a lit
        # emissive factor means the colour lives in the emissive slot.
        if texture_index is None and max(base_color[:3]) == 0.0 and max(emissive) > 0.0:
            base_color = tuple(emissive) + (base_color[3],)
            from_emissive = True

        materials.append(Material(
            index=index,
            name=raw.get("name", f"material_{index}"),
            base_color=base_color,
            base_color_texture=texture_index,
            alpha_mode=raw.get("alphaMode", "OPAQUE"),
            alpha_cutoff=float(raw.get("alphaCutoff", 0.5)),
            double_sided=bool(raw.get("doubleSided", False)),
            from_emissive=from_emissive,
        ))
    return materials


def _read_images(doc: dict, buffers: list[bytes], base: Path) -> list[Image]:
    images: list[Image] = []
    for index, raw in enumerate(doc.get("images", [])):
        name = raw.get("name", f"image_{index}")
        if "bufferView" in raw:
            view = doc["bufferViews"][raw["bufferView"]]
            blob = buffers[view.get("buffer", 0)]
            start = view.get("byteOffset", 0)
            data = blob[start:start + view["byteLength"]]
            mime = raw.get("mimeType", "")
        else:
            uri = raw.get("uri", "")
            if uri.startswith("data:"):
                header, _, payload = uri.partition(",")
                data = base64.b64decode(payload)
                mime = header[5:].split(";", 1)[0]
            else:
                from urllib.parse import unquote

                target = (base / unquote(uri)).resolve()
                if not target.is_file():
                    raise GltfError(f"image {index} points at a missing file: {target}")
                data = target.read_bytes()
                mime = raw.get("mimeType", "")
        images.append(Image(index=index, name=name, data=data, mime_type=mime))
    return images


def _read_skins(doc: dict, buffers: list[bytes]) -> list[Skin]:
    """Joint list and inverse bind matrices, as row-major 4x4."""
    skins: list[Skin] = []
    for raw in doc.get("skins", []):
        joints = list(raw.get("joints", []))
        matrices: list[list[list[float]]] = []
        if "inverseBindMatrices" in raw:
            for flat in _read_accessor(doc, buffers, raw["inverseBindMatrices"]):
                # glTF stores matrices column-major; transpose into row-major.
                matrices.append([[flat[column * 4 + row] for column in range(4)]
                                 for row in range(4)])
        else:
            matrices = [[[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
                        for _ in joints]
        skins.append(Skin(joints, matrices))
    return skins


def _read_animations(doc: dict, buffers: list[bytes]) -> list[Animation]:
    animations: list[Animation] = []
    for index, raw in enumerate(doc.get("animations", [])):
        samplers = []
        for sampler in raw.get("samplers", []):
            times = [v[0] for v in _read_accessor(doc, buffers, sampler["input"])]
            values = [tuple(v) for v in _read_accessor(doc, buffers, sampler["output"])]
            samplers.append(Sampler(times, values, sampler.get("interpolation", "LINEAR")))
        channels = []
        for channel in raw.get("channels", []):
            target = channel.get("target", {})
            if "node" not in target:
                continue
            channels.append(Channel(target["node"], target.get("path", ""), channel["sampler"]))
        animations.append(Animation(raw.get("name", f"animation_{index}"), channels, samplers))
    return animations


def load(path: Path) -> Gltf:
    path = Path(path).resolve()
    raw = path.read_bytes()
    if raw[:4] == GLB_MAGIC:
        doc, binary = _read_glb(raw)
    else:
        try:
            doc = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GltfError(
                f"{path.name} is neither a GLB container nor readable glTF JSON "
                f"({error}). A .glb that starts with anything but 'glTF' is usually "
                "a file that was renamed rather than exported."
            ) from error
        binary = b""

    asset_version = doc.get("asset", {}).get("version", "")
    if not asset_version.startswith("2."):
        raise GltfError(f"{path.name}: unsupported glTF version {asset_version!r}")

    buffers = _resolve_buffers(doc, path.parent, binary)
    nodes = _read_nodes(doc)

    scene_index = doc.get("scene", 0)
    scenes = doc.get("scenes", [])
    if scenes:
        roots = list(scenes[scene_index].get("nodes", []))
    else:
        child_of = {child for node in nodes for child in node.children}
        roots = [node.index for node in nodes if node.index not in child_of]

    return Gltf(
        path=path,
        json=doc,
        nodes=nodes,
        skins=_read_skins(doc, buffers),
        meshes=_read_meshes(doc, buffers),
        materials=_read_materials(doc),
        images=_read_images(doc, buffers, path.parent),
        animations=_read_animations(doc, buffers),
        roots=roots,
    )
