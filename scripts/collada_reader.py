#!/usr/bin/env python3
"""COLLADA (.dae) reader that hands back the same structures as the glTF reader.

The converter downstream never learns which format a model came from: this
module produces `gltf_reader`'s Node, Primitive, Material, Image, Skin and
Animation objects, so both formats meet at the same place.

Three things differ from glTF and are dealt with here rather than downstream:

  * COLLADA indexes each attribute separately, so positions and UVs have to be
    de-indexed into the single shared buffer a mesh needs.
  * Node transforms are matrices, sometimes a stack of translate/rotate/scale
    elements. They are composed and decomposed into the TRS the pipeline uses.
  * The file declares its own up axis. A Z-up document is rotated into Y-up on
    load, which is the one orientation question glTF leaves to guesswork.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlparse

from gltf_reader import (
    Animation, Channel, Gltf, GltfError, Image, Material, Node, Primitive,
    Sampler, Skin,
)

NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
IDENTITY4 = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]

# COLLADA states its up axis; Y-up is what the rest of the pipeline expects.
Z_UP_TO_Y_UP = [[1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]]
X_UP_TO_Y_UP = [[0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]]


class ColladaError(GltfError):
    pass


def _local(element) -> str:
    return element.tag.split("}")[-1]


def _floats(text: str | None) -> list[float]:
    return [float(v) for v in text.split()] if text and text.strip() else []


def _ints(text: str | None) -> list[int]:
    return [int(v) for v in text.split()] if text and text.strip() else []


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _decompose(matrix, label: str):
    """Split a node matrix into the translation, quaternion and scale a Node holds."""
    translation = (matrix[0][3], matrix[1][3], matrix[2][3])
    columns = [[matrix[row][col] for row in range(3)] for col in range(3)]
    scale = [math.sqrt(sum(v * v for v in column)) or 1.0 for column in columns]

    basis = [[columns[col][row] / scale[col] for col in range(3)] for row in range(3)]
    if (basis[0][0] * (basis[1][1] * basis[2][2] - basis[1][2] * basis[2][1])
            - basis[0][1] * (basis[1][0] * basis[2][2] - basis[1][2] * basis[2][0])
            + basis[0][2] * (basis[1][0] * basis[2][1] - basis[1][1] * basis[2][0])) < 0:
        # A mirrored node cannot be expressed as a rotation, and the engine has
        # no way to render one; say so rather than emit a quietly wrong pose.
        raise ColladaError(f"node {label!r} has a mirrored transform, which cannot be converted")

    trace = basis[0][0] + basis[1][1] + basis[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w, x = 0.25 * s, (basis[2][1] - basis[1][2]) / s
        y, z = (basis[0][2] - basis[2][0]) / s, (basis[1][0] - basis[0][1]) / s
    elif basis[0][0] > basis[1][1] and basis[0][0] > basis[2][2]:
        s = math.sqrt(1.0 + basis[0][0] - basis[1][1] - basis[2][2]) * 2
        w, x = (basis[2][1] - basis[1][2]) / s, 0.25 * s
        y, z = (basis[0][1] + basis[1][0]) / s, (basis[0][2] + basis[2][0]) / s
    elif basis[1][1] > basis[2][2]:
        s = math.sqrt(1.0 + basis[1][1] - basis[0][0] - basis[2][2]) * 2
        w, x = (basis[0][2] - basis[2][0]) / s, (basis[0][1] + basis[1][0]) / s
        y, z = 0.25 * s, (basis[1][2] + basis[2][1]) / s
    else:
        s = math.sqrt(1.0 + basis[2][2] - basis[0][0] - basis[1][1]) * 2
        w, x = (basis[1][0] - basis[0][1]) / s, (basis[0][2] + basis[2][0]) / s
        y, z = (basis[1][2] + basis[2][1]) / s, 0.25 * s
    length = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    return translation, (x / length, y / length, z / length, w / length), tuple(scale)


def _node_matrix(node) -> list[list[float]]:
    """Compose a node's transform elements, in document order as the spec requires."""
    out = IDENTITY4
    for child in node:
        kind = _local(child)
        if kind == "matrix":
            values = _floats(child.text)
            if len(values) != 16:
                raise ColladaError("a <matrix> element does not hold 16 numbers")
            out = _matmul(out, [values[i * 4:(i + 1) * 4] for i in range(4)])
        elif kind == "translate":
            x, y, z = _floats(child.text)[:3]
            out = _matmul(out, [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
        elif kind == "rotate":
            values = _floats(child.text)
            if len(values) < 4:
                continue
            ax, ay, az, degrees = values[:4]
            length = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
            ax, ay, az = ax / length, ay / length, az / length
            angle = math.radians(degrees)
            c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
            out = _matmul(out, [
                [t * ax * ax + c, t * ax * ay - s * az, t * ax * az + s * ay, 0.0],
                [t * ax * ay + s * az, t * ay * ay + c, t * ay * az - s * ax, 0.0],
                [t * ax * az - s * ay, t * ay * az + s * ax, t * az * az + c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
        elif kind == "scale":
            x, y, z = _floats(child.text)[:3]
            out = _matmul(out, [[x, 0, 0, 0], [0, y, 0, 0], [0, 0, z, 0], [0, 0, 0, 1]])
    return out


class _Document:
    def __init__(self, path: Path):
        self.path = path
        self.root = ET.parse(path).getroot()
        self.by_id: dict[str, ET.Element] = {}
        for element in self.root.iter():
            ident = element.get("id")
            if ident:
                self.by_id[ident] = element

    def find(self, url: str | None):
        if not url:
            return None
        return self.by_id.get(url.lstrip("#"))

    def source_values(self, element) -> tuple[list[float], int]:
        """A <source>'s numbers, plus how many make one element."""
        accessor = element.find("c:technique_common/c:accessor", NS)
        stride = int(accessor.get("stride", "1")) if accessor is not None else 1
        array = element.find("c:float_array", NS)
        if array is None:
            names = element.find("c:Name_array", NS)
            if names is not None:
                return [], stride
            raise ColladaError(f"source {element.get('id')} holds no float array")
        return _floats(array.text), stride

    def source_names(self, element) -> list[str]:
        array = element.find("c:Name_array", NS)
        return array.text.split() if array is not None and array.text else []


def _read_images(doc: _Document, base: Path) -> tuple[list[Image], dict[str, int]]:
    images: list[Image] = []
    index_of: dict[str, int] = {}
    for element in doc.root.findall("c:library_images/c:image", NS):
        source = element.find("c:init_from", NS)
        raw = (source.text or "").strip() if source is not None else ""
        if not raw:
            continue
        parsed = urlparse(raw)
        if parsed.scheme == "file":
            target = Path(unquote(parsed.path).lstrip("/"))
        else:
            target = Path(unquote(raw))
        if not target.is_absolute():
            target = base / target
        if not target.is_file():
            raise ColladaError(
                f"image {element.get('id')!r} points at a missing file: {target}. "
                "COLLADA keeps textures beside the .dae rather than inside it, so the "
                "whole folder has to travel with the model."
            )
        index_of[element.get("id", "")] = len(images)
        images.append(Image(index=len(images), name=element.get("id", target.stem),
                            data=target.read_bytes(), mime_type=""))
    return images, index_of


def _effect_diffuse(doc: _Document, effect, image_index: dict[str, int]):
    """The diffuse colour or texture of an effect's common profile."""
    profile = effect.find("c:profile_COMMON", NS)
    if profile is None:
        return (1.0, 1.0, 1.0, 1.0), None

    samplers: dict[str, str] = {}
    surfaces: dict[str, str] = {}
    for param in profile.findall("c:newparam", NS):
        sid = param.get("sid", "")
        surface = param.find("c:surface/c:init_from", NS)
        if surface is not None:
            surfaces[sid] = (surface.text or "").strip()
        sampler = param.find("c:sampler2D/c:source", NS)
        if sampler is not None:
            samplers[sid] = (sampler.text or "").strip()

    technique = profile.find("c:technique", NS)
    if technique is None:
        return (1.0, 1.0, 1.0, 1.0), None
    for shading in ("lambert", "phong", "blinn", "constant"):
        model = technique.find(f"c:{shading}", NS)
        if model is None:
            continue
        diffuse = model.find("c:diffuse", NS)
        if diffuse is None:
            continue
        texture = diffuse.find("c:texture", NS)
        if texture is not None:
            name = texture.get("texture", "")
            # The reference may be a sampler, a surface, or the image directly.
            name = samplers.get(name, name)
            name = surfaces.get(name, name)
            if name in image_index:
                return (1.0, 1.0, 1.0, 1.0), image_index[name]
            raise ColladaError(f"effect {effect.get('id')!r} samples an unknown image {name!r}")
        colour = diffuse.find("c:color", NS)
        if colour is not None:
            values = _floats(colour.text)
            while len(values) < 4:
                values.append(1.0)
            return tuple(values[:4]), None
    return (1.0, 1.0, 1.0, 1.0), None


def _read_materials(doc: _Document, image_index: dict[str, int]):
    materials: list[Material] = []
    index_of: dict[str, int] = {}
    for element in doc.root.findall("c:library_materials/c:material", NS):
        instance = element.find("c:instance_effect", NS)
        effect = doc.find(instance.get("url") if instance is not None else None)
        base_color, texture = ((1.0, 1.0, 1.0, 1.0), None)
        if effect is not None:
            base_color, texture = _effect_diffuse(doc, effect, image_index)
        index_of[element.get("id", "")] = len(materials)
        materials.append(Material(
            index=len(materials),
            name=element.get("name") or element.get("id", f"material_{len(materials)}"),
            base_color=base_color,
            base_color_texture=texture,
            alpha_mode="OPAQUE",
            alpha_cutoff=0.5,
            # COLLADA has no double-sided flag in the common profile. Treating
            # every surface as two-sided would double each model's face count,
            # so the converter's own closedness test decides instead.
            double_sided=False,
        ))
    return materials, index_of


def _primitive_blocks(mesh) -> list[ET.Element]:
    blocks = []
    for child in mesh:
        if _local(child) in ("triangles", "polylist", "polygons"):
            blocks.append(child)
        elif _local(child) in ("trifans", "tristrips", "lines", "linestrips"):
            raise ColladaError(
                f"<{_local(child)}> is not supported; re-export with triangles or polygons"
            )
    return blocks


def _read_influences(doc: _Document, skin) -> list[list[tuple[int, float]]]:
    """Per-position joint influences, in the order <vertex_weights> lists them."""
    element = skin.find("c:vertex_weights", NS)
    if element is None:
        return []
    joint_offset = weight_offset = 0
    weights: list[float] = []
    stride = 0
    for entry in element.findall("c:input", NS):
        offset = int(entry.get("offset", "0"))
        stride = max(stride, offset + 1)
        if entry.get("semantic") == "JOINT":
            joint_offset = offset
        elif entry.get("semantic") == "WEIGHT":
            weight_offset = offset
            source = doc.find(entry.get("source"))
            if source is not None:
                weights, _ = doc.source_values(source)

    counts = _ints(element.find("c:vcount", NS).text) if element.find("c:vcount", NS) is not None else []
    pairs = _ints(element.find("c:v", NS).text) if element.find("c:v", NS) is not None else []

    influences: list[list[tuple[int, float]]] = []
    cursor = 0
    for count in counts:
        entry: list[tuple[int, float]] = []
        for _ in range(count):
            base = cursor * stride
            joint = pairs[base + joint_offset]
            index = pairs[base + weight_offset]
            entry.append((joint, weights[index] if 0 <= index < len(weights) else 0.0))
            cursor += 1
        influences.append(entry)
    return influences


def _read_geometry(doc: _Document, geometry, material_index, bindings,
                   influences=None) -> list[Primitive]:
    mesh = geometry.find("c:mesh", NS)
    if mesh is None:
        return []
    primitives: list[Primitive] = []
    for block in _primitive_blocks(mesh):
        inputs = {}
        stride = 0
        for element in block.findall("c:input", NS):
            offset = int(element.get("offset", "0"))
            stride = max(stride, offset + 1)
            semantic = element.get("semantic", "")
            if semantic == "TEXCOORD" and int(element.get("set", "0") or 0) != 0:
                continue          # only the first UV set reaches the target
            inputs[semantic] = (offset, element.get("source", ""))
        if "VERTEX" not in inputs:
            raise ColladaError(f"geometry {geometry.get('id')!r} has a block with no VERTEX input")

        vertex_element = doc.find(inputs["VERTEX"][1])
        position_source = None
        shared_uv_source = None
        shared_normal_source = None
        if vertex_element is not None:
            for element in vertex_element.findall("c:input", NS):
                if element.get("semantic") == "POSITION":
                    position_source = doc.find(element.get("source"))
                elif element.get("semantic") == "NORMAL":
                    shared_normal_source = doc.find(element.get("source"))
                elif element.get("semantic") == "TEXCOORD":
                    # <vertices> may declare any per-vertex input, and a VERTEX
                    # input pulls all of them in at the same index. Exporters
                    # that share one UV per position write them here rather than
                    # on the primitive, which is equally legal and easy to miss.
                    shared_uv_source = doc.find(element.get("source"))
        if position_source is None:
            raise ColladaError(f"geometry {geometry.get('id')!r} has no POSITION source")
        positions, position_stride = doc.source_values(position_source)

        uv_values: list[float] = []
        uv_stride = 2
        if "TEXCOORD" in inputs:
            uv_source = doc.find(inputs["TEXCOORD"][1])
            if uv_source is not None:
                uv_values, uv_stride = doc.source_values(uv_source)
        elif shared_uv_source is not None:
            uv_values, uv_stride = doc.source_values(shared_uv_source)
            inputs["TEXCOORD"] = (inputs["VERTEX"][0], "")   # indexed as the position is

        # Normals say which side of a face is the front. They are kept only for
        # that: an open shell encloses no volume, so nothing else in the file
        # answers the question, and guessing at it turns models inside out.
        normal_values: list[float] = []
        normal_stride = 3
        normal_offset = None
        if "NORMAL" in inputs:
            normal_source = doc.find(inputs["NORMAL"][1])
            if normal_source is not None:
                normal_values, normal_stride = doc.source_values(normal_source)
                normal_offset = inputs["NORMAL"][0]
        elif shared_normal_source is not None:
            normal_values, normal_stride = doc.source_values(shared_normal_source)
            normal_offset = inputs["VERTEX"][0]      # indexed as the position is

        indices = _ints(block.find("c:p", NS).text if block.find("c:p", NS) is not None else "")
        counts = _ints(block.find("c:vcount", NS).text) if block.find("c:vcount", NS) is not None else None
        if _local(block) == "polygons":
            indices = []
            counts = []
            for polygon in block.findall("c:p", NS):
                values = _ints(polygon.text)
                counts.append(len(values) // max(stride, 1))
                indices.extend(values)
        if counts is None:
            counts = [3] * (len(indices) // (3 * max(stride, 1)))

        # COLLADA indexes each attribute on its own, so a shared buffer has to be
        # rebuilt: one output vertex per distinct combination of indices.
        out_positions: list[tuple[float, float, float]] = []
        out_uvs: list[tuple[float, float]] = []
        out_normals: list[tuple[float, float, float]] = []
        out_positions_index: list[int] = []
        slot_of: dict[tuple[int, int], int] = {}
        triangles: list[int] = []
        position_offset = inputs["VERTEX"][0]
        uv_offset = inputs["TEXCOORD"][0] if "TEXCOORD" in inputs else None

        cursor = 0
        for corner_count in counts:
            corners = []
            for corner in range(corner_count):
                base = (cursor + corner) * stride
                pi = indices[base + position_offset]
                ui = indices[base + uv_offset] if uv_offset is not None else -1
                key = (pi, ui)
                slot = slot_of.get(key)
                if slot is None:
                    slot = len(out_positions)
                    slot_of[key] = slot
                    out_positions.append(tuple(
                        positions[pi * position_stride:pi * position_stride + 3]))
                    out_positions_index.append(pi)
                    # One normal per output vertex, from the first corner that
                    # made it. Splitting vertices by normal too would multiply
                    # them for no gain: the target has no lighting, and a side
                    # is all that is being asked of them.
                    if normal_offset is not None and normal_values:
                        ni = indices[base + normal_offset]
                        out_normals.append(tuple(
                            normal_values[ni * normal_stride:ni * normal_stride + 3]))
                    else:
                        out_normals.append((0.0, 0.0, 0.0))
                    if uv_offset is not None and uv_values:
                        u, v = uv_values[ui * uv_stride:ui * uv_stride + 2]
                        # COLLADA's T axis runs up from the bottom-left; glTF and
                        # every image file run down from the top-left.
                        out_uvs.append((u, 1.0 - v))
                    else:
                        out_uvs.append((0.0, 0.0))
                corners.append(slot)
            for fan in range(1, corner_count - 1):        # triangulate as a fan
                triangles.extend([corners[0], corners[fan], corners[fan + 1]])
            cursor += corner_count

        # Skin weights are listed per position, so they follow the de-indexing.
        # The target has no skinning at all, but the rest pose has to be baked,
        # and on this format that pose lives in the skin matrices: a bind shape
        # often carries the whole model's scale.
        out_joints = out_weights = None
        if influences:
            out_joints = []
            out_weights = []
            for pi in out_positions_index:
                entry = sorted(influences[pi] if pi < len(influences) else [],
                               key=lambda pair: -pair[1])[:4]
                total = sum(weight for _, weight in entry) or 1.0
                bones = [bone for bone, _ in entry] + [0] * (4 - len(entry))
                shares = [weight / total for _, weight in entry] + [0.0] * (4 - len(entry))
                out_joints.append(tuple(bones))
                out_weights.append(tuple(shares))

        symbol = block.get("material")
        target = bindings.get(symbol, symbol)
        material = material_index.get(target)
        primitives.append(Primitive(
            positions=out_positions,
            uvs=out_uvs if uv_offset is not None else None,
            indices=triangles,
            material=material,
            joints=out_joints,
            weights=out_weights,
            normals=out_normals if normal_offset is not None and normal_values else None,
        ))
    return primitives


def _read_skin(doc: _Document, controller, joint_node_index):
    """Return (Skin, geometry, per-position influences) or None."""
    skin = controller.find("c:skin", NS)
    if skin is None:
        return None
    geometry = doc.find(skin.get("source"))
    if geometry is None:
        return None

    bind_shape = IDENTITY4
    element = skin.find("c:bind_shape_matrix", NS)
    if element is not None:
        values = _floats(element.text)
        if len(values) == 16:
            bind_shape = [values[i * 4:(i + 1) * 4] for i in range(4)]

    joints_element = skin.find("c:joints", NS)
    names: list[str] = []
    inverse_bind: list[list[list[float]]] = []
    if joints_element is not None:
        for entry in joints_element.findall("c:input", NS):
            source = doc.find(entry.get("source"))
            if source is None:
                continue
            if entry.get("semantic") == "JOINT":
                names = doc.source_names(source)
            elif entry.get("semantic") == "INV_BIND_MATRIX":
                values, _ = doc.source_values(source)
                inverse_bind = [
                    [values[i * 16 + r * 4:i * 16 + r * 4 + 4] for r in range(4)]
                    for i in range(len(values) // 16)
                ]
    while len(inverse_bind) < len(names):
        inverse_bind.append([row[:] for row in IDENTITY4])

    # Folding the bind shape into each inverse bind matrix makes the result match
    # glTF's skin matrix exactly, so the converter needs no special case.
    inverse_bind = [_matmul(m, bind_shape) for m in inverse_bind]
    joints = [joint_node_index.get(name, -1) for name in names]
    return Skin(joints=joints, inverse_bind=inverse_bind), geometry, _read_influences(doc, skin)


def _sampler_sources(doc, sampler) -> dict[str, tuple]:
    """The INPUT/OUTPUT/INTERPOLATION sources a sampler names, already read."""
    out: dict[str, tuple] = {}
    for element in sampler.findall("c:input", NS):
        semantic = element.get("semantic", "")
        source = doc.find(element.get("source"))
        if source is None:
            continue
        if semantic == "INTERPOLATION":
            out[semantic] = (doc.source_names(source), 1)
        else:
            out[semantic] = doc.source_values(source)
    return out


def _channel_target(target: str) -> tuple[str, str]:
    """Split `joint0/transform` into the node it names and the member it drives."""
    node, _, member = target.partition("/")
    member = member.split("(")[0]          # `transform(3)(1)` drives one cell
    return node, member


def _quaternion_run(quaternions: list) -> list:
    """Make a sequence of quaternions continuous.

    Decomposing each keyframe on its own can hand back q or -q for the same
    rotation, and the two slerp opposite ways round. Flipping each key to the
    near side of the one before it is what stops a joint spinning the long way
    between two frames that barely differ.
    """
    out = []
    for quaternion in quaternions:
        if out and sum(a * b for a, b in zip(quaternion, out[-1])) < 0.0:
            quaternion = tuple(-v for v in quaternion)
        out.append(tuple(quaternion))
    return out


def _read_animations(doc: _Document, joint_node_index: dict[str, int],
                     node_depth: dict[int, int], conversion,
                     notes: list[str] | None = None) -> list[Animation]:
    """Read the animation library into the TRS channels the pipeline consumes.

    COLLADA usually keys a whole 4x4 per frame -- that is what every exporter
    writes for a skeleton -- while the engine stores a rotation per joint and a
    position for the root. Each keyed matrix is therefore decomposed here, at
    load time, so the resampler downstream sees the same shape it sees from
    glTF and needs to know nothing about the format.

    `library_animation_clips` names the takes and says which per-joint tracks
    belong to each. Without it every track is one animation, which is what a
    single-take export means.

    A keyed scale is dropped rather than refused. The engine's matrix stack
    holds a rotation and a position per joint and has nowhere to put a scale, so
    no conversion exists; refusing would throw away every other take in the file
    over an effect a few joints use. What is dropped is counted and reported.
    """
    library = doc.root.find("c:library_animations", NS)
    if library is None:
        return []

    tracks: dict[str, dict] = {}
    scaled: dict[str, float] = {}

    def read(element) -> None:
        for nested in element.findall("c:animation", NS):
            read(nested)
        channel = element.find("c:channel", NS)
        sampler_element = None
        if channel is not None:
            sampler_element = doc.find(channel.get("source"))
        if channel is None or sampler_element is None:
            return
        name, member = _channel_target(channel.get("target", ""))
        node = joint_node_index.get(name)
        if node is None:
            return                                  # a track for a pruned node
        sources = _sampler_sources(doc, sampler_element)
        if "INPUT" not in sources or "OUTPUT" not in sources:
            return
        times, _ = sources["INPUT"]
        values, stride = sources["OUTPUT"]
        interpolation = "LINEAR"
        if "INTERPOLATION" in sources:
            names = sources["INTERPOLATION"][0]
            if names and names[0] == "STEP":
                interpolation = "STEP"

        entry = {"node": node, "times": list(times), "member": member,
                 "interpolation": interpolation}
        if stride == 16:
            matrices = [[values[k * 16 + r * 4: k * 16 + r * 4 + 4] for r in range(4)]
                        for k in range(len(times))]
            if node_depth.get(node) == 0:
                matrices = [_matmul(conversion, m) for m in matrices]
            positions, quaternions = [], []
            for key, matrix in enumerate(matrices):
                translation, rotation, scale = _decompose(matrix, f"{name} key {key}")
                worst = max(abs(s - 1.0) for s in scale)
                if worst > 0.01:
                    scaled[name] = max(scaled.get(name, 0.0), worst)
                positions.append(translation)
                quaternions.append(rotation)
            entry["rotation"] = _quaternion_run(quaternions)
            entry["translation"] = positions
        elif stride == 3 and member in ("translate", "translation", "location"):
            entry["translation"] = [tuple(values[k * 3:k * 3 + 3])
                                    for k in range(len(times))]
        else:
            raise ColladaError(
                f"animation {element.get('id')!r} drives {member!r} with {stride} "
                "value(s) per key; only a full 4x4 transform or a translation can be "
                "converted, because the engine stores one rotation per joint"
            )
        tracks[element.get("id", f"track{len(tracks)}")] = entry

    read(library)
    if not tracks:
        return []

    clips = doc.root.find("c:library_animation_clips", NS)
    grouped: list[tuple[str, list[str]]] = []
    if clips is not None:
        for clip in clips.findall("c:animation_clip", NS):
            members = [(instance.get("url") or "").lstrip("#")
                       for instance in clip.findall("c:instance_animation", NS)]
            members = [m for m in members if m in tracks]
            if members:
                grouped.append((clip.get("name") or clip.get("id", ""), members))
    if not grouped:
        grouped = [("", list(tracks))]

    animations: list[Animation] = []
    for name, members in grouped:
        samplers: list[Sampler] = []
        channels: list[Channel] = []
        # A clip's own zero is wherever its keys start; the engine plays every
        # take from tick zero, so the times are rebased here.
        origin = min(tracks[m]["times"][0] for m in members if tracks[m]["times"])
        for member in members:
            entry = tracks[member]
            times = [t - origin for t in entry["times"]]
            for path in ("rotation", "translation"):
                if path not in entry:
                    continue
                samplers.append(Sampler(times=times, values=entry[path],
                                        interpolation=entry["interpolation"]))
                channels.append(Channel(node=entry["node"], path=path,
                                        sampler=len(samplers) - 1))
        animations.append(Animation(name=name, channels=channels, samplers=samplers))

    if scaled and notes is not None:
        worst = max(scaled.values())
        names = ", ".join(sorted(scaled)[:4])
        notes.append(
            f"{len(scaled)} joint(s) are scaled by the animation, up to "
            f"{1.0 + worst:.2f}x ({names}). The engine has no per-joint scale, so "
            "the scale is dropped and only the rotation and position are kept."
        )
    return animations


def load(path: Path) -> Gltf:
    path = Path(path).resolve()
    doc = _Document(path)

    up = doc.root.find("c:asset/c:up_axis", NS)
    up_axis = (up.text or "Y_UP").strip() if up is not None else "Y_UP"
    if up_axis == "Z_UP":
        conversion = Z_UP_TO_Y_UP
    elif up_axis == "X_UP":
        conversion = X_UP_TO_Y_UP
    else:
        conversion = IDENTITY4

    images, image_index = _read_images(doc, path.parent)
    materials, material_index = _read_materials(doc, image_index)

    scene_ref = doc.root.find("c:scene/c:instance_visual_scene", NS)
    scene = doc.find(scene_ref.get("url") if scene_ref is not None else None)
    if scene is None:
        scene = doc.root.find("c:library_visual_scenes/c:visual_scene", NS)
    if scene is None:
        raise ColladaError(f"{path.name}: no visual scene")

    nodes: list[Node] = []
    meshes: list[list[Primitive]] = []
    skins: list[Skin] = []
    roots: list[int] = []
    joint_node_index: dict[str, int] = {}
    node_depth: dict[int, int] = {}
    pending: list[tuple[int, ET.Element]] = []

    def add_node(element, depth: int) -> int:
        index = len(nodes)
        node_depth[index] = depth
        matrix = _node_matrix(element)
        label = element.get("id") or element.get("name") or f"node_{index}"
        if depth == 0:
            matrix = _matmul(conversion, matrix)
        translation, rotation, scale = _decompose(matrix, label)
        nodes.append(Node(index=index, name=label, children=[], mesh=None, skin=None,
                          translation=translation, rotation=rotation, scale=scale))
        for key in (element.get("sid"), element.get("id"), element.get("name")):
            if key and key not in joint_node_index:
                joint_node_index[key] = index
        pending.append((index, element))
        for child in element.findall("c:node", NS):
            nodes[index].children.append(add_node(child, depth + 1))
        return index

    for element in scene.findall("c:node", NS):
        roots.append(add_node(element, 0))

    def bindings_of(instance) -> dict[str, str]:
        out: dict[str, str] = {}
        for entry in instance.findall("c:bind_material/c:technique_common/c:instance_material", NS):
            out[entry.get("symbol", "")] = (entry.get("target", "") or "").lstrip("#")
        return out

    for index, element in pending:
        instance = element.find("c:instance_geometry", NS)
        controller = element.find("c:instance_controller", NS)
        geometry = None
        influences = None
        if controller is not None:
            result = _read_skin(doc, doc.find(controller.get("url")), joint_node_index)
            if result is not None:
                skin, geometry, influences = result
                nodes[index].skin = len(skins)
                skins.append(skin)
            instance = controller
        elif instance is not None:
            geometry = doc.find(instance.get("url"))
        if geometry is None:
            continue
        primitives = _read_geometry(doc, geometry, material_index, bindings_of(instance),
                                    influences)
        if not primitives:
            continue
        nodes[index].mesh = len(meshes)
        meshes.append(primitives)

    for skin in skins:
        if any(joint < 0 for joint in skin.joints):
            raise ColladaError(
                f"{path.name}: a skin references a joint that is not in the scene"
            )

    notes: list[str] = []
    animations = _read_animations(doc, joint_node_index, node_depth, conversion, notes)

    return Gltf(path=path, json={"notes": notes}, nodes=nodes, skins=skins, meshes=meshes,
                materials=materials, images=images, animations=animations,
                roots=roots)
