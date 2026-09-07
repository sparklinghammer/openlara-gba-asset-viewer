#!/usr/bin/env python3
"""Autodesk FBX reader (ASCII) that hands back the same structures as the glTF reader.

The fourth front-end, alongside glTF, COLLADA and OBJ, and the one most animated
models arrive in. Only the ASCII flavour is read: the binary one is a different
container holding the same tree, and nothing in this project has needed it yet.

FBX differs from the other three in ways that all have to be dealt with here:

  * The file is a flat pool of objects plus a `Connections` list. Nothing is
    nested -- a bone's parent, a mesh's material, a curve's target are all
    edges in that graph, so the graph is rebuilt first and read afterwards.
  * Transforms are properties (`Lcl Translation`, `Lcl Rotation`, `Lcl Scaling`)
    with rotation given as Euler degrees in a declared order, not as a matrix.
  * Animation is keyed one component at a time: a separate curve for X, Y and Z
    of each of translation, rotation and scale. They are recombined here into
    the quaternion track the pipeline uses, sampled at the union of their key
    times so no curve's detail is lost to another's sparser one.
  * Times are in FBX ticks, of which there are 46 186 158 000 per second.
  * Matrices are stored transposed relative to the convention used here.
"""

from __future__ import annotations

import math
from pathlib import Path

from gltf_reader import (
    Animation, Channel, Gltf, GltfError, Image, Material, Node, Primitive,
    Sampler, Skin,
)

FBX_TICKS_PER_SECOND = 46186158000.0

IMAGE_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".bmp": "image/bmp", ".tga": "image/x-tga",
}

# FbxEuler::EOrder. The value names the order the axes are applied in; the
# composite matrix multiplies them in the reverse of that order.
EULER_ORDER = {
    0: "XYZ", 1: "XZY", 2: "YZX", 3: "YXZ", 4: "ZXY", 5: "ZYX", 6: "XYZ",
}


class FbxError(GltfError):
    pass


# --------------------------------------------------------------------------
# the file tree
# --------------------------------------------------------------------------

class FbxNode:
    __slots__ = ("name", "values", "children")

    def __init__(self, name: str, values: list):
        self.name = name
        self.values = values
        self.children: list[FbxNode] = []

    def find(self, name: str):
        for child in self.children:
            if child.name == name:
                return child
        return None

    def find_all(self, name: str):
        return [child for child in self.children if child.name == name]

    def value(self, name: str, index: int = 0, default=None):
        child = self.find(name)
        if child is None or index >= len(child.values):
            return default
        return child.values[index]

    def __repr__(self) -> str:                      # debugging aid
        return f"<{self.name} {self.values[:3]} ({len(self.children)})>"


def _split_values(text: str) -> list:
    """Split a property list on commas, keeping quoted strings whole.

    A quoted value keeps its own spaces but not the ones around the quotes, so
    `, "LimbNode"` yields `LimbNode` and a path with spaces survives intact.
    """
    out: list = []
    token = ""
    quoted = False
    was_quoted = False
    for character in text:
        if character == '"':
            if not quoted:
                token = ""              # drop whitespace before the quote
                was_quoted = True
            quoted = not quoted
            continue
        if character == "," and not quoted:
            out.append(_coerce(token, was_quoted))
            token = ""
            was_quoted = False
            continue
        if was_quoted and not quoted:
            continue                    # ignore whatever trails the closing quote
        token += character
    if token.strip() or was_quoted:
        out.append(_coerce(token, was_quoted))
    return out


def _coerce(token: str, quoted: bool):
    if quoted:
        return token
    token = token.strip()
    if not token:
        return ""
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def parse(path: Path) -> FbxNode:
    """Read the whole file into a tree of FbxNode.

    Written line by line rather than with a character tokenizer because the
    format is line oriented in practice and the files are large: this one is
    36 MB, and the array payloads are most of it.
    """
    root = FbxNode("", [])
    stack = [root]
    array: list[str] | None = None
    array_node: FbxNode | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if array is not None:
                # Inside a `*N { a: ... }` payload, which wraps across lines.
                if line == "}" or line.endswith("}"):
                    array.append(line.rstrip("}").strip())
                    text = " ".join(array)
                    if text.startswith("a:"):
                        text = text[2:]
                    array_node.values = [
                        _coerce(v, False) for v in text.split(",") if v.strip()
                    ]
                    array = None
                    array_node = None
                    stack.pop()
                    continue
                array.append(line[2:].strip() if line.startswith("a:") else line)
                continue

            if not line or line.startswith(";"):
                continue
            if line == "}":
                if len(stack) > 1:
                    stack.pop()
                continue

            opens = line.endswith("{")
            body = line[:-1].strip() if opens else line
            name, _, rest = body.partition(":")
            name = name.strip()
            values = _split_values(rest.strip()) if rest.strip() else []

            node = FbxNode(name, values)
            stack[-1].children.append(node)
            if opens:
                stack.append(node)
                # `Vertices: *2181 {` announces a numeric payload.
                if values and isinstance(values[0], str) and values[0].startswith("*"):
                    array = []
                    array_node = node
                elif values and isinstance(values[0], int) and rest.strip().startswith("*"):
                    array = []
                    array_node = node
    return root


# --------------------------------------------------------------------------
# maths
# --------------------------------------------------------------------------

def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)]


def _transpose(values: list) -> list[list[float]]:
    """A 16-float FBX matrix, in the row-vector order it stores, as row-major."""
    return [[float(values[column * 4 + row]) for column in range(4)]
            for row in range(4)]


def _axis_matrix(axis: int, degrees: float):
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    if axis == 0:
        return [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]]
    if axis == 1:
        return [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]]
    return [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def euler_quaternion(degrees, order: str = "XYZ"):
    """Euler angles in degrees to a quaternion, honouring the axis order."""
    matrix = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    # The order names the sequence the axes are APPLIED in, so eEulerXYZ turns
    # about X first. Acting on column vectors that is Rz * Ry * Rx -- the listed
    # axes multiply right to left. Composing them the other way round is the
    # classic FBX import bug; here it put the shoulders 180 degrees out, which
    # is what comparing against the same model in COLLADA caught.
    for letter in reversed(order):
        axis = "XYZ".index(letter)
        matrix = _matmul(matrix, _axis_matrix(axis, degrees[axis]))
    return matrix_quaternion(matrix)


def matrix_quaternion(m):
    trace = m[0][0] + m[1][1] + m[2][2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w, x = 0.25 * s, (m[2][1] - m[1][2]) / s
        y, z = (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        w, x = (m[2][1] - m[1][2]) / s, 0.25 * s
        y, z = (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        w, x = (m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s
        y, z = 0.25 * s, (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        w, x = (m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s
        y, z = (m[1][2] + m[2][1]) / s, 0.25 * s
    length = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    return (x / length, y / length, z / length, w / length)


# --------------------------------------------------------------------------
# the object graph
# --------------------------------------------------------------------------

class _Scene:
    """The object pool and the connection graph that gives it shape."""

    def __init__(self, root: FbxNode):
        self.objects: dict[int, FbxNode] = {}
        self.kind: dict[int, tuple[str, str, str]] = {}
        objects = root.find("Objects")
        if objects is None:
            raise FbxError("no Objects section; is this an ASCII FBX file?")
        for node in objects.children:
            if not node.values or not isinstance(node.values[0], int):
                continue
            uid = node.values[0]
            name = node.values[1] if len(node.values) > 1 else ""
            sub = node.values[2] if len(node.values) > 2 else ""
            self.objects[uid] = node
            self.kind[uid] = (node.name, str(name), str(sub))

        self.children: dict[int, list[int]] = {}
        self.parents: dict[int, list[int]] = {}
        self.properties: dict[int, list[tuple[int, str]]] = {}
        connections = root.find("Connections")
        if connections is not None:
            for edge in connections.find_all("C"):
                if len(edge.values) < 3:
                    continue
                mode, child, parent = edge.values[0], edge.values[1], edge.values[2]
                self.children.setdefault(parent, []).append(child)
                self.parents.setdefault(child, []).append(parent)
                if mode == "OP" and len(edge.values) > 3:
                    self.properties.setdefault(parent, []).append(
                        (child, str(edge.values[3])))

    def label(self, uid: int) -> str:
        """The bare name of an object, past the `Model::` style prefix."""
        raw = self.kind.get(uid, ("", "", ""))[1]
        return raw.split("::", 1)[1] if "::" in raw else raw

    def children_of(self, uid: int, node_type: str = "", sub_type: str = ""):
        out = []
        for child in self.children.get(uid, ()):
            kind = self.kind.get(child)
            if kind is None:
                continue
            if node_type and kind[0] != node_type:
                continue
            if sub_type and kind[2] != sub_type:
                continue
            out.append(child)
        return out


def _properties(node: FbxNode) -> dict[str, list]:
    """A node's Properties70 block, keyed by property name."""
    out: dict[str, list] = {}
    block = node.find("Properties70")
    if block is None:
        return out
    for entry in block.find_all("P"):
        if entry.values:
            out[str(entry.values[0])] = entry.values[1:]
    return out


def _local_trs(node: FbxNode):
    """A model's translation, rotation quaternion and scale."""
    props = _properties(node)

    def vector(name, fallback):
        values = props.get(name)
        if not values or len(values) < 6:
            return fallback
        return tuple(float(v) for v in values[3:6])

    translation = vector("Lcl Translation", (0.0, 0.0, 0.0))
    rotation = vector("Lcl Rotation", (0.0, 0.0, 0.0))
    scale = vector("Lcl Scaling", (1.0, 1.0, 1.0))
    order_values = props.get("RotationOrder")
    order = EULER_ORDER.get(int(order_values[3]) if order_values and len(order_values) > 3
                            else 0, "XYZ")
    return translation, euler_quaternion(rotation, order), scale, order, rotation


# --------------------------------------------------------------------------
# reading the scene
# --------------------------------------------------------------------------

def _read_images(scene: _Scene, base: Path):
    """One Image per Video clip; Textures point at them."""
    images: list[Image] = []
    by_uid: dict[int, int] = {}
    for uid, (node_type, _, sub) in scene.kind.items():
        if node_type != "Video" or sub != "Clip":
            continue
        node = scene.objects[uid]
        name = node.value("RelativeFilename") or node.value("Filename") or ""
        name = str(name).replace("\\", "/").split("/")[-1]
        if not name:
            continue
        target = base / name
        if not target.is_file():
            raise FbxError(
                f"texture named by the FBX is missing: {name}. FBX keeps its images "
                "beside the file rather than inside it, so the whole folder has to "
                "travel with the model."
            )
        by_uid[uid] = len(images)
        images.append(Image(index=len(images), name=Path(name).stem,
                            data=target.read_bytes(),
                            mime_type=IMAGE_MIME.get(target.suffix.lower(), "")))
    return images, by_uid


def _read_materials(scene: _Scene, image_of: dict[int, int]):
    """One Material per Material object, with the image its texture reaches."""
    materials: list[Material] = []
    by_uid: dict[int, int] = {}
    for uid, (node_type, _, _) in scene.kind.items():
        if node_type != "Material":
            continue
        node = scene.objects[uid]
        props = _properties(node)
        colour = (1.0, 1.0, 1.0, 1.0)
        for key in ("DiffuseColor", "Diffuse"):
            values = props.get(key)
            if values and len(values) >= 6:
                colour = (float(values[3]), float(values[4]), float(values[5]), 1.0)
                break

        texture = None
        for child, _property in scene.properties.get(uid, ()):
            if scene.kind.get(child, ("",))[0] != "Texture":
                continue
            for video in scene.children_of(child, "Video"):
                if video in image_of:
                    texture = image_of[video]
                    break
            if texture is not None:
                break

        by_uid[uid] = len(materials)
        materials.append(Material(
            index=len(materials), name=scene.label(uid) or f"material_{len(materials)}",
            base_color=colour, base_color_texture=texture,
            alpha_mode="OPAQUE", alpha_cutoff=0.5,
            # FBX has no double-sided flag the converter can trust, so the
            # closedness test decides, as it does for COLLADA and OBJ.
            double_sided=False,
        ))
    return materials, by_uid


def _layer_values(layer, array_name: str, index_name: str):
    """One layer element's data, its optional index array and its mapping."""
    if layer is None:
        return None, None, "", ""
    data = layer.find(array_name)
    index = layer.find(index_name)
    return (data.values if data else None,
            index.values if index else None,
            str(layer.value("MappingInformationType") or ""),
            str(layer.value("ReferenceInformationType") or ""))


def _read_geometry(scene: _Scene, uid: int, material_slots: list):
    """The mesh, split into one primitive per material."""
    node = scene.objects[uid]
    raw = node.find("Vertices")
    faces = node.find("PolygonVertexIndex")
    if raw is None or faces is None:
        raise FbxError(f"geometry {scene.label(uid)!r} has no vertices")
    positions = [tuple(float(v) for v in raw.values[i * 3:i * 3 + 3])
                 for i in range(len(raw.values) // 3)]

    uvs, uv_index, uv_mapping, uv_reference = _layer_values(
        node.find("LayerElementUV"), "UV", "UVIndex")
    materials, _, material_mapping, _ = _layer_values(
        node.find("LayerElementMaterial"), "Materials", "Materials")
    # Normals are kept only to tell which side of a face is the front: an open
    # shell encloses no volume, so nothing else in the file answers that.
    normals, normal_index, normal_mapping, normal_reference = _layer_values(
        node.find("LayerElementNormal"), "Normals", "NormalsIndex")

    # PolygonVertexIndex marks the last corner of each polygon by storing its
    # complement; that is the only thing delimiting one polygon from the next.
    polygons = []
    current = []
    for position, value in enumerate(faces.values):
        index = int(value)
        last = index < 0
        if last:
            index = -index - 1
        current.append((index, position))
        if last:
            polygons.append(current)
            current = []
    if current:
        polygons.append(current)

    def uv_at(vertex: int, corner: int):
        if not uvs:
            return None
        slot = vertex if uv_mapping == "ByVertice" else corner
        if uv_reference == "IndexToDirect" and uv_index:
            if slot >= len(uv_index):
                return None
            slot = int(uv_index[slot])
        if slot * 2 + 1 >= len(uvs):
            return None
        # The V axis runs up from the bottom-left, as in COLLADA and OBJ.
        return (float(uvs[slot * 2]), 1.0 - float(uvs[slot * 2 + 1]))

    def normal_at(vertex: int, corner: int):
        if not normals:
            return None
        slot = vertex if normal_mapping in ("ByVertice", "ByVertex") else corner
        if normal_reference == "IndexToDirect" and normal_index:
            if slot >= len(normal_index):
                return None
            slot = int(normal_index[slot])
        if slot * 3 + 2 >= len(normals):
            return None
        return tuple(float(normals[slot * 3 + i]) for i in range(3))

    def material_at(polygon: int):
        if not materials:
            return None
        if material_mapping == "AllSame":
            slot = int(materials[0])
        elif polygon < len(materials):
            slot = int(materials[polygon])
        else:
            return None
        return material_slots[slot] if slot < len(material_slots) else None

    buckets = {}
    for number, polygon in enumerate(polygons):
        material = material_at(number)
        bucket = buckets.get(material)
        if bucket is None:
            bucket = {"positions": [], "uvs": [], "normals": [], "indices": [],
                      "slots": {}, "origin": [], "has_uv": False,
                      "has_normal": False}
            buckets[material] = bucket
        corners = []
        for vertex, corner in polygon:
            uv = uv_at(vertex, corner)
            key = (vertex, uv)
            slot = bucket["slots"].get(key)
            if slot is None:
                slot = len(bucket["positions"])
                bucket["slots"][key] = slot
                bucket["positions"].append(positions[vertex])
                bucket["uvs"].append(uv or (0.0, 0.0))
                bucket["origin"].append(vertex)
                # One normal per output slot, from the first corner that made
                # it; splitting slots by normal too would multiply them for
                # nothing, the target having no lighting.
                normal = normal_at(vertex, corner)
                bucket["normals"].append(normal or (0.0, 0.0, 0.0))
                if uv is not None:
                    bucket["has_uv"] = True
                if normal is not None:
                    bucket["has_normal"] = True
            corners.append(slot)
        for fan in range(1, len(corners) - 1):        # quads and ngons as fans
            bucket["indices"].extend([corners[0], corners[fan], corners[fan + 1]])

    primitives = []
    origins = []
    for material, bucket in buckets.items():
        if not bucket["indices"]:
            continue
        primitives.append(Primitive(
            positions=bucket["positions"],
            uvs=bucket["uvs"] if bucket["has_uv"] else None,
            indices=bucket["indices"],
            material=material,
            normals=bucket["normals"] if bucket["has_normal"] else None,
        ))
        # The vertex each output slot came from, needed to map skin weights,
        # which FBX lists against the original vertex numbering.
        origins.append(bucket["origin"])
    return primitives, origins


def _read_skin(scene: _Scene, geometry_uid: int, node_of: dict, primitives, origins):
    """The skin's joints and inverse binds, and per-vertex influences."""
    skins = scene.children_of(geometry_uid, "Deformer", "Skin")
    if not skins:
        return None
    clusters = scene.children_of(skins[0], "Deformer", "Cluster")
    if not clusters:
        return None

    joints = []
    inverse_bind = []
    influences = {}
    for cluster_uid in clusters:
        cluster = scene.objects[cluster_uid]
        bone = next((m for m in scene.children_of(cluster_uid, "Model")
                     if m in node_of), None)
        if bone is None:
            continue
        transform = cluster.find("Transform")
        # A cluster stores the inverse bind directly, transposed as every FBX
        # matrix is.
        matrix = (_transpose(transform.values)
                  if transform is not None and len(transform.values) == 16
                  else [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)])
        slot = len(joints)
        joints.append(node_of[bone])
        inverse_bind.append(matrix)

        indexes = cluster.find("Indexes")
        weights = cluster.find("Weights")
        if indexes is None or weights is None:
            continue
        for vertex, weight in zip(indexes.values, weights.values):
            if float(weight) > 0.0:
                influences.setdefault(int(vertex), []).append((slot, float(weight)))

    if not joints:
        return None

    for primitive, back in zip(primitives, origins):
        primitive.joints = []
        primitive.weights = []
        for vertex in back:
            entry = sorted(influences.get(vertex, []), key=lambda pair: -pair[1])[:4]
            total = sum(weight for _, weight in entry) or 1.0
            bones = [bone for bone, _ in entry] + [0] * (4 - len(entry))
            shares = [weight / total for _, weight in entry] + [0.0] * (4 - len(entry))
            primitive.joints.append(tuple(bones))
            primitive.weights.append(tuple(shares))
    return Skin(joints=joints, inverse_bind=inverse_bind)


def _sample(times: list, values: list, at: float, fallback: float) -> float:
    """One curve evaluated at a time, linearly, held flat past its ends."""
    if not times:
        return fallback
    if at <= times[0]:
        return values[0]
    if at >= times[-1]:
        return values[-1]
    high = 1
    while high < len(times) and times[high] < at:
        high += 1
    low = high - 1
    span = times[high] - times[low]
    if span <= 0:
        return values[low]
    return values[low] + (values[high] - values[low]) * (at - times[low]) / span


def _read_animations(scene: _Scene, node_of: dict, orders: dict,
                     notes: list) -> list[Animation]:
    """Recombine the per-component curves into rotation and translation tracks.

    FBX keys X, Y and Z of each channel as three separate curves, which need not
    even share key times. They are sampled onto the union of their times so no
    curve's detail is lost to another's sparser one, and the three Euler angles
    then become the quaternion the pipeline carries.
    """
    # Which object and property each curve or curve node feeds.
    target_of: dict = {}
    for parent, entries in scene.properties.items():
        for child, name in entries:
            target_of.setdefault(child, []).append((parent, name))

    scaled: dict = {}
    animations: list[Animation] = []
    stacks = [uid for uid, kind in scene.kind.items() if kind[0] == "AnimationStack"]
    for stack in stacks:
        samplers: list[Sampler] = []
        channels: list[Channel] = []
        tracks: dict = {}

        for layer in scene.children_of(stack, "AnimationLayer"):
            for curve_node in scene.children_of(layer, "AnimationCurveNode"):
                target = next(((uid, name) for uid, name in target_of.get(curve_node, ())
                               if uid in node_of), None)
                if target is None:
                    continue
                model, channel_name = target
                if channel_name not in ("Lcl Rotation", "Lcl Translation", "Lcl Scaling"):
                    continue
                holder = tracks.setdefault(model, {}).setdefault(channel_name, {})
                for curve in scene.children_of(curve_node, "AnimationCurve"):
                    component = next((name for uid, name in target_of.get(curve, ())
                                      if uid == curve_node), None)
                    if component not in ("d|X", "d|Y", "d|Z"):
                        continue
                    node = scene.objects[curve]
                    key_time = node.find("KeyTime")
                    key_value = node.find("KeyValueFloat")
                    if key_time is None or key_value is None:
                        continue
                    holder[component] = (
                        [float(t) / FBX_TICKS_PER_SECOND for t in key_time.values],
                        [float(v) for v in key_value.values],
                    )
                # Components without a curve keep the curve node's own default.
                defaults = _properties(scene.objects[curve_node])
                for component in ("d|X", "d|Y", "d|Z"):
                    if component not in holder:
                        value = defaults.get(component)
                        constant = float(value[3]) if value and len(value) > 3 else 0.0
                        holder[component] = ([], [constant])

        origin = None
        for channel_map in tracks.values():
            for components in channel_map.values():
                for times, _ in components.values():
                    if times:
                        origin = times[0] if origin is None else min(origin, times[0])
        origin = origin or 0.0

        for model, channel_map in tracks.items():
            for channel_name, components in channel_map.items():
                moments = sorted({t for times, _ in components.values() for t in times})
                if not moments:
                    continue
                fallback = 1.0 if channel_name == "Lcl Scaling" else 0.0
                sampled = []
                for at in moments:
                    triple = []
                    for component in ("d|X", "d|Y", "d|Z"):
                        times, values = components[component]
                        triple.append(_sample(times, values, at, fallback)
                                      if times else
                                      (values[0] if values else fallback))
                    sampled.append(triple)

                times = [t - origin for t in moments]
                if channel_name == "Lcl Scaling":
                    worst = max(max(abs(v - 1.0) for v in triple) for triple in sampled)
                    if worst > 0.01:
                        label = scene.label(model)
                        scaled[label] = max(scaled.get(label, 0.0), worst)
                    continue                # the engine has no per-joint scale
                if channel_name == "Lcl Rotation":
                    order = orders.get(node_of[model], "XYZ")
                    run = []
                    for triple in sampled:
                        quaternion = euler_quaternion(triple, order)
                        # Keep the run continuous, or a slerp between two nearly
                        # equal keys takes the long way round.
                        if run and sum(a * b for a, b in zip(quaternion, run[-1])) < 0.0:
                            quaternion = tuple(-v for v in quaternion)
                        run.append(quaternion)
                    samplers.append(Sampler(times=times, values=run,
                                            interpolation="LINEAR"))
                    channels.append(Channel(node=node_of[model], path="rotation",
                                            sampler=len(samplers) - 1))
                else:
                    samplers.append(Sampler(times=times,
                                            values=[tuple(t) for t in sampled],
                                            interpolation="LINEAR"))
                    channels.append(Channel(node=node_of[model], path="translation",
                                            sampler=len(samplers) - 1))

        if channels:
            animations.append(Animation(name=scene.label(stack), channels=channels,
                                        samplers=samplers))

    if scaled:
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
    scene = _Scene(parse(path))

    images, image_of = _read_images(scene, path.parent)
    materials, material_of = _read_materials(scene, image_of)

    # Models become nodes, in the order the file lists them so the result is
    # reproducible; the hierarchy comes from the connection graph.
    model_uids = [uid for uid, kind in scene.kind.items() if kind[0] == "Model"]
    node_of = {uid: index for index, uid in enumerate(model_uids)}
    nodes: list[Node] = []
    orders: dict = {}
    for index, uid in enumerate(model_uids):
        translation, rotation, scale, order, _ = _local_trs(scene.objects[uid])
        orders[index] = order
        nodes.append(Node(index=index, name=scene.label(uid) or f"node_{index}",
                          children=[], mesh=None, skin=None,
                          translation=translation, rotation=rotation, scale=scale))

    roots: list[int] = []
    for uid in model_uids:
        parents = [p for p in scene.parents.get(uid, ()) if p in node_of]
        if parents:
            nodes[node_of[parents[0]]].children.append(node_of[uid])
        else:
            roots.append(node_of[uid])
    if not roots:
        raise FbxError(f"{path.name}: no model is attached to the scene root")

    meshes: list[list[Primitive]] = []
    skins: list[Skin] = []
    for uid in model_uids:
        geometries = scene.children_of(uid, "Geometry", "Mesh")
        if not geometries:
            continue
        # A geometry names its materials by their position on the owning model.
        slots = [material_of.get(m) for m in scene.children_of(uid, "Material")]
        primitives, origins = _read_geometry(scene, geometries[0], slots)
        if not primitives:
            continue
        skin = _read_skin(scene, geometries[0], node_of, primitives, origins)
        nodes[node_of[uid]].mesh = len(meshes)
        meshes.append(primitives)
        if skin is not None:
            nodes[node_of[uid]].skin = len(skins)
            skins.append(skin)

    if not meshes:
        raise FbxError(f"{path.name}: no model carries a mesh")

    notes: list = []
    animations = _read_animations(scene, node_of, orders, notes)

    # FBX declares an up axis in GlobalSettings, but exporters disagree about
    # whether they baked it, exactly as in COLLADA. It is left alone and
    # `rotate` in pack.json remains the recourse.
    return Gltf(path=path, json={"notes": notes}, nodes=nodes, skins=skins,
                meshes=meshes, materials=materials, images=images,
                animations=animations, roots=roots)
