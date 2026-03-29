from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .mesh_primitives import box_mesh, cylinder_mesh
from .prop import Prop


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
DRILL_XML = ASSETS / "drill" / "material.xml"
DRILL_MESH = ASSETS / "drill" / "meshes" / "material_collision.stl"
CYLINDER_RADIUS = 0.045
CYLINDER_HALF_HEIGHT = 0.165
CYLINDER_SIDES = 32


def _load_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as stream:
        stream.read(80)
        triangle_count = struct.unpack("<I", stream.read(4))[0]
        vertices = np.zeros((3 * triangle_count, 3), dtype=np.float32)
        faces = np.zeros((triangle_count, 3), dtype=np.int32)
        for triangle_index in range(triangle_count):
            stream.read(12)
            triangle = np.frombuffer(stream.read(36), dtype="<f4").reshape(3, 3).copy()
            stream.read(2)
            base = 3 * triangle_index
            vertices[base : base + 3] = triangle
            faces[triangle_index] = np.array([base, base + 1, base + 2], dtype=np.int32)
    return vertices, faces


def _parse_vec3(text: str) -> np.ndarray:
    values = [float(value) for value in str(text).split()]
    if len(values) != 3:
        raise ValueError(f"Expected 3 values, got {text!r}")
    return np.asarray(values, dtype=np.float32)


def _drill_com_local() -> np.ndarray:
    tree = ET.parse(DRILL_XML)
    inertial = tree.find("./worldbody/body/inertial")
    if inertial is None or "pos" not in inertial.attrib:
        return np.zeros(3, dtype=np.float32)
    return _parse_vec3(inertial.attrib["pos"])


def make_cylinder_prop() -> tuple[Prop, dict[str, object]]:
    vertices, faces = cylinder_mesh(CYLINDER_RADIUS, CYLINDER_HALF_HEIGHT, sides=CYLINDER_SIDES)
    return (
        Prop(vertices, faces, pos=np.zeros(3, dtype=float), name="cylinder"),
        {
            "kind": "cylinder",
            "radius": CYLINDER_RADIUS,
            "half_height": CYLINDER_HALF_HEIGHT,
            "sides": CYLINDER_SIDES,
            "pos": [0.0, 0.0, 0.0],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "com_local": [0.0, 0.0, 0.0],
            "name": "cylinder",
        },
    )


def make_cube_prop(size: float) -> tuple[Prop, dict[str, object]]:
    half = 0.5 * float(size)
    vertices, faces = box_mesh((half, half, half))
    return (
        Prop(vertices, faces, pos=np.zeros(3, dtype=float), name="cube"),
        {
            "kind": "cube",
            "size": float(size),
            "pos": [0.0, 0.0, 0.0],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "com_local": [0.0, 0.0, 0.0],
            "name": "cube",
        },
    )


def make_drill_prop() -> tuple[Prop, dict[str, object]]:
    vertices, faces = _load_binary_stl(DRILL_MESH)
    com_local = _drill_com_local()
    return (
        Prop(
            vertices,
            faces,
            pos=np.zeros(3, dtype=float),
            com_local=com_local,
            name="drill",
            rgba=np.array([0.34, 0.36, 0.39, 1.0], dtype=float),
        ),
        {
            "kind": "drill",
            "mesh_path": str(DRILL_MESH.relative_to(ROOT)),
            "xml_path": str(DRILL_XML.relative_to(ROOT)),
            "pos": [0.0, 0.0, 0.0],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "com_local": com_local.astype(float).tolist(),
            "name": "drill",
        },
    )


def make_named_prop(kind: str, *, cube_size: float) -> tuple[Prop, dict[str, object]]:
    kind = str(kind).lower()
    if kind == "cube":
        return make_cube_prop(cube_size)
    if kind == "drill":
        return make_drill_prop()
    return make_cylinder_prop()


def prop_from_metadata(meta: dict[str, object]) -> Prop:
    kind = str(meta.get("kind", ""))
    if kind == "cube":
        prop, _ = make_cube_prop(float(meta["size"]))
    elif kind == "drill":
        prop, _ = make_drill_prop()
    elif kind == "cylinder":
        prop, _ = make_cylinder_prop()
    else:
        raise ValueError(f"Unsupported prop kind: {kind!r}")

    prop.pos = np.asarray(meta.get("pos", [0.0, 0.0, 0.0]), dtype=float).reshape(3).copy()
    prop.quat = np.asarray(meta.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=float).reshape(4).copy()
    prop.com_local = np.asarray(meta.get("com_local", [0.0, 0.0, 0.0]), dtype=float).reshape(3).copy()
    prop.name = str(meta.get("name", prop.name))
    return prop
