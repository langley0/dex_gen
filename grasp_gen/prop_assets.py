from __future__ import annotations

import json
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
DECOR01_XML = ASSETS / "decor_01" / "decor_01.xml"
DECOR01_MESH = ASSETS / "decor_01" / "meshes" / "decor_01_collision.stl"
GENERATED_ASSETS = ASSETS / "generated"
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


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                vertices.append([float(token) for token in line.split()[1:4]])
                continue
            if not line.startswith("f "):
                continue
            indices = [int(token.split("/", 1)[0]) - 1 for token in line.split()[1:]]
            if len(indices) == 3:
                faces.append(indices)
            else:
                for start in range(1, len(indices) - 1):
                    faces.append([indices[0], indices[start], indices[start + 1]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".obj":
        return _load_obj(path)
    if suffix == ".stl":
        return _load_binary_stl(path)
    raise ValueError(f"Unsupported mesh format: {path}")


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


def _decor01_com_local() -> np.ndarray:
    tree = ET.parse(DECOR01_XML)
    inertial = tree.find("./worldbody/body/inertial")
    if inertial is None or "pos" not in inertial.attrib:
        return np.zeros(3, dtype=np.float32)
    return _parse_vec3(inertial.attrib["pos"])


def make_cylinder_prop(
    radius: float = CYLINDER_RADIUS,
    half_height: float = CYLINDER_HALF_HEIGHT,
) -> tuple[Prop, dict[str, object]]:
    radius = float(radius)
    half_height = float(half_height)
    vertices, faces = cylinder_mesh(radius, half_height, sides=CYLINDER_SIDES)
    return (
        Prop(vertices, faces, pos=np.zeros(3, dtype=float), name="cylinder"),
        {
            "kind": "cylinder",
            "radius": radius,
            "half_height": half_height,
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


def make_box_prop(size_xyz: tuple[float, float, float]) -> tuple[Prop, dict[str, object]]:
    size = np.asarray(size_xyz, dtype=float).reshape(3)
    if np.any(size <= 0.0):
        raise ValueError("Box size entries must be positive.")
    half = 0.5 * size
    vertices, faces = box_mesh(tuple(float(value) for value in half))
    return (
        Prop(vertices, faces, pos=np.zeros(3, dtype=float), name="box"),
        {
            "kind": "box",
            "size_xyz": size.astype(float).tolist(),
            "pos": [0.0, 0.0, 0.0],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "com_local": [0.0, 0.0, 0.0],
            "name": "box",
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


def make_decor01_prop() -> tuple[Prop, dict[str, object]]:
    vertices, faces = _load_binary_stl(DECOR01_MESH)
    com_local = _decor01_com_local()
    return (
        Prop(
            vertices,
            faces,
            pos=np.zeros(3, dtype=float),
            com_local=com_local,
            name="decor01",
            rgba=np.array([0.30, 0.78, 0.92, 1.0], dtype=float),
        ),
        {
            "kind": "decor01",
            "mesh_path": str(DECOR01_MESH.relative_to(ROOT)),
            "xml_path": str(DECOR01_XML.relative_to(ROOT)),
            "pos": [0.0, 0.0, 0.0],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "com_local": com_local.astype(float).tolist(),
            "name": "decor01",
        },
    )


def make_generated_asset_prop(asset_name: str) -> tuple[Prop, dict[str, object]]:
    asset_dir = GENERATED_ASSETS / asset_name
    metadata_path = asset_dir / "asset.json"
    if not metadata_path.exists():
        raise ValueError(f"Generated asset {asset_name!r} was not found at {metadata_path}.")

    raw_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    mesh_path_value = raw_meta.get("mesh_path")
    if not isinstance(mesh_path_value, str) or not mesh_path_value:
        raise ValueError(f"Generated asset {asset_name!r} must define a non-empty mesh_path.")
    mesh_path = (ROOT / mesh_path_value).resolve()
    vertices, faces = _load_mesh(mesh_path)

    rgba_raw = raw_meta.get("rgba", [0.91, 0.58, 0.19, 1.0])
    friction_raw = raw_meta.get("friction", [1.1, 0.05, 0.01])
    prop = Prop(
        vertices,
        faces,
        pos=np.asarray(raw_meta.get("pos", [0.0, 0.0, 0.0]), dtype=float),
        quat=np.asarray(raw_meta.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=float),
        com_local=np.asarray(raw_meta.get("com_local", [0.0, 0.0, 0.0]), dtype=float),
        name=str(raw_meta.get("name", asset_name)),
        rgba=np.asarray(rgba_raw, dtype=float),
        friction=np.asarray(friction_raw, dtype=float),
        condim=int(raw_meta.get("condim", 4)),
    )
    metadata = dict(raw_meta)
    metadata.setdefault("kind", "mesh_asset")
    metadata["asset_name"] = asset_name
    metadata["mesh_path"] = str(mesh_path.relative_to(ROOT))
    metadata.setdefault("name", asset_name)
    return prop, metadata


def make_named_prop(
    kind: str,
    *,
    cube_size: float,
    box_size: tuple[float, float, float] | None = None,
    cylinder_radius: float = CYLINDER_RADIUS,
    cylinder_half_height: float = CYLINDER_HALF_HEIGHT,
) -> tuple[Prop, dict[str, object]]:
    kind = str(kind).lower()
    if kind == "cube":
        return make_cube_prop(cube_size)
    if kind == "box":
        if box_size is None:
            raise ValueError("box_size is required when kind='box'.")
        return make_box_prop(box_size)
    if kind == "drill":
        return make_drill_prop()
    if kind == "decor01":
        return make_decor01_prop()
    return make_cylinder_prop(radius=cylinder_radius, half_height=cylinder_half_height)


def prop_from_metadata(meta: dict[str, object]) -> Prop:
    kind = str(meta.get("kind", ""))
    if kind == "cube":
        prop, _ = make_cube_prop(float(meta["size"]))
    elif kind == "box":
        size_values = meta.get("size_xyz")
        if not isinstance(size_values, (list, tuple)) or len(size_values) != 3:
            raise ValueError("Box metadata must include size_xyz with three entries.")
        prop, _ = make_box_prop(tuple(float(value) for value in size_values))
    elif kind == "drill":
        prop, _ = make_drill_prop()
    elif kind == "decor01":
        prop, _ = make_decor01_prop()
    elif kind == "cylinder":
        prop, _ = make_cylinder_prop(
            radius=float(meta.get("radius", CYLINDER_RADIUS)),
            half_height=float(meta.get("half_height", CYLINDER_HALF_HEIGHT)),
        )
    elif "mesh_path" in meta:
        mesh_path = (ROOT / str(meta["mesh_path"])).resolve()
        vertices, faces = _load_mesh(mesh_path)
        prop = Prop(
            vertices,
            faces,
            pos=np.asarray(meta.get("pos", [0.0, 0.0, 0.0]), dtype=float),
            quat=np.asarray(meta.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=float),
            com_local=np.asarray(meta.get("com_local", [0.0, 0.0, 0.0]), dtype=float),
            name=str(meta.get("name", mesh_path.stem)),
            rgba=np.asarray(meta.get("rgba", [0.91, 0.58, 0.19, 1.0]), dtype=float),
            friction=np.asarray(meta.get("friction", [1.1, 0.05, 0.01]), dtype=float),
            condim=int(meta.get("condim", 4)),
        )
    else:
        raise ValueError(f"Unsupported prop kind: {kind!r}")

    prop.pos = np.asarray(meta.get("pos", [0.0, 0.0, 0.0]), dtype=float).reshape(3).copy()
    prop.quat = np.asarray(meta.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=float).reshape(4).copy()
    prop.com_local = np.asarray(meta.get("com_local", [0.0, 0.0, 0.0]), dtype=float).reshape(3).copy()
    prop.name = str(meta.get("name", prop.name))
    return prop
