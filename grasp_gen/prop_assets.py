from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .mesh_primitives import box_mesh, cylinder_mesh
from .prop_cloud import SurfaceCloudConfig, build_surface_point_cloud
from .prop import Prop


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
DRILL_XML = ASSETS / "drill" / "material.xml"
DRILL_MESH = ASSETS / "drill" / "meshes" / "material_collision.stl"
DECOR01_XML = ASSETS / "decor_01" / "decor_01.xml"
DECOR01_MESH = ASSETS / "decor_01" / "meshes" / "decor_01_collision.stl"
CYLINDER_RADIUS = 0.045
CYLINDER_HALF_HEIGHT = 0.165
CYLINDER_SIDES = 32
DEFAULT_SURFACE_CLOUD = SurfaceCloudConfig()


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


def _decor01_com_local() -> np.ndarray:
    tree = ET.parse(DECOR01_XML)
    inertial = tree.find("./worldbody/body/inertial")
    if inertial is None or "pos" not in inertial.attrib:
        return np.zeros(3, dtype=np.float32)
    return _parse_vec3(inertial.attrib["pos"])


def _cloud_meta(config: SurfaceCloudConfig, point_count: int) -> dict[str, object]:
    return {
        "spacing": float(config.spacing),
        "seed": int(config.seed),
        "oversample": int(config.oversample),
        "min_points": int(config.min_points),
        "max_points": None if config.max_points is None else int(config.max_points),
        "point_count": int(point_count),
    }


def _cloud_config_from_meta(meta: dict[str, object]) -> SurfaceCloudConfig:
    raw = meta.get("surface_cloud")
    if not isinstance(raw, dict):
        return DEFAULT_SURFACE_CLOUD
    return SurfaceCloudConfig(
        spacing=float(raw.get("spacing", DEFAULT_SURFACE_CLOUD.spacing)),
        seed=int(raw.get("seed", DEFAULT_SURFACE_CLOUD.seed)),
        oversample=int(raw.get("oversample", DEFAULT_SURFACE_CLOUD.oversample)),
        min_points=int(raw.get("min_points", DEFAULT_SURFACE_CLOUD.min_points)),
        max_points=(
            None
            if raw.get("max_points", DEFAULT_SURFACE_CLOUD.max_points) is None
            else int(raw.get("max_points", DEFAULT_SURFACE_CLOUD.max_points))
        ),
    )


def _build_prop(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    pos: np.ndarray,
    name: str,
    com_local: np.ndarray | None = None,
    quat: np.ndarray | None = None,
    rgba: np.ndarray | None = None,
    cloud_config: SurfaceCloudConfig | None = None,
) -> Prop:
    cloud_cfg = DEFAULT_SURFACE_CLOUD if cloud_config is None else cloud_config
    surface_cloud = build_surface_point_cloud(vertices, faces, config=cloud_cfg)
    return Prop(
        vertices,
        faces,
        pos=pos,
        com_local=com_local,
        quat=quat,
        name=name,
        rgba=rgba,
        surface_cloud=surface_cloud,
    )


def make_cylinder_prop(*, cloud_config: SurfaceCloudConfig | None = None) -> tuple[Prop, dict[str, object]]:
    vertices, faces = cylinder_mesh(CYLINDER_RADIUS, CYLINDER_HALF_HEIGHT, sides=CYLINDER_SIDES)
    cloud_cfg = DEFAULT_SURFACE_CLOUD if cloud_config is None else cloud_config
    prop = _build_prop(vertices, faces, pos=np.zeros(3, dtype=float), name="cylinder", cloud_config=cloud_cfg)
    return (
        prop,
        {
            "kind": "cylinder",
            "radius": CYLINDER_RADIUS,
            "half_height": CYLINDER_HALF_HEIGHT,
            "sides": CYLINDER_SIDES,
            "pos": [0.0, 0.0, 0.0],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "com_local": [0.0, 0.0, 0.0],
            "name": "cylinder",
            "surface_cloud": _cloud_meta(cloud_cfg, len(prop.surface_cloud.points_local)),
        },
    )


def make_cube_prop(size: float, *, cloud_config: SurfaceCloudConfig | None = None) -> tuple[Prop, dict[str, object]]:
    half = 0.5 * float(size)
    vertices, faces = box_mesh((half, half, half))
    cloud_cfg = DEFAULT_SURFACE_CLOUD if cloud_config is None else cloud_config
    prop = _build_prop(vertices, faces, pos=np.zeros(3, dtype=float), name="cube", cloud_config=cloud_cfg)
    return (
        prop,
        {
            "kind": "cube",
            "size": float(size),
            "pos": [0.0, 0.0, 0.0],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "com_local": [0.0, 0.0, 0.0],
            "name": "cube",
            "surface_cloud": _cloud_meta(cloud_cfg, len(prop.surface_cloud.points_local)),
        },
    )


def make_drill_prop(*, cloud_config: SurfaceCloudConfig | None = None) -> tuple[Prop, dict[str, object]]:
    vertices, faces = _load_binary_stl(DRILL_MESH)
    com_local = _drill_com_local()
    cloud_cfg = DEFAULT_SURFACE_CLOUD if cloud_config is None else cloud_config
    prop = _build_prop(
        vertices,
        faces,
        pos=np.zeros(3, dtype=float),
        com_local=com_local,
        name="drill",
        rgba=np.array([0.34, 0.36, 0.39, 1.0], dtype=float),
        cloud_config=cloud_cfg,
    )
    return (
        prop,
        {
            "kind": "drill",
            "mesh_path": str(DRILL_MESH.relative_to(ROOT)),
            "xml_path": str(DRILL_XML.relative_to(ROOT)),
            "pos": [0.0, 0.0, 0.0],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "com_local": com_local.astype(float).tolist(),
            "name": "drill",
            "surface_cloud": _cloud_meta(cloud_cfg, len(prop.surface_cloud.points_local)),
        },
    )


def make_decor01_prop(*, cloud_config: SurfaceCloudConfig | None = None) -> tuple[Prop, dict[str, object]]:
    vertices, faces = _load_binary_stl(DECOR01_MESH)
    com_local = _decor01_com_local()
    cloud_cfg = DEFAULT_SURFACE_CLOUD if cloud_config is None else cloud_config
    prop = _build_prop(
        vertices,
        faces,
        pos=np.zeros(3, dtype=float),
        com_local=com_local,
        name="decor01",
        rgba=np.array([0.80, 0.86, 0.93, 1.0], dtype=float),
        cloud_config=cloud_cfg,
    )
    return (
        prop,
        {
            "kind": "decor01",
            "mesh_path": str(DECOR01_MESH.relative_to(ROOT)),
            "xml_path": str(DECOR01_XML.relative_to(ROOT)),
            "pos": [0.0, 0.0, 0.0],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "com_local": com_local.astype(float).tolist(),
            "name": "decor01",
            "surface_cloud": _cloud_meta(cloud_cfg, len(prop.surface_cloud.points_local)),
        },
    )


def make_named_prop(kind: str, *, cube_size: float) -> tuple[Prop, dict[str, object]]:
    kind = str(kind).lower()
    if kind == "cube":
        return make_cube_prop(cube_size)
    if kind == "drill":
        return make_drill_prop()
    if kind == "decor01":
        return make_decor01_prop()
    return make_cylinder_prop()


def prop_from_metadata(meta: dict[str, object]) -> Prop:
    kind = str(meta.get("kind", ""))
    cloud_config = _cloud_config_from_meta(meta)
    if kind == "cube":
        prop, _ = make_cube_prop(float(meta["size"]), cloud_config=cloud_config)
    elif kind == "drill":
        prop, _ = make_drill_prop(cloud_config=cloud_config)
    elif kind == "decor01":
        prop, _ = make_decor01_prop(cloud_config=cloud_config)
    elif kind == "cylinder":
        prop, _ = make_cylinder_prop(cloud_config=cloud_config)
    else:
        raise ValueError(f"Unsupported prop kind: {kind!r}")

    prop.pos = np.asarray(meta.get("pos", [0.0, 0.0, 0.0]), dtype=float).reshape(3).copy()
    prop.quat = np.asarray(meta.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=float).reshape(4).copy()
    prop.com_local = np.asarray(meta.get("com_local", [0.0, 0.0, 0.0]), dtype=float).reshape(3).copy()
    prop.name = str(meta.get("name", prop.name))
    return prop
