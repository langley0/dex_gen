from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
DRILL_MESH = ASSETS / "drill" / "meshes" / "material_collision.stl"
DECOR01_MESH = ASSETS / "decor_01" / "meshes" / "decor_01_collision.stl"


@dataclass(frozen=True)
class MeshData:
    vertices: np.ndarray
    faces: np.ndarray
    name: str


def _load_binary_stl(path: Path) -> MeshData:
    with path.open("rb") as stream:
        stream.read(80)
        triangle_count = struct.unpack("<I", stream.read(4))[0]
        vertices = np.zeros((triangle_count * 3, 3), dtype=np.float32)
        faces = np.zeros((triangle_count, 3), dtype=np.int32)
        for triangle_index in range(triangle_count):
            stream.read(12)
            triangle = np.frombuffer(stream.read(36), dtype="<f4").reshape(3, 3).copy()
            stream.read(2)
            base = triangle_index * 3
            vertices[base : base + 3] = triangle
            faces[triangle_index] = np.asarray([base, base + 1, base + 2], dtype=np.int32)
    return MeshData(vertices=vertices, faces=faces, name=path.stem)


def _load_obj(path: Path) -> MeshData:
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
    return MeshData(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        name=path.stem,
    )


def _box_mesh(size: float) -> MeshData:
    half = 0.5 * float(size)
    vertices = np.asarray(
        [
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.int32,
    )
    return MeshData(vertices=vertices, faces=faces, name="cube")


def _cylinder_mesh(radius: float, half_height: float, sides: int = 32) -> MeshData:
    ring = []
    for index in range(sides):
        angle = 2.0 * math.pi * float(index) / float(sides)
        ring.append([radius * math.cos(angle), radius * math.sin(angle)])
    vertices = [[0.0, 0.0, -half_height], [0.0, 0.0, half_height]]
    for x, y in ring:
        vertices.append([x, y, -half_height])
    for x, y in ring:
        vertices.append([x, y, half_height])
    faces: list[list[int]] = []
    for index in range(sides):
        next_index = (index + 1) % sides
        bottom_a = 2 + index
        bottom_b = 2 + next_index
        top_a = 2 + sides + index
        top_b = 2 + sides + next_index
        faces.append([0, bottom_b, bottom_a])
        faces.append([1, top_a, top_b])
        faces.append([bottom_a, bottom_b, top_b])
        faces.append([bottom_a, top_b, top_a])
    return MeshData(vertices=np.asarray(vertices, dtype=np.float32), faces=np.asarray(faces, dtype=np.int32), name="cylinder")


def load_object_mesh(meta: dict[str, object]) -> MeshData:
    kind = str(meta.get("kind", "")).lower()
    if kind == "cube":
        return _box_mesh(float(meta.get("size", 0.07)))
    if kind == "cylinder":
        return _cylinder_mesh(float(meta.get("radius", 0.045)), float(meta.get("half_height", 0.165)), int(meta.get("sides", 32)))
    if kind == "drill":
        return _load_binary_stl(DRILL_MESH)
    if kind == "decor01":
        return _load_binary_stl(DECOR01_MESH)
    mesh_path = meta.get("mesh_path")
    if isinstance(mesh_path, str):
        resolved = (ROOT / mesh_path).resolve()
        if resolved.suffix.lower() == ".obj":
            return _load_obj(resolved)
        return _load_binary_stl(resolved)
    raise ValueError(f"Unsupported object kind for DGA-style data conversion: {kind!r}")


def sample_mesh_points(mesh: MeshData, num_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    vertices = mesh.vertices.astype(np.float32, copy=False)
    triangles = vertices[mesh.faces]
    edges_a = triangles[:, 1] - triangles[:, 0]
    edges_b = triangles[:, 2] - triangles[:, 0]
    face_normals = np.cross(edges_a, edges_b)
    face_areas = np.linalg.norm(face_normals, axis=1)
    if not np.any(face_areas > 0.0):
        raise ValueError(f"Mesh {mesh.name!r} has no valid triangle area for point sampling.")
    face_normals /= np.maximum(face_areas[:, None], 1.0e-8)
    probabilities = face_areas / np.sum(face_areas)

    rng = np.random.default_rng(int(seed))
    face_indices = rng.choice(len(mesh.faces), size=int(num_points), replace=True, p=probabilities)
    sampled_triangles = triangles[face_indices]
    sampled_normals = face_normals[face_indices]

    r1 = np.sqrt(rng.random(int(num_points), dtype=np.float32))
    r2 = rng.random(int(num_points), dtype=np.float32)
    points = (
        (1.0 - r1)[:, None] * sampled_triangles[:, 0]
        + (r1 * (1.0 - r2))[:, None] * sampled_triangles[:, 1]
        + (r1 * r2)[:, None] * sampled_triangles[:, 2]
    )
    return points.astype(np.float32), sampled_normals.astype(np.float32)
