from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SURFACE_EPS = 1.0e-12


@dataclass(frozen=True)
class SurfaceCloudConfig:
    spacing: float = 0.01
    seed: int = 0
    oversample: int = 6
    min_points: int = 256
    max_points: int | None = 4096


@dataclass(frozen=True)
class SurfacePointCloud:
    points_local: np.ndarray
    normals_local: np.ndarray
    area_weights: np.ndarray
    spacing: float
    surface_area: float


def _mesh_triangles(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {faces.shape}")
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("mesh must contain vertices and faces.")
    if np.min(faces) < 0 or np.max(faces) >= len(vertices):
        raise ValueError("faces contain an out-of-range vertex index.")
    return np.asarray(vertices[faces], dtype=np.float32)


def _triangle_areas_and_normals(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edge_01 = triangles[:, 1] - triangles[:, 0]
    edge_02 = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(edge_01, edge_02)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    areas = 0.5 * norms[:, 0]
    safe_normals = np.divide(
        cross,
        np.clip(norms, SURFACE_EPS, None),
        out=np.zeros_like(cross),
        where=norms > SURFACE_EPS,
    )
    return areas.astype(np.float32), safe_normals.astype(np.float32)


def _target_point_count(surface_area: float, config: SurfaceCloudConfig) -> int:
    if config.spacing <= 0.0:
        raise ValueError("SurfaceCloudConfig.spacing must be positive.")
    if config.oversample <= 0:
        raise ValueError("SurfaceCloudConfig.oversample must be positive.")
    if config.min_points <= 0:
        raise ValueError("SurfaceCloudConfig.min_points must be positive.")

    base = int(np.ceil(float(surface_area) / float(config.spacing * config.spacing)))
    target = max(base, int(config.min_points))
    if config.max_points is not None:
        if int(config.max_points) <= 0:
            raise ValueError("SurfaceCloudConfig.max_points must be positive when set.")
        target = min(target, int(config.max_points))
    return max(target, 1)


def _sample_surface_candidates(
    triangles: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    candidate_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    total_area = float(np.sum(areas))
    if total_area <= SURFACE_EPS:
        raise ValueError("mesh surface area must be positive.")

    rng = np.random.default_rng(int(seed))
    triangle_indices = rng.choice(len(triangles), size=int(candidate_count), p=areas / total_area)
    selected = triangles[triangle_indices]
    selected_normals = normals[triangle_indices]

    u = rng.random(int(candidate_count), dtype=np.float32)
    v = rng.random(int(candidate_count), dtype=np.float32)
    sqrt_u = np.sqrt(u)
    w0 = 1.0 - sqrt_u
    w1 = sqrt_u * (1.0 - v)
    w2 = sqrt_u * v
    points = (
        w0[:, None] * selected[:, 0]
        + w1[:, None] * selected[:, 1]
        + w2[:, None] * selected[:, 2]
    ).astype(np.float32)
    return points, selected_normals.astype(np.float32)


def _farthest_point_indices(points: np.ndarray, target_count: int) -> np.ndarray:
    point_count = int(points.shape[0])
    if target_count >= point_count:
        return np.arange(point_count, dtype=np.int32)

    selected = np.empty(target_count, dtype=np.int32)
    centroid = np.mean(points, axis=0)
    distances = np.sum((points - centroid[None, :]) ** 2, axis=1)
    selected[0] = int(np.argmax(distances))

    min_distance_sq = np.sum((points - points[selected[0]][None, :]) ** 2, axis=1)
    for index in range(1, target_count):
        next_index = int(np.argmax(min_distance_sq))
        selected[index] = next_index
        distance_sq = np.sum((points - points[next_index][None, :]) ** 2, axis=1)
        min_distance_sq = np.minimum(min_distance_sq, distance_sq)
    return selected


def build_surface_point_cloud(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    config: SurfaceCloudConfig | None = None,
) -> SurfacePointCloud:
    config = SurfaceCloudConfig() if config is None else config
    triangles = _mesh_triangles(vertices, faces)
    areas, normals = _triangle_areas_and_normals(triangles)
    surface_area = float(np.sum(areas))
    if surface_area <= SURFACE_EPS:
        raise ValueError("mesh surface area must be positive.")

    target_count = _target_point_count(surface_area, config)
    candidate_count = max(target_count, int(config.oversample) * target_count)
    candidates, candidate_normals = _sample_surface_candidates(
        triangles,
        normals,
        areas,
        candidate_count=candidate_count,
        seed=int(config.seed),
    )
    selected = _farthest_point_indices(candidates, target_count)
    points_local = np.asarray(candidates[selected], dtype=np.float32)
    normals_local = np.asarray(candidate_normals[selected], dtype=np.float32)
    area_weights = np.full(target_count, surface_area / float(target_count), dtype=np.float32)
    return SurfacePointCloud(
        points_local=points_local,
        normals_local=normals_local,
        area_weights=area_weights,
        spacing=float(config.spacing),
        surface_area=surface_area,
    )
