from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .prop import Prop


VOXEL_EPS = 1.0e-12


@dataclass(frozen=True)
class PropSDFConfig:
    voxel_size: float = 3.0e-3
    padding: float = 1.0e-2


@dataclass(frozen=True)
class PropSDFGrid:
    min_corner_local: np.ndarray
    voxel_size: float
    values: np.ndarray

    @property
    def max_corner_local(self) -> np.ndarray:
        shape = np.asarray(self.values.shape, dtype=float) - 1.0
        return self.min_corner_local + self.voxel_size * shape


def _sample_triangle(triangle: np.ndarray, step: float) -> np.ndarray:
    a, b, c = triangle
    edges = (np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b))
    count = max(int(np.ceil(max(edges) / max(step, VOXEL_EPS))), 1)
    points: list[np.ndarray] = []
    for i in range(count + 1):
        for j in range(count + 1 - i):
            u = float(i) / float(count)
            v = float(j) / float(count)
            w = 1.0 - u - v
            points.append(u * a + v * b + w * c)
    return np.asarray(points, dtype=np.float32)


def _surface_samples(vertices: np.ndarray, faces: np.ndarray, voxel_size: float) -> np.ndarray:
    sample_step = 0.5 * float(voxel_size)
    parts = [_sample_triangle(vertices[face], sample_step) for face in faces]
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 3), dtype=np.float32)


def build_prop_sdf_grid(prop: Prop, cfg: PropSDFConfig | None = None) -> PropSDFGrid:
    cfg = PropSDFConfig() if cfg is None else cfg
    if cfg.voxel_size <= 0.0:
        raise ValueError("cfg.voxel_size must be positive.")
    if cfg.padding < 0.0:
        raise ValueError("cfg.padding must be non-negative.")

    vertices = np.asarray(prop.vertices, dtype=np.float32)
    faces = np.asarray(prop.faces, dtype=np.int32)
    surface_points = _surface_samples(vertices, faces, cfg.voxel_size)

    min_corner = np.min(vertices, axis=0) - float(cfg.padding)
    max_corner = np.max(vertices, axis=0) + float(cfg.padding)
    grid_shape = np.maximum(np.ceil((max_corner - min_corner) / float(cfg.voxel_size)).astype(np.int32) + 1, 3)

    boundary = np.zeros(tuple(int(v) for v in grid_shape.tolist()), dtype=bool)
    if len(surface_points) > 0:
        indices = np.floor((surface_points - min_corner[None, :]) / float(cfg.voxel_size)).astype(np.int32)
        indices = np.clip(indices, 0, grid_shape[None, :] - 1)
        boundary[indices[:, 0], indices[:, 1], indices[:, 2]] = True

    structure = np.ones((3, 3, 3), dtype=bool)
    boundary = ndimage.binary_closing(boundary, structure=structure, iterations=1)
    solid = ndimage.binary_fill_holes(boundary)

    outside = ndimage.distance_transform_edt(~solid, sampling=float(cfg.voxel_size)).astype(np.float32)
    inside = ndimage.distance_transform_edt(solid, sampling=float(cfg.voxel_size)).astype(np.float32)
    half = np.float32(0.5 * float(cfg.voxel_size))
    values = outside - half
    values[solid] = -(inside[solid] - half)

    return PropSDFGrid(
        min_corner_local=min_corner.astype(np.float32),
        voxel_size=float(cfg.voxel_size),
        values=values.astype(np.float32),
    )
