from __future__ import annotations

import math

import numpy as np


def box_mesh(size_xyz: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    size = np.asarray(size_xyz, dtype=float).reshape(3)
    if np.any(size <= 0.0):
        raise ValueError("Box size entries must be positive.")
    hx, hy, hz = 0.5 * size
    vertices = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def cylinder_mesh(radius: float, half_height: float, *, sides: int = 32) -> tuple[np.ndarray, np.ndarray]:
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    if half_height <= 0.0:
        raise ValueError("half_height must be positive.")
    if sides < 3:
        raise ValueError("sides must be at least 3.")

    theta = np.linspace(0.0, 2.0 * np.pi, num=sides, endpoint=False, dtype=float)
    ring = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    bottom = np.column_stack([ring, -np.full(sides, half_height, dtype=float)])
    top = np.column_stack([ring, np.full(sides, half_height, dtype=float)])
    vertices = np.vstack(
        [
            bottom,
            top,
            np.array([[0.0, 0.0, -half_height], [0.0, 0.0, half_height]], dtype=float),
        ]
    ).astype(np.float32)

    bottom_center = 2 * sides
    top_center = bottom_center + 1
    faces: list[list[int]] = []
    for index in range(sides):
        next_index = (index + 1) % sides
        faces.append([index, next_index, sides + next_index])
        faces.append([index, sides + next_index, sides + index])
        faces.append([bottom_center, next_index, index])
        faces.append([top_center, sides + index, sides + next_index])
    return vertices, np.asarray(faces, dtype=np.int32)


def regular_prism_mesh(radius: float, half_height: float, *, sides: int) -> tuple[np.ndarray, np.ndarray]:
    return cylinder_mesh(radius=radius, half_height=half_height, sides=sides)


def ellipsoid_mesh(
    radii_xyz: tuple[float, float, float],
    *,
    latitude_segments: int = 16,
    longitude_segments: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    radii = np.asarray(radii_xyz, dtype=float).reshape(3)
    if np.any(radii <= 0.0):
        raise ValueError("Ellipsoid radii must be positive.")
    if latitude_segments < 3:
        raise ValueError("latitude_segments must be at least 3.")
    if longitude_segments < 3:
        raise ValueError("longitude_segments must be at least 3.")

    vertices: list[list[float]] = [[0.0, 0.0, float(radii[2])]]
    for lat_index in range(1, latitude_segments):
        phi = math.pi * float(lat_index) / float(latitude_segments)
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        for lon_index in range(longitude_segments):
            theta = 2.0 * math.pi * float(lon_index) / float(longitude_segments)
            vertices.append(
                [
                    float(radii[0] * sin_phi * math.cos(theta)),
                    float(radii[1] * sin_phi * math.sin(theta)),
                    float(radii[2] * cos_phi),
                ]
            )
    south_pole_index = len(vertices)
    vertices.append([0.0, 0.0, -float(radii[2])])

    faces: list[list[int]] = []
    first_ring_start = 1
    for lon_index in range(longitude_segments):
        next_index = (lon_index + 1) % longitude_segments
        faces.append([0, first_ring_start + next_index, first_ring_start + lon_index])

    ring_count = latitude_segments - 1
    for ring_index in range(ring_count - 1):
        ring_start = 1 + ring_index * longitude_segments
        next_ring_start = ring_start + longitude_segments
        for lon_index in range(longitude_segments):
            next_index = (lon_index + 1) % longitude_segments
            current = ring_start + lon_index
            current_next = ring_start + next_index
            below = next_ring_start + lon_index
            below_next = next_ring_start + next_index
            faces.append([current, current_next, below_next])
            faces.append([current, below_next, below])

    last_ring_start = 1 + (ring_count - 1) * longitude_segments
    for lon_index in range(longitude_segments):
        next_index = (lon_index + 1) % longitude_segments
        faces.append([south_pole_index, last_ring_start + lon_index, last_ring_start + next_index])

    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def octahedron_mesh(radius: float) -> tuple[np.ndarray, np.ndarray]:
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    r = float(radius)
    vertices = np.asarray(
        [
            [r, 0.0, 0.0],
            [-r, 0.0, 0.0],
            [0.0, r, 0.0],
            [0.0, -r, 0.0],
            [0.0, 0.0, r],
            [0.0, 0.0, -r],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 4, 2],
            [2, 4, 1],
            [1, 4, 3],
            [3, 4, 0],
            [2, 5, 0],
            [1, 5, 2],
            [3, 5, 1],
            [0, 5, 3],
        ],
        dtype=np.int32,
    )
    return vertices, faces
