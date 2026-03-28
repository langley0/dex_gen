from __future__ import annotations

import numpy as np


def box_mesh(half_size: np.ndarray | tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    hx, hy, hz = np.asarray(half_size, dtype=float).reshape(3)
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
        dtype=float,
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


def cylinder_mesh(
    radius: float,
    half_height: float,
    *,
    sides: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
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
    )

    bottom_center = 2 * sides
    top_center = bottom_center + 1
    faces: list[list[int]] = []
    for i in range(sides):
        j = (i + 1) % sides
        faces.append([i, j, sides + j])
        faces.append([i, sides + j, sides + i])
        faces.append([bottom_center, j, i])
        faces.append([top_center, sides + i, sides + j])
    return vertices, np.asarray(faces, dtype=np.int32)
