from __future__ import annotations

import numpy as np


def normalize_vector(vector: np.ndarray, *, eps: float = 1.0e-8) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm <= eps:
        return np.zeros_like(array, dtype=np.float32)
    return (array / norm).astype(np.float32)


def quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-8:
        quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        quat = quat / norm
    w, x, y, z = quat
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def ortho6d_to_matrix(ortho6d: np.ndarray) -> np.ndarray:
    values = np.asarray(ortho6d, dtype=np.float32).reshape(6)
    x_raw = values[:3]
    y_raw = values[3:]
    x_axis = normalize_vector(x_raw)
    z_axis = normalize_vector(np.cross(x_axis, y_raw))
    if not np.any(z_axis):
        z_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    y_axis = normalize_vector(np.cross(z_axis, x_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32)
