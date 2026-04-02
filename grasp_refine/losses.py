from __future__ import annotations

import jax
import jax.numpy as jnp


def _pairwise_sqdist(x: jax.Array, y: jax.Array) -> jax.Array:
    x_sq = jnp.sum(jnp.square(x), axis=-1, keepdims=True)
    y_sq = jnp.sum(jnp.square(y), axis=-1)[..., None, :]
    xy = jnp.einsum("bid,bjd->bij", x, y)
    return jnp.maximum(x_sq + y_sq - 2.0 * xy, 0.0)


def _batch_gather_points(points: jax.Array, indices: jax.Array) -> jax.Array:
    batch_index = jnp.arange(points.shape[0], dtype=jnp.int32)[:, None]
    return points[batch_index, indices]


def erf_loss(obj_pcd_nor: jax.Array, hand_pcd: jax.Array) -> jax.Array:
    obj_pcd = obj_pcd_nor[:, :, :3]
    obj_nor = obj_pcd_nor[:, :, 3:6]
    sqdist = _pairwise_sqdist(hand_pcd, obj_pcd)
    indices = jnp.argmin(sqdist, axis=-1)
    distances = jnp.sqrt(jnp.take_along_axis(sqdist, indices[:, :, None], axis=-1)[..., 0])
    nearest_points = _batch_gather_points(obj_pcd, indices)
    nearest_normals = _batch_gather_points(obj_nor, indices)
    positive_sign = jnp.sum((nearest_points - hand_pcd) * nearest_normals, axis=-1) > 0.0
    collision_value = jnp.where(positive_sign, distances, 0.0)
    return jnp.mean(jnp.max(collision_value, axis=1))


def spf_loss(dis_points: jax.Array, obj_pcd: jax.Array, threshold: float = 0.02) -> jax.Array:
    sqdist = _pairwise_sqdist(dis_points, obj_pcd)
    min_sq = jnp.min(sqdist, axis=-1)
    mask = min_sq < float(threshold) ** 2
    return jnp.sum(jnp.where(mask, jnp.sqrt(min_sq), 0.0)) / (jnp.sum(mask.astype(jnp.float32)) + 1.0e-5)


def srf_loss(points: jax.Array, threshold: float = 0.02) -> jax.Array:
    batch_size = points.shape[0]
    dist = jnp.sqrt(jnp.sum(jnp.square(points[:, :, None, :] - points[:, None, :, :] + 1.0e-13), axis=-1))
    eye = jnp.eye(points.shape[1], dtype=bool)[None, :, :]
    dist = jnp.where(eye, 1.0e6, dist)
    penalty = jnp.maximum(float(threshold) - dist, 0.0)
    return jnp.sum(penalty) / float(batch_size)
