from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


BALANCE_EPS = 1.0e-8


class EquilibriumTerms(NamedTuple):
    energy: jax.Array
    force_residual: jax.Array
    torque_residual: jax.Array
    sum_force: jax.Array
    sum_torque: jax.Array
    contact_weights: jax.Array


def triangle_normals_local_np(triangles_local: np.ndarray) -> np.ndarray:
    triangles_local = np.asarray(triangles_local, dtype=float)
    ab = triangles_local[:, 1] - triangles_local[:, 0]
    ac = triangles_local[:, 2] - triangles_local[:, 0]
    normals = np.cross(ab, ac)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    fallback = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=float), (len(triangles_local), 1))
    normals = np.where(norm > BALANCE_EPS, normals / np.maximum(norm, BALANCE_EPS), fallback)
    return np.asarray(normals, dtype=np.float32)


def mesh_scale_np(vertices: np.ndarray) -> float:
    centered = np.asarray(vertices, dtype=float) - np.mean(np.asarray(vertices, dtype=float), axis=0, keepdims=True)
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    return max(scale, 1.0e-6)


def zero_terms(
    batch_size: int,
    contact_count: int,
    *,
    dtype: jax.Array,
) -> EquilibriumTerms:
    zeros = jnp.zeros((batch_size,), dtype=dtype.dtype)
    zero_vec = jnp.zeros((batch_size, 3), dtype=dtype.dtype)
    weights = jnp.full((batch_size, contact_count), 1.0 / max(contact_count, 1), dtype=dtype.dtype)
    return EquilibriumTerms(
        energy=zeros,
        force_residual=zeros,
        torque_residual=zeros,
        sum_force=zero_vec,
        sum_torque=zero_vec,
        contact_weights=weights,
    )


def _uniform_weights(batch_size: int, contact_count: int, dtype: jax.Array) -> jax.Array:
    return jnp.full((batch_size, contact_count), 1.0 / float(contact_count), dtype=dtype.dtype)


def _weighted_result(
    forces: jax.Array,
    torques: jax.Array,
    weights: jax.Array,
    object_scale: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    weighted_force = jnp.einsum("bk,bkj->bj", weights, forces)
    weighted_torque = jnp.einsum("bk,bkj->bj", weights, torques)
    force_residual = jnp.linalg.norm(weighted_force, axis=1)
    torque_residual = jnp.linalg.norm(weighted_torque, axis=1)
    torque_scaled = torque_residual / jnp.maximum(object_scale, BALANCE_EPS)
    return weighted_force, weighted_torque, force_residual, torque_residual, torque_scaled


def _project_simplex(weights: jax.Array) -> jax.Array:
    sorted_weights = jnp.sort(weights, axis=1)[:, ::-1]
    partial = jnp.cumsum(sorted_weights, axis=1) - 1.0
    index = jnp.arange(weights.shape[1], dtype=weights.dtype)[None, :] + 1.0
    mask = sorted_weights - partial / index > 0.0
    rho = jnp.sum(mask, axis=1, dtype=jnp.int32) - 1
    theta = jnp.take_along_axis(partial, rho[:, None], axis=1)[:, 0] / (rho.astype(weights.dtype) + 1.0)
    return jnp.maximum(weights - theta[:, None], 0.0)


def _solve_wrench_weights(wrench: jax.Array, iterations: int) -> jax.Array:
    batch_size, _, contact_count = wrench.shape
    gram = jnp.einsum("brk,brl->bkl", wrench, wrench)
    eig_max = jnp.max(jnp.linalg.eigvalsh(gram), axis=1)
    step = 1.0 / jnp.maximum(eig_max, BALANCE_EPS)
    weights = _uniform_weights(batch_size, contact_count, wrench)

    def body_fn(_: int, current: jax.Array) -> jax.Array:
        grad = jnp.einsum("bkl,bl->bk", gram, current)
        proposal = current - step[:, None] * grad
        return _project_simplex(proposal)

    return jax.lax.fori_loop(0, int(iterations), body_fn, weights)


def wrench_terms(
    nearest_points_world: jax.Array,
    nearest_normals_world: jax.Array,
    object_center_world: jax.Array,
    object_scale: jax.Array,
    *,
    iterations: int,
) -> EquilibriumTerms:
    forces = -nearest_normals_world
    torques = jnp.cross(nearest_points_world - object_center_world[None, None, :], forces)
    scaled_torques = torques / jnp.maximum(object_scale, BALANCE_EPS)[:, None, None]
    wrench = jnp.concatenate([forces, scaled_torques], axis=2)
    weights = _solve_wrench_weights(jnp.swapaxes(wrench, 1, 2), iterations)
    sum_force, sum_torque, force_residual, torque_residual, torque_scaled = _weighted_result(
        forces,
        torques,
        weights,
        object_scale,
    )
    energy = jnp.sqrt(jnp.square(force_residual) + jnp.square(torque_scaled))
    return EquilibriumTerms(
        energy=energy,
        force_residual=force_residual,
        torque_residual=torque_residual,
        sum_force=sum_force,
        sum_torque=sum_torque,
        contact_weights=weights,
    )
