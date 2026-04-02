from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from grasp_gen.hand import Hand, InitConfig


ORTHO6D_EPS = 1.0e-8


@dataclass(frozen=True)
class _KinematicSpec:
    parent_indices: np.ndarray
    body_local_pos: jax.Array
    body_local_rot: jax.Array
    joint_qpos_indices: np.ndarray
    joint_axes: jax.Array


@dataclass(frozen=True)
class DgaHandPointSpec:
    side: str
    kin: _KinematicSpec
    surface_body_indices: jax.Array
    surface_local_positions: jax.Array
    distance_body_indices: jax.Array
    distance_local_positions: jax.Array
    key_body_indices: jax.Array
    key_local_positions: jax.Array


def _safe_normalize_jax(vector: jax.Array, fallback: jax.Array) -> jax.Array:
    norm = jnp.linalg.norm(vector, axis=-1, keepdims=True)
    normalized = vector / jnp.maximum(norm, ORTHO6D_EPS)
    return jnp.where(norm > ORTHO6D_EPS, normalized, fallback)


def _orthogonal_vector_jax(primary: jax.Array) -> jax.Array:
    helper_x = jnp.asarray([1.0, 0.0, 0.0], dtype=primary.dtype)
    helper_y = jnp.asarray([0.0, 1.0, 0.0], dtype=primary.dtype)
    helper = jnp.where(jnp.abs(primary[..., :1]) < 0.9, helper_x, helper_y)
    ortho = jnp.cross(primary, helper)
    fallback = jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0], dtype=primary.dtype), primary.shape)
    return _safe_normalize_jax(ortho, fallback)


def _ortho6d_to_matrix_jax(ortho6d: jax.Array) -> jax.Array:
    ortho6d = jnp.asarray(ortho6d, dtype=jnp.float32)
    first = _safe_normalize_jax(
        ortho6d[..., :3],
        jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0], dtype=ortho6d.dtype), ortho6d[..., :3].shape),
    )
    second_raw = ortho6d[..., 3:6] - jnp.sum(first * ortho6d[..., 3:6], axis=-1, keepdims=True) * first
    second = _safe_normalize_jax(second_raw, _orthogonal_vector_jax(first))
    third = jnp.cross(first, second)
    return jnp.stack([first, second, third], axis=-1)


def _axis_angle_rotation_jax(axis: jax.Array, angle: jax.Array) -> jax.Array:
    axis = _safe_normalize_jax(axis, jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float32))
    x, y, z = axis
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    one_minus_c = 1.0 - c
    return jnp.stack(
        [
            jnp.stack([c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s], axis=-1),
            jnp.stack([y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s], axis=-1),
            jnp.stack([z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c], axis=-1),
        ],
        axis=-2,
    )


def _forward_kinematics_batch(spec: _KinematicSpec, hand_pose: jax.Array) -> tuple[jax.Array, jax.Array]:
    batch_size = int(hand_pose.shape[0])
    root_pos = hand_pose[:, :3]
    root_rot = _ortho6d_to_matrix_jax(hand_pose[:, 3:9])
    hand_qpos = hand_pose[:, 9:]

    body_positions = [root_pos]
    body_rotations = [root_rot]

    for body_index in range(1, len(spec.parent_indices)):
        parent_index = int(spec.parent_indices[body_index])
        parent_pos = body_positions[parent_index]
        parent_rot = body_rotations[parent_index]
        local_pos = spec.body_local_pos[body_index]
        local_rot = spec.body_local_rot[body_index]
        qpos_index = int(spec.joint_qpos_indices[body_index])

        if qpos_index >= 0:
            joint_angle = hand_qpos[:, qpos_index]
            joint_axis = spec.joint_axes[body_index]
            joint_rot = _axis_angle_rotation_jax(joint_axis, joint_angle)
            local_total_rot = jnp.einsum("ij,bjk->bik", local_rot, joint_rot)
        else:
            local_total_rot = jnp.broadcast_to(local_rot[None, :, :], (batch_size, 3, 3))

        body_pos = parent_pos + jnp.einsum("bij,j->bi", parent_rot, local_pos)
        body_rot = jnp.einsum("bij,bjk->bik", parent_rot, local_total_rot)
        body_positions.append(body_pos)
        body_rotations.append(body_rot)

    return jnp.stack(body_positions, axis=1), jnp.stack(body_rotations, axis=1)


def _body_local_points_to_world(
    body_positions: jax.Array,
    body_rotations: jax.Array,
    body_indices: jax.Array,
    local_positions: jax.Array,
) -> jax.Array:
    batch_indices = jnp.arange(body_positions.shape[0], dtype=jnp.int32)[:, None]
    selected_body_positions = body_positions[batch_indices, body_indices]
    selected_body_rotations = body_rotations[batch_indices, body_indices]
    return selected_body_positions + jnp.einsum("bnij,nj->bni", selected_body_rotations, local_positions)


def _key_points_from_surface_cloud(surface_body_indices: np.ndarray, surface_local_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_bodies = np.unique(surface_body_indices)
    key_body_indices: list[int] = []
    key_local_positions: list[np.ndarray] = []
    for body_index in unique_bodies.tolist():
        mask = surface_body_indices == int(body_index)
        body_points = surface_local_positions[mask]
        if body_points.size == 0:
            continue
        key_body_indices.append(int(body_index))
        key_local_positions.append(np.mean(body_points, axis=0, dtype=np.float32))
    if not key_body_indices:
        raise ValueError("Unable to build key point set from hand surface cloud.")
    return np.asarray(key_body_indices, dtype=np.int32), np.asarray(key_local_positions, dtype=np.float32)


@lru_cache(maxsize=2)
def load_dga_hand_point_spec(side: str = "right") -> DgaHandPointSpec:
    hand = Hand(side)
    contact_batch = hand._contact_batch(InitConfig())
    kin = _KinematicSpec(
        parent_indices=hand._kin.parent_indices.copy(),
        body_local_pos=hand._kin.body_local_pos,
        body_local_rot=hand._kin.body_local_rot,
        joint_qpos_indices=hand._kin.joint_qpos_indices.copy(),
        joint_axes=hand._kin.joint_axes,
    )
    surface_body_indices = np.asarray(contact_batch.dense_body_indices, dtype=np.int32)
    surface_local_positions = np.asarray(contact_batch.dense_local_positions, dtype=np.float32)
    key_body_indices, key_local_positions = _key_points_from_surface_cloud(surface_body_indices, surface_local_positions)
    return DgaHandPointSpec(
        side=side,
        kin=kin,
        surface_body_indices=contact_batch.dense_body_indices,
        surface_local_positions=contact_batch.dense_local_positions,
        distance_body_indices=contact_batch.body_indices,
        distance_local_positions=contact_batch.local_positions,
        key_body_indices=jnp.asarray(key_body_indices, dtype=jnp.int32),
        key_local_positions=jnp.asarray(key_local_positions, dtype=jnp.float32),
    )


def full_pose_surface_points(spec: DgaHandPointSpec, full_pose: jax.Array) -> jax.Array:
    body_positions, body_rotations = _forward_kinematics_batch(spec.kin, full_pose)
    return _body_local_points_to_world(body_positions, body_rotations, spec.surface_body_indices, spec.surface_local_positions)


def full_pose_distance_points(spec: DgaHandPointSpec, full_pose: jax.Array) -> jax.Array:
    body_positions, body_rotations = _forward_kinematics_batch(spec.kin, full_pose)
    return _body_local_points_to_world(body_positions, body_rotations, spec.distance_body_indices, spec.distance_local_positions)


def full_pose_key_points(spec: DgaHandPointSpec, full_pose: jax.Array) -> jax.Array:
    body_positions, body_rotations = _forward_kinematics_batch(spec.kin, full_pose)
    return _body_local_points_to_world(body_positions, body_rotations, spec.key_body_indices, spec.key_local_positions)
