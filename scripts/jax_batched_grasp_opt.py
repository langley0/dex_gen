#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np


PENETRATION_SURFACE_EPS = 1.0e-3
ROOT_HEIGHT_FLOOR = 0.03
ORTHO6D_EPS = 1.0e-8
RMS_DENOM_EPS = 1.0e-6
BEST_SCORE_EPS = 1.0e-12


class HandKinematicsSpec(NamedTuple):
    parent_indices: np.ndarray
    body_local_pos: jax.Array
    body_local_rot: jax.Array
    joint_qpos_indices: np.ndarray
    joint_axes: jax.Array


class BatchedSceneArrays(NamedTuple):
    point_body_indices: jax.Array
    point_local_positions: jax.Array
    penetration_body_indices: jax.Array
    penetration_local_positions: jax.Array
    penetration_area_weights: jax.Array
    qpos_lower: jax.Array
    qpos_upper: jax.Array
    object_center: jax.Array
    object_geom_size: jax.Array
    object_geom_pos: jax.Array
    object_geom_rot: jax.Array


class BatchedMetrics(NamedTuple):
    score: jax.Array
    e_dis: jax.Array
    e_pen: jax.Array
    penetration_depth: jax.Array
    selected_penetration: jax.Array


class BatchedOptimizerState(NamedTuple):
    hand_pose: jax.Array
    contact_indices: jax.Array
    best_hand_pose: jax.Array
    best_contact_indices: jax.Array
    best_metrics: BatchedMetrics
    ema_grad: jax.Array
    accepted_steps: jax.Array
    rejected_steps: jax.Array
    rng_key: jax.Array


class BatchedStepSnapshot(NamedTuple):
    current_hand_pose: jax.Array
    current_contact_indices: jax.Array
    current_metrics: BatchedMetrics
    accepted: jax.Array


@dataclass(frozen=True)
class BatchedSampleResult:
    initial_contact_indices: tuple[int, ...]
    best_state_vector: np.ndarray
    best_contact_indices: tuple[int, ...]
    accepted_steps: int
    rejected_steps: int
    final_temperature: float
    final_step_size: float
    trace: list[dict[str, Any]]
    step_stats: list[dict[str, Any]]


def _quat_to_matrix_np(quat: np.ndarray) -> np.ndarray:
    matrix = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=float))
    return matrix.reshape(3, 3)


def _normalize_vector_np(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= ORTHO6D_EPS:
        return np.asarray(fallback, dtype=float)
    return np.asarray(vector, dtype=float) / norm


def _orthogonal_vector_np(primary: np.ndarray) -> np.ndarray:
    helper = np.array([1.0, 0.0, 0.0], dtype=float) if abs(float(primary[0])) < 0.9 else np.array(
        [0.0, 1.0, 0.0],
        dtype=float,
    )
    ortho = np.cross(primary, helper)
    return _normalize_vector_np(ortho, np.array([0.0, 0.0, 1.0], dtype=float))


def _matrix_to_ortho6d_np(rotation: np.ndarray) -> np.ndarray:
    return np.concatenate([rotation[:, 0], rotation[:, 1]], axis=0).astype(np.float32)


def _ortho6d_to_matrix_np(ortho6d: np.ndarray) -> np.ndarray:
    ortho6d = np.asarray(ortho6d, dtype=float)
    first = _normalize_vector_np(ortho6d[:3], np.array([1.0, 0.0, 0.0], dtype=float))
    second_raw = ortho6d[3:6] - first * float(np.dot(first, ortho6d[3:6]))
    second = _normalize_vector_np(second_raw, _orthogonal_vector_np(first))
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=1)


def _rotation_matrix_to_quaternion_np(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm <= ORTHO6D_EPS:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    quat /= norm
    if quat[0] < 0.0:
        quat *= -1.0
    return quat


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


def _ensure_supported_scene(scenes: list[Any]) -> None:
    if not scenes:
        raise ValueError("At least one optimization scene is required.")
    first_scene = scenes[0]
    if first_scene.object_geom_type is None:
        raise NotImplementedError("JAX batched optimization requires a single primitive object geom.")
    supported_geom_types = {
        int(mujoco.mjtGeom.mjGEOM_BOX),
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    }
    if int(first_scene.object_geom_type) not in supported_geom_types:
        raise NotImplementedError(
            f"JAX batched optimization does not support geom type {int(first_scene.object_geom_type)}."
        )
    point_count = len(first_scene.point_records)
    penetration_count = int(first_scene.penetration_local_positions.shape[0])
    for scene in scenes[1:]:
        if scene.object_geom_type is None or int(scene.object_geom_type) != int(first_scene.object_geom_type):
            raise NotImplementedError("Mixed or unsupported object geom types are not supported in batched mode.")
        if len(scene.point_records) != point_count:
            raise ValueError("All batched scenes must share the same hand point count.")
        if int(scene.penetration_local_positions.shape[0]) != penetration_count:
            raise ValueError("All batched scenes must share the same penetration sample count.")


def _extract_hand_kinematics_spec(scene: Any) -> tuple[HandKinematicsSpec, dict[int, int]]:
    model = scene.model
    hand_body_ids = [
        body_id
        for body_id in range(model.nbody)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "").startswith("inspire_")
    ]
    if not hand_body_ids:
        raise ValueError("No Inspire hand bodies were found in the optimization scene.")
    if scene.root_body_id not in hand_body_ids:
        raise ValueError("The optimization scene root body is not part of the extracted hand body set.")
    hand_body_ids = [scene.root_body_id] + [body_id for body_id in hand_body_ids if body_id != scene.root_body_id]
    id_map = {old_id: new_id for new_id, old_id in enumerate(hand_body_ids)}

    parent_indices: list[int] = []
    body_local_pos: list[np.ndarray] = []
    body_local_rot: list[np.ndarray] = []
    joint_qpos_indices: list[int] = []
    joint_axes: list[np.ndarray] = []

    for new_body_id, old_body_id in enumerate(hand_body_ids):
        if new_body_id == 0:
            parent_indices.append(-1)
            body_local_pos.append(np.zeros(3, dtype=np.float32))
            body_local_rot.append(np.eye(3, dtype=np.float32))
            joint_qpos_indices.append(-1)
            joint_axes.append(np.zeros(3, dtype=np.float32))
            continue

        parent_old_id = int(model.body_parentid[old_body_id])
        if parent_old_id not in id_map:
            raise ValueError("Encountered a hand body whose parent is outside the extracted hand kinematic tree.")
        parent_indices.append(id_map[parent_old_id])
        body_local_pos.append(np.asarray(model.body_pos[old_body_id], dtype=np.float32))
        body_local_rot.append(_quat_to_matrix_np(model.body_quat[old_body_id]).astype(np.float32))

        joint_count = int(model.body_jntnum[old_body_id])
        if joint_count == 0:
            joint_qpos_indices.append(-1)
            joint_axes.append(np.zeros(3, dtype=np.float32))
            continue
        if joint_count != 1:
            raise NotImplementedError("Batched JAX kinematics currently supports at most one joint per body.")
        joint_id = int(model.body_jntadr[old_body_id])
        if np.linalg.norm(model.jnt_pos[joint_id]) > 1.0e-8:
            raise NotImplementedError("Batched JAX kinematics currently requires zero joint offsets.")
        joint_qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        joint_axes.append(np.asarray(model.jnt_axis[joint_id], dtype=np.float32))

    return (
        HandKinematicsSpec(
            parent_indices=np.asarray(parent_indices, dtype=np.int32),
            body_local_pos=jnp.asarray(np.stack(body_local_pos, axis=0), dtype=jnp.float32),
            body_local_rot=jnp.asarray(np.stack(body_local_rot, axis=0), dtype=jnp.float32),
            joint_qpos_indices=np.asarray(joint_qpos_indices, dtype=np.int32),
            joint_axes=jnp.asarray(np.stack(joint_axes, axis=0), dtype=jnp.float32),
        ),
        id_map,
    )


def _extract_batched_scene_arrays(scenes: list[Any]) -> tuple[HandKinematicsSpec, BatchedSceneArrays]:
    _ensure_supported_scene(scenes)
    hand_spec, body_id_map = _extract_hand_kinematics_spec(scenes[0])

    point_body_indices = []
    point_local_positions = []
    penetration_body_indices = []
    penetration_local_positions = []
    penetration_area_weights = []
    object_center = []
    object_geom_size = []
    object_geom_pos = []
    object_geom_rot = []

    for scene in scenes:
        point_body_indices.append(np.asarray([body_id_map[int(body_id)] for body_id in scene.point_body_ids], dtype=np.int32))
        point_local_positions.append(np.asarray(scene.point_local_positions, dtype=np.float32))
        penetration_body_indices.append(
            np.asarray([body_id_map[int(body_id)] for body_id in scene.penetration_body_index_array], dtype=np.int32)
        )
        penetration_local_positions.append(np.asarray(scene.penetration_local_positions, dtype=np.float32))
        penetration_area_weights.append(np.asarray(scene.penetration_area_weights, dtype=np.float32))
        object_center.append(np.asarray(scene.object_center, dtype=np.float32))
        object_geom_size.append(np.asarray(scene.object_geom_size, dtype=np.float32))
        object_geom_pos.append(np.asarray(scene.object_geom_pos, dtype=np.float32))
        object_geom_rot.append(np.asarray(scene.object_geom_rot, dtype=np.float32))

    first_scene = scenes[0]
    scene_arrays = BatchedSceneArrays(
        point_body_indices=jnp.asarray(np.stack(point_body_indices, axis=0), dtype=jnp.int32),
        point_local_positions=jnp.asarray(np.stack(point_local_positions, axis=0), dtype=jnp.float32),
        penetration_body_indices=jnp.asarray(np.stack(penetration_body_indices, axis=0), dtype=jnp.int32),
        penetration_local_positions=jnp.asarray(np.stack(penetration_local_positions, axis=0), dtype=jnp.float32),
        penetration_area_weights=jnp.asarray(np.stack(penetration_area_weights, axis=0), dtype=jnp.float32),
        qpos_lower=jnp.asarray(first_scene.qpos_lower, dtype=jnp.float32),
        qpos_upper=jnp.asarray(first_scene.qpos_upper, dtype=jnp.float32),
        object_center=jnp.asarray(np.stack(object_center, axis=0), dtype=jnp.float32),
        object_geom_size=jnp.asarray(np.stack(object_geom_size, axis=0), dtype=jnp.float32),
        object_geom_pos=jnp.asarray(np.stack(object_geom_pos, axis=0), dtype=jnp.float32),
        object_geom_rot=jnp.asarray(np.stack(object_geom_rot, axis=0), dtype=jnp.float32),
    )
    return hand_spec, scene_arrays


def _project_hand_pose(
    hand_pose: jax.Array,
    scene_arrays: BatchedSceneArrays,
    root_position_margin: float,
) -> jax.Array:
    root_pos = hand_pose[:, :3]
    root_rotation_6d = hand_pose[:, 3:9]
    hand_qpos = hand_pose[:, 9:]

    offset = root_pos - scene_arrays.object_center
    offset_norm = jnp.linalg.norm(offset, axis=1, keepdims=True)
    limited_offset = jnp.where(
        offset_norm > root_position_margin,
        offset * (root_position_margin / jnp.maximum(offset_norm, ORTHO6D_EPS)),
        offset,
    )
    root_pos = scene_arrays.object_center + limited_offset
    root_pos = root_pos.at[:, 2].set(jnp.maximum(root_pos[:, 2], ROOT_HEIGHT_FLOOR))
    hand_qpos = jnp.clip(hand_qpos, scene_arrays.qpos_lower[None, :], scene_arrays.qpos_upper[None, :])
    return jnp.concatenate([root_pos, root_rotation_6d, hand_qpos], axis=1)


def _forward_kinematics_batch(hand_spec: HandKinematicsSpec, hand_pose: jax.Array) -> tuple[jax.Array, jax.Array]:
    batch_size = hand_pose.shape[0]
    root_pos = hand_pose[:, :3]
    root_rot = _ortho6d_to_matrix_jax(hand_pose[:, 3:9])
    hand_qpos = hand_pose[:, 9:]

    body_positions = [root_pos]
    body_rotations = [root_rot]

    for body_index in range(1, len(hand_spec.parent_indices)):
        parent_index = int(hand_spec.parent_indices[body_index])
        parent_pos = body_positions[parent_index]
        parent_rot = body_rotations[parent_index]

        local_pos = hand_spec.body_local_pos[body_index]
        local_rot = hand_spec.body_local_rot[body_index]
        qpos_index = int(hand_spec.joint_qpos_indices[body_index])

        if qpos_index >= 0:
            joint_angle = hand_qpos[:, qpos_index]
            axis = hand_spec.joint_axes[body_index]
            joint_rot = _axis_angle_rotation_jax(axis, joint_angle)
            local_total_rot = jnp.einsum("ij,bjk->bik", local_rot, joint_rot)
        else:
            local_total_rot = jnp.broadcast_to(local_rot[None, :, :], (batch_size, 3, 3))

        body_pos = parent_pos + jnp.einsum("bij,j->bi", parent_rot, local_pos)
        body_rot = jnp.einsum("bij,bjk->bik", parent_rot, local_total_rot)
        body_positions.append(body_pos)
        body_rotations.append(body_rot)

    return jnp.stack(body_positions, axis=1), jnp.stack(body_rotations, axis=1)


def _axis_angle_rotation_jax(axis: jax.Array, angle: jax.Array) -> jax.Array:
    axis = jnp.asarray(axis, dtype=jnp.float32)
    axis = axis / jnp.maximum(jnp.linalg.norm(axis), ORTHO6D_EPS)
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


def _body_local_points_to_world(
    body_positions: jax.Array,
    body_rotations: jax.Array,
    body_indices: jax.Array,
    local_positions: jax.Array,
) -> jax.Array:
    batch_indices = jnp.arange(body_positions.shape[0], dtype=jnp.int32)[:, None]
    selected_body_positions = body_positions[batch_indices, body_indices]
    selected_body_rotations = body_rotations[batch_indices, body_indices]
    return selected_body_positions + jnp.einsum("bnij,bnj->bni", selected_body_rotations, local_positions)


def _box_signed_distance(points_local: jax.Array, half_extents: jax.Array) -> jax.Array:
    q = jnp.abs(points_local) - half_extents[:, None, :]
    outside = jnp.maximum(q, 0.0)
    outside_distance = jnp.linalg.norm(outside, axis=-1)
    inside_distance = jnp.minimum(jnp.max(q, axis=-1), 0.0)
    return outside_distance + inside_distance


def _sphere_signed_distance(points_local: jax.Array, radius: jax.Array) -> jax.Array:
    return jnp.linalg.norm(points_local, axis=-1) - radius[:, None]


def _capsule_signed_distance(points_local: jax.Array, radius: jax.Array, half_length: jax.Array) -> jax.Array:
    segment_z = jnp.clip(points_local[..., 2], -half_length[:, None], half_length[:, None])
    delta = points_local - jnp.stack(
        [
            jnp.zeros_like(segment_z),
            jnp.zeros_like(segment_z),
            segment_z,
        ],
        axis=-1,
    )
    return jnp.linalg.norm(delta, axis=-1) - radius[:, None]


def _cylinder_signed_distance(points_local: jax.Array, radius: jax.Array, half_height: jax.Array) -> jax.Array:
    radial = jnp.sqrt(points_local[..., 0] * points_local[..., 0] + points_local[..., 1] * points_local[..., 1])
    radial_delta = radial - radius[:, None]
    height_delta = jnp.abs(points_local[..., 2]) - half_height[:, None]
    outside_radial = jnp.maximum(radial_delta, 0.0)
    outside_height = jnp.maximum(height_delta, 0.0)
    outside_distance = jnp.sqrt(outside_radial * outside_radial + outside_height * outside_height)
    inside_distance = jnp.minimum(jnp.maximum(radial_delta, height_delta), 0.0)
    return outside_distance + inside_distance


def _make_score_functions(
    hand_spec: HandKinematicsSpec,
    scene_arrays: BatchedSceneArrays,
    object_geom_type: int,
    distance_weight: float,
    penetration_weight: float,
    root_position_margin: float,
):
    if object_geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        sdf_fn = lambda points_local: _box_signed_distance(points_local, scene_arrays.object_geom_size[:, :3])
    elif object_geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        sdf_fn = lambda points_local: _sphere_signed_distance(points_local, scene_arrays.object_geom_size[:, 0])
    elif object_geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        sdf_fn = lambda points_local: _capsule_signed_distance(
            points_local,
            scene_arrays.object_geom_size[:, 0],
            scene_arrays.object_geom_size[:, 1],
        )
    elif object_geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        sdf_fn = lambda points_local: _cylinder_signed_distance(
            points_local,
            scene_arrays.object_geom_size[:, 0],
            scene_arrays.object_geom_size[:, 1],
        )
    else:
        raise NotImplementedError(f"Unsupported batched object geom type: {object_geom_type}.")

    def score_metrics_only(hand_pose: jax.Array, contact_indices: jax.Array) -> tuple[BatchedMetrics, jax.Array]:
        projected_hand_pose = _project_hand_pose(hand_pose, scene_arrays, root_position_margin)
        body_positions, body_rotations = _forward_kinematics_batch(hand_spec, projected_hand_pose)

        point_world_positions = _body_local_points_to_world(
            body_positions,
            body_rotations,
            scene_arrays.point_body_indices,
            scene_arrays.point_local_positions,
        )
        batch_indices = jnp.arange(point_world_positions.shape[0], dtype=jnp.int32)[:, None]
        selected_world_positions = point_world_positions[batch_indices, contact_indices]
        selected_points_local = jnp.einsum(
            "bkj,bji->bki",
            selected_world_positions - scene_arrays.object_geom_pos[:, None, :],
            scene_arrays.object_geom_rot,
        )
        contact_signed_distances = sdf_fn(selected_points_local)
        e_dis = jnp.sum(jnp.abs(contact_signed_distances), axis=1)
        selected_penetration = jnp.any(contact_signed_distances < 0.0, axis=1)

        if scene_arrays.penetration_local_positions.shape[1] == 0:
            e_pen = jnp.zeros_like(e_dis)
            penetration_depth = jnp.zeros_like(e_dis)
        else:
            penetration_world_positions = _body_local_points_to_world(
                body_positions,
                body_rotations,
                scene_arrays.penetration_body_indices,
                scene_arrays.penetration_local_positions,
            )
            penetration_points_local = jnp.einsum(
                "bkj,bji->bki",
                penetration_world_positions - scene_arrays.object_geom_pos[:, None, :],
                scene_arrays.object_geom_rot,
            )
            penetration_signed_distances = sdf_fn(penetration_points_local)
            penetration_depths = jnp.maximum(-(penetration_signed_distances + PENETRATION_SURFACE_EPS), 0.0)
            e_pen = jnp.sum(scene_arrays.penetration_area_weights * penetration_depths, axis=1)
            penetration_depth = jnp.max(penetration_depths, axis=1)

        score = distance_weight * e_dis + penetration_weight * e_pen
        return (
            BatchedMetrics(
                score=score,
                e_dis=e_dis,
                e_pen=e_pen,
                penetration_depth=penetration_depth,
                selected_penetration=selected_penetration,
            ),
            projected_hand_pose,
        )

    def total_score_and_aux(hand_pose: jax.Array, contact_indices: jax.Array):
        metrics, projected_hand_pose = score_metrics_only(hand_pose, contact_indices)
        return jnp.sum(metrics.score), (metrics, projected_hand_pose)

    return (
        jax.jit(score_metrics_only),
        jax.jit(jax.value_and_grad(total_score_and_aux, argnums=0, has_aux=True)),
    )


def _make_step_function(
    score_metrics_only,
    score_and_grad,
    batch_size: int,
    contact_count: int,
    point_count: int,
    starting_temperature: float,
    temperature_decay: float,
    annealing_period: int,
    step_size: float,
    step_size_period: int,
    mu: float,
    switch_possibility: float,
):
    starting_temperature_array = jnp.asarray(starting_temperature, dtype=jnp.float32)
    temperature_decay_array = jnp.asarray(temperature_decay, dtype=jnp.float32)
    step_size_array = jnp.asarray(step_size, dtype=jnp.float32)
    mu_array = jnp.asarray(mu, dtype=jnp.float32)
    switch_possibility_array = jnp.asarray(switch_possibility, dtype=jnp.float32)

    @jax.jit
    def step_fn(
        optimizer_state: BatchedOptimizerState,
        step_index: jax.Array,
    ) -> tuple[BatchedOptimizerState, BatchedStepSnapshot]:
        step_size_power = jnp.floor_divide(step_index, step_size_period).astype(jnp.float32)
        step_size_value = step_size_array * jnp.power(temperature_decay_array, step_size_power)
        temperature_power = jnp.floor_divide(step_index, annealing_period).astype(jnp.float32)
        temperature_value = starting_temperature_array * jnp.power(temperature_decay_array, temperature_power)

        (_, (current_metrics, current_projected_hand_pose)), gradient = score_and_grad(
            optimizer_state.hand_pose,
            optimizer_state.contact_indices,
        )

        ema_grad = (
            mu_array * jnp.mean(jnp.square(gradient), axis=0)
            + (1.0 - mu_array) * optimizer_state.ema_grad
        )
        proposed_hand_pose = current_projected_hand_pose - (
            step_size_value * gradient / (jnp.sqrt(ema_grad)[None, :] + RMS_DENOM_EPS)
        )

        next_rng_key, switch_key, index_key, accept_key = jax.random.split(optimizer_state.rng_key, 4)
        switch_mask = jax.random.uniform(switch_key, shape=(batch_size, contact_count)) < switch_possibility_array
        sampled_contact_indices = jax.random.randint(
            index_key,
            shape=(batch_size, contact_count),
            minval=0,
            maxval=point_count,
        )
        proposed_contact_indices = jnp.where(switch_mask, sampled_contact_indices, optimizer_state.contact_indices)
        proposed_metrics, proposed_projected_hand_pose = score_metrics_only(proposed_hand_pose, proposed_contact_indices)

        alpha = jax.random.uniform(accept_key, shape=(batch_size,))
        accept = alpha < jnp.exp(
            (current_metrics.score - proposed_metrics.score) / jnp.maximum(temperature_value, ORTHO6D_EPS)
        )

        accept_mask = accept[:, None]
        next_hand_pose = jnp.where(accept_mask, proposed_projected_hand_pose, current_projected_hand_pose)
        next_contact_indices = jnp.where(accept_mask, proposed_contact_indices, optimizer_state.contact_indices)
        next_metrics = BatchedMetrics(
            score=jnp.where(accept, proposed_metrics.score, current_metrics.score),
            e_dis=jnp.where(accept, proposed_metrics.e_dis, current_metrics.e_dis),
            e_pen=jnp.where(accept, proposed_metrics.e_pen, current_metrics.e_pen),
            penetration_depth=jnp.where(accept, proposed_metrics.penetration_depth, current_metrics.penetration_depth),
            selected_penetration=jnp.where(accept, proposed_metrics.selected_penetration, current_metrics.selected_penetration),
        )

        is_new_best = next_metrics.score < (optimizer_state.best_metrics.score - BEST_SCORE_EPS)
        best_hand_pose = jnp.where(is_new_best[:, None], next_hand_pose, optimizer_state.best_hand_pose)
        best_contact_indices = jnp.where(
            is_new_best[:, None],
            next_contact_indices,
            optimizer_state.best_contact_indices,
        )
        best_metrics = BatchedMetrics(
            score=jnp.where(is_new_best, next_metrics.score, optimizer_state.best_metrics.score),
            e_dis=jnp.where(is_new_best, next_metrics.e_dis, optimizer_state.best_metrics.e_dis),
            e_pen=jnp.where(is_new_best, next_metrics.e_pen, optimizer_state.best_metrics.e_pen),
            penetration_depth=jnp.where(is_new_best, next_metrics.penetration_depth, optimizer_state.best_metrics.penetration_depth),
            selected_penetration=jnp.where(
                is_new_best,
                next_metrics.selected_penetration,
                optimizer_state.best_metrics.selected_penetration,
            ),
        )

        next_state = BatchedOptimizerState(
            hand_pose=next_hand_pose,
            contact_indices=next_contact_indices,
            best_hand_pose=best_hand_pose,
            best_contact_indices=best_contact_indices,
            best_metrics=best_metrics,
            ema_grad=ema_grad,
            accepted_steps=optimizer_state.accepted_steps + accept.astype(jnp.int32),
            rejected_steps=optimizer_state.rejected_steps + (~accept).astype(jnp.int32),
            rng_key=next_rng_key,
        )
        snapshot = BatchedStepSnapshot(
            current_hand_pose=next_hand_pose,
            current_contact_indices=next_contact_indices,
            current_metrics=next_metrics,
            accepted=accept,
        )
        return next_state, snapshot

    return step_fn


def _seed_to_jax_key(seed: int) -> jax.Array:
    return jax.random.key(np.uint32(int(seed) % (2**32)))


def _extract_initial_hand_pose_vectors(scenes: list[Any]) -> np.ndarray:
    hand_pose_vectors = []
    for scene in scenes:
        root_rotation = _quat_to_matrix_np(scene.initial_state.root_quat)
        root_rotation_6d = _matrix_to_ortho6d_np(root_rotation)
        hand_pose_vectors.append(
            np.concatenate(
                [
                    np.asarray(scene.initial_state.root_pos, dtype=np.float32),
                    root_rotation_6d,
                    np.asarray(scene.initial_state.hand_qpos, dtype=np.float32),
                ],
                axis=0,
            )
        )
    return np.stack(hand_pose_vectors, axis=0)


def _hand_pose_to_packed_state_vector_np(hand_pose: np.ndarray) -> np.ndarray:
    hand_pose = np.asarray(hand_pose, dtype=float)
    root_pos = hand_pose[:3].copy()
    root_rotation = _ortho6d_to_matrix_np(hand_pose[3:9])
    root_quat = _rotation_matrix_to_quaternion_np(root_rotation)
    hand_qpos = hand_pose[9:].copy()
    return np.concatenate([root_pos, root_quat, hand_qpos], axis=0)


def _hand_pose_batch_to_packed_state_vectors_np(hand_pose_batch: np.ndarray) -> np.ndarray:
    return np.stack(
        [_hand_pose_to_packed_state_vector_np(hand_pose) for hand_pose in np.asarray(hand_pose_batch, dtype=float)],
        axis=0,
    )


def _scheduled_value(base: float, decay: float, period: int, step_index: int) -> float:
    exponent = step_index // period
    return float(base * (decay ** exponent))


def run_batched_pose_optimization(
    config: Any,
    scenes: list[Any],
    pose_samples: list[Any],
    contact_finger_fn,
    state_trace_entry_fn,
) -> list[BatchedSampleResult]:
    if len(scenes) != len(pose_samples):
        raise ValueError("The number of optimization scenes must match the number of pose samples.")
    if not scenes:
        return []

    hand_spec, scene_arrays = _extract_batched_scene_arrays(scenes)
    object_geom_type = int(scenes[0].object_geom_type)
    score_metrics_only, score_and_grad = _make_score_functions(
        hand_spec,
        scene_arrays,
        object_geom_type=object_geom_type,
        distance_weight=float(config.loss.distance_weight),
        penetration_weight=float(config.loss.penetration_weight),
        root_position_margin=float(config.optimize.root_position_margin),
    )

    batch_size = len(scenes)
    contact_count = int(config.contacts.contact_count)
    point_count = len(scenes[0].point_records)
    step_fn = _make_step_function(
        score_metrics_only,
        score_and_grad,
        batch_size=batch_size,
        contact_count=contact_count,
        point_count=point_count,
        starting_temperature=float(config.optimize.starting_temperature),
        temperature_decay=float(config.optimize.temperature_decay),
        annealing_period=int(config.optimize.annealing_period),
        step_size=float(config.optimize.step_size),
        step_size_period=int(config.optimize.stepsize_period),
        mu=float(config.optimize.mu),
        switch_possibility=float(config.contacts.switch_possibility),
    )

    rng_key = _seed_to_jax_key(int(config.run.random_seed))
    rng_key, initial_contact_key = jax.random.split(rng_key)
    initial_contact_indices = jax.random.randint(
        initial_contact_key,
        shape=(batch_size, contact_count),
        minval=0,
        maxval=point_count,
    )
    initial_hand_pose = jnp.asarray(_extract_initial_hand_pose_vectors(scenes), dtype=jnp.float32)
    initial_metrics, projected_initial_hand_pose = score_metrics_only(initial_hand_pose, initial_contact_indices)
    optimizer_state = BatchedOptimizerState(
        hand_pose=projected_initial_hand_pose,
        contact_indices=initial_contact_indices,
        best_hand_pose=projected_initial_hand_pose,
        best_contact_indices=initial_contact_indices,
        best_metrics=initial_metrics,
        ema_grad=jnp.zeros(projected_initial_hand_pose.shape[1], dtype=jnp.float32),
        accepted_steps=jnp.zeros(batch_size, dtype=jnp.int32),
        rejected_steps=jnp.zeros(batch_size, dtype=jnp.int32),
        rng_key=rng_key,
    )

    trace_per_sample: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]
    step_stats_per_sample: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]

    initial_hand_pose_host = np.asarray(projected_initial_hand_pose)
    initial_state_host = _hand_pose_batch_to_packed_state_vectors_np(initial_hand_pose_host)
    initial_contacts_host = np.asarray(initial_contact_indices)
    initial_metrics_host = {
        "score": np.asarray(initial_metrics.score),
        "e_dis": np.asarray(initial_metrics.e_dis),
        "e_pen": np.asarray(initial_metrics.e_pen),
        "penetration_depth": np.asarray(initial_metrics.penetration_depth),
        "selected_penetration": np.asarray(initial_metrics.selected_penetration),
    }
    for sample_index in range(batch_size):
        trace_per_sample[sample_index].append(
            state_trace_entry_fn(
                step=0,
                state_vector=initial_state_host[sample_index],
                contact_indices=initial_contacts_host[sample_index],
                score=initial_metrics_host["score"][sample_index],
                e_dis=initial_metrics_host["e_dis"][sample_index],
                e_pen=initial_metrics_host["e_pen"][sample_index],
                penetration_depth=initial_metrics_host["penetration_depth"][sample_index],
                selected_penetration=bool(initial_metrics_host["selected_penetration"][sample_index]),
                temperature=float(config.optimize.starting_temperature),
                step_size=float(config.optimize.step_size),
                accepted=None,
            )
        )

    for step_index in range(int(config.optimize.max_steps)):
        optimizer_state, step_snapshot = step_fn(
            optimizer_state,
            jnp.asarray(step_index, dtype=jnp.int32),
        )

        step_number = step_index + 1
        step_size_value = _scheduled_value(
            float(config.optimize.step_size),
            float(config.optimize.temperature_decay),
            int(config.optimize.stepsize_period),
            step_index,
        )
        temperature_value = _scheduled_value(
            float(config.optimize.starting_temperature),
            float(config.optimize.temperature_decay),
            int(config.optimize.annealing_period),
            step_index,
        )
        should_log = (
            step_number == 1
            or step_number % int(config.optimize.log_period) == 0
            or step_number == int(config.optimize.max_steps)
        )
        should_trace = (
            step_number % int(config.optimize.trace_stride) == 0
            or step_number == int(config.optimize.max_steps)
        )

        snapshot_host: dict[str, np.ndarray] = {}
        if should_log or should_trace:
            hand_pose_host = np.asarray(step_snapshot.current_hand_pose)
            snapshot_host = {
                "current_state": _hand_pose_batch_to_packed_state_vectors_np(hand_pose_host),
                "current_contact_indices": np.asarray(step_snapshot.current_contact_indices),
                "score": np.asarray(step_snapshot.current_metrics.score),
                "e_dis": np.asarray(step_snapshot.current_metrics.e_dis),
                "e_pen": np.asarray(step_snapshot.current_metrics.e_pen),
                "penetration_depth": np.asarray(step_snapshot.current_metrics.penetration_depth),
                "selected_penetration": np.asarray(step_snapshot.current_metrics.selected_penetration),
                "accepted": np.asarray(step_snapshot.accepted),
            }

        if should_log:
            for sample_index, scene in enumerate(scenes):
                current_fingers = contact_finger_fn(
                    tuple(int(value) for value in snapshot_host["current_contact_indices"][sample_index].tolist()),
                    scene.point_records,
                )
                print(
                    f"[pose {pose_samples[sample_index].sample_index:02d}] "
                    f"step={step_number:04d} energy={snapshot_host['score'][sample_index]:.6f} "
                    f"accepted={int(snapshot_host['accepted'][sample_index])} temp={temperature_value:.4f} "
                    f"step_size={step_size_value:.5f} k={len(snapshot_host['current_contact_indices'][sample_index])} "
                    f"fingers={','.join(current_fingers)}"
                )

        if should_trace:
            for sample_index, scene in enumerate(scenes):
                current_indices = snapshot_host["current_contact_indices"][sample_index]
                current_fingers = contact_finger_fn(
                    tuple(int(value) for value in current_indices.tolist()),
                    scene.point_records,
                )
                trace_per_sample[sample_index].append(
                    state_trace_entry_fn(
                        step=step_number,
                        state_vector=snapshot_host["current_state"][sample_index],
                        contact_indices=current_indices,
                        score=snapshot_host["score"][sample_index],
                        e_dis=snapshot_host["e_dis"][sample_index],
                        e_pen=snapshot_host["e_pen"][sample_index],
                        penetration_depth=snapshot_host["penetration_depth"][sample_index],
                        selected_penetration=bool(snapshot_host["selected_penetration"][sample_index]),
                        temperature=temperature_value,
                        step_size=step_size_value,
                        accepted=bool(snapshot_host["accepted"][sample_index]),
                    )
                )
                step_stats_per_sample[sample_index].append(
                    {
                        "step": int(step_number),
                        "score": float(snapshot_host["score"][sample_index]),
                        "accepted": bool(snapshot_host["accepted"][sample_index]),
                        "temperature": float(temperature_value),
                        "step_size": float(step_size_value),
                        "contact_count": len(current_indices),
                        "fingers": list(current_fingers),
                    }
                )

    best_hand_pose_host = np.asarray(optimizer_state.best_hand_pose)
    best_state_vectors_host = _hand_pose_batch_to_packed_state_vectors_np(best_hand_pose_host)
    best_contact_indices_host = np.asarray(optimizer_state.best_contact_indices)
    accepted_steps_host = np.asarray(optimizer_state.accepted_steps)
    rejected_steps_host = np.asarray(optimizer_state.rejected_steps)
    initial_contact_indices_host = np.asarray(initial_contact_indices)
    final_step_index = int(config.optimize.max_steps) - 1
    final_temperature = _scheduled_value(
        float(config.optimize.starting_temperature),
        float(config.optimize.temperature_decay),
        int(config.optimize.annealing_period),
        final_step_index,
    )
    final_step_size = _scheduled_value(
        float(config.optimize.step_size),
        float(config.optimize.temperature_decay),
        int(config.optimize.stepsize_period),
        final_step_index,
    )

    results: list[BatchedSampleResult] = []
    for sample_index in range(batch_size):
        results.append(
            BatchedSampleResult(
                initial_contact_indices=tuple(int(value) for value in initial_contact_indices_host[sample_index].tolist()),
                best_state_vector=np.asarray(best_state_vectors_host[sample_index], dtype=float),
                best_contact_indices=tuple(int(value) for value in best_contact_indices_host[sample_index].tolist()),
                accepted_steps=int(accepted_steps_host[sample_index]),
                rejected_steps=int(rejected_steps_host[sample_index]),
                final_temperature=float(final_temperature),
                final_step_size=float(final_step_size),
                trace=trace_per_sample[sample_index],
                step_stats=step_stats_per_sample[sample_index],
            )
        )
    return results
