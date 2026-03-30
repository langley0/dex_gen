from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from grasp_gen.grasp_energy import (
    GraspEnergyConfig,
    GraspEnergyModel,
    _body_local_points_to_world,
    _closest_points_on_triangles,
    _forward_kinematics_batch,
)
from grasp_gen.grasp_optimizer_io import GraspRunArtifact, load_grasp_run
from grasp_gen.hand import Hand
from grasp_gen.hand_contacts import ContactConfig
from grasp_gen.prop_assets import prop_from_metadata
from grasp_refine.batch import BatchRefineCallbacks
from grasp_refine.refine import CONTACT_LOCAL_EPS, RefineConfig, RefineEnergyTerms, SingleRefineCallbacks, SourceGrasp
from grasp_sampling.scene import build_physics_scene


SUPPORT_GROUP_COUNT = 5


def _contact_config_from_metadata(metadata: dict[str, Any]) -> ContactConfig:
    contact_meta = dict(metadata.get("contact", {}))
    return ContactConfig(
        n_per_seg=int(contact_meta.get("n_per_seg", 10)),
        thumb_weight=float(contact_meta.get("thumb_weight", 1.0)),
        palm_clearance=float(contact_meta.get("palm_clearance", 8.0e-3)),
        target_spacing=float(contact_meta.get("target_spacing", 5.0e-3)),
        cloud_scale=float(contact_meta.get("cloud_scale", 1.00935)),
    )


def _energy_config_from_metadata(metadata: dict[str, Any]) -> GraspEnergyConfig:
    energy_meta = dict(metadata.get("energy", {}))
    return GraspEnergyConfig(
        distance_weight=float(energy_meta.get("distance_weight", 1.0)),
        equilibrium_weight=float(energy_meta.get("equilibrium_weight", 1.0)),
        penetration_weight=float(energy_meta.get("penetration_weight", 100.0)),
        wrench_iters=int(energy_meta.get("wrench_iters", 24)),
        sdf_voxel_size=float(energy_meta.get("sdf_voxel_size", 3.0e-3)),
        sdf_padding=float(energy_meta.get("sdf_padding", 1.0e-2)),
        root_position_margin=float(energy_meta.get("root_position_margin", 0.35)),
        root_height_floor=float(energy_meta.get("root_height_floor", 0.03)),
    )


def _object_local(points_world: jax.Array, energy_model: GraspEnergyModel) -> jax.Array:
    return jnp.einsum(
        "...j,jm->...m",
        points_world - energy_model.prop_mesh.origin_world,
        energy_model.prop_mesh.rotation_world,
    )


def _selected_contact_world_single(hand_pose: jax.Array, contact_indices: jax.Array, energy_model: GraspEnergyModel) -> jax.Array:
    body_positions, body_rotations = _forward_kinematics_batch(energy_model.hand_spec, hand_pose[None, :])
    contact_world_positions = _body_local_points_to_world(
        body_positions,
        body_rotations,
        energy_model.hand_spec.contact_body_indices,
        energy_model.hand_spec.contact_local_positions,
    )
    return contact_world_positions[0, contact_indices]


def _selected_contact_world_batch(hand_pose: jax.Array, contact_indices: jax.Array, energy_model: GraspEnergyModel) -> jax.Array:
    body_positions, body_rotations = _forward_kinematics_batch(energy_model.hand_spec, hand_pose)
    contact_world_positions = _body_local_points_to_world(
        body_positions,
        body_rotations,
        energy_model.hand_spec.contact_body_indices,
        energy_model.hand_spec.contact_local_positions,
    )
    batch_indices = jnp.arange(hand_pose.shape[0], dtype=jnp.int32)[:, None]
    return contact_world_positions[batch_indices, contact_indices]


def _cloud_world_batch(hand_pose: jax.Array, energy_model: GraspEnergyModel) -> jax.Array:
    body_positions, body_rotations = _forward_kinematics_batch(energy_model.hand_spec, hand_pose)
    return _body_local_points_to_world(
        body_positions,
        body_rotations,
        energy_model.hand_spec.cloud_body_indices,
        energy_model.hand_spec.cloud_local_positions,
    )


def _body_local_points_to_world_per_sample(
    body_positions: jax.Array,
    body_rotations: jax.Array,
    body_indices: jax.Array,
    local_positions: jax.Array,
) -> jax.Array:
    batch_indices = jnp.arange(body_positions.shape[0], dtype=jnp.int32)[:, None]
    selected_body_positions = body_positions[batch_indices, body_indices]
    selected_body_rotations = body_rotations[batch_indices, body_indices]
    return selected_body_positions + jnp.einsum("bnij,bnj->bni", selected_body_rotations, local_positions)


def _initial_contact_targets_local_single(initial_hand_pose: np.ndarray, contact_indices: np.ndarray, energy_model: GraspEnergyModel) -> np.ndarray:
    selected_world = _selected_contact_world_single(
        jnp.asarray(initial_hand_pose, dtype=jnp.float32),
        jnp.asarray(contact_indices, dtype=jnp.int32),
        energy_model,
    )[None, :, :]
    selected_local = _object_local(selected_world, energy_model)
    _, _, nearest_local = _closest_points_on_triangles(selected_local, energy_model.prop_mesh)
    return np.asarray(nearest_local[0], dtype=np.float32)


def initial_contact_targets_local_batch(hand_pose: np.ndarray, contact_indices: np.ndarray, energy_model: GraspEnergyModel) -> np.ndarray:
    selected_world = _selected_contact_world_batch(
        jnp.asarray(hand_pose, dtype=jnp.float32),
        jnp.asarray(contact_indices, dtype=jnp.int32),
        energy_model,
    )
    selected_local = _object_local(selected_world, energy_model)
    _, _, nearest_local = _closest_points_on_triangles(selected_local, energy_model.prop_mesh)
    return np.asarray(nearest_local, dtype=np.float32)


def _support_group_bodies(contact_body_indices: np.ndarray, contact_indices: np.ndarray, palm_body_index: int) -> list[int]:
    ordered: list[int] = [int(palm_body_index)]
    for body_index in contact_body_indices[np.asarray(contact_indices, dtype=np.int32)].tolist():
        body_id = int(body_index)
        if body_id not in ordered:
            ordered.append(body_id)
        if len(ordered) >= SUPPORT_GROUP_COUNT:
            break
    return ordered


def _support_patch_single(
    initial_hand_pose: np.ndarray,
    contact_indices: np.ndarray,
    energy_model: GraspEnergyModel,
    *,
    palm_body_index: int,
    config: RefineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cloud_body_indices = np.asarray(energy_model.hand_spec.cloud_body_indices, dtype=np.int32)
    cloud_local_positions = np.asarray(energy_model.hand_spec.cloud_local_positions, dtype=np.float32)
    cloud_world = np.asarray(
        _cloud_world_batch(jnp.asarray(initial_hand_pose[None, :], dtype=jnp.float32), energy_model)[0],
        dtype=np.float32,
    )
    cloud_local_object = np.asarray(_object_local(cloud_world[None, :, :], energy_model)[0], dtype=np.float32)
    unsigned, _, nearest_local = _closest_points_on_triangles(
        jnp.asarray(cloud_local_object[None, :, :], dtype=jnp.float32),
        energy_model.prop_mesh,
    )
    unsigned = np.asarray(unsigned[0], dtype=np.float32)
    nearest_local = np.asarray(nearest_local[0], dtype=np.float32)
    sigma = max(float(config.support_distance_sigma), CONTACT_LOCAL_EPS)
    max_distance = float(config.support_max_distance)
    per_body = max(int(config.support_points_per_body), 1)
    total_points = SUPPORT_GROUP_COUNT * per_body
    support_body = np.full((total_points,), -1, dtype=np.int32)
    support_local = np.zeros((total_points, 3), dtype=np.float32)
    support_target = np.zeros((total_points, 3), dtype=np.float32)
    support_weight = np.zeros((total_points,), dtype=np.float32)

    contact_body_indices = np.asarray(energy_model.hand_spec.contact_body_indices, dtype=np.int32)
    group_bodies = _support_group_bodies(contact_body_indices, contact_indices, int(palm_body_index))
    cursor = 0
    for body_id in group_bodies:
        body_mask = cloud_body_indices == int(body_id)
        if not np.any(body_mask):
            cursor += per_body
            continue
        body_indices = np.flatnonzero(body_mask)
        body_distance = unsigned[body_indices]
        keep_mask = body_distance <= max_distance
        if np.any(keep_mask):
            body_indices = body_indices[keep_mask]
            body_distance = body_distance[keep_mask]
        order = np.argsort(body_distance, kind="stable")
        chosen = body_indices[order[:per_body]]
        chosen_distance = unsigned[chosen]
        count = len(chosen)
        if count > 0:
            span = slice(cursor, cursor + count)
            support_body[span] = cloud_body_indices[chosen]
            support_local[span] = cloud_local_positions[chosen]
            support_target[span] = nearest_local[chosen]
            support_weight[span] = np.exp(-chosen_distance / sigma).astype(np.float32)
        cursor += per_body
    return support_body, support_local, support_target, support_weight


def _support_patch_batch(
    initial_hand_pose: np.ndarray,
    contact_indices: np.ndarray,
    energy_model: GraspEnergyModel,
    *,
    palm_body_index: int,
    config: RefineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch_size = int(initial_hand_pose.shape[0])
    per_body = max(int(config.support_points_per_body), 1)
    total_points = SUPPORT_GROUP_COUNT * per_body
    support_body = np.full((batch_size, total_points), -1, dtype=np.int32)
    support_local = np.zeros((batch_size, total_points, 3), dtype=np.float32)
    support_target = np.zeros((batch_size, total_points, 3), dtype=np.float32)
    support_weight = np.zeros((batch_size, total_points), dtype=np.float32)
    for batch_index in range(batch_size):
        body, local, target, weight = _support_patch_single(
            initial_hand_pose[batch_index],
            contact_indices[batch_index],
            energy_model,
            palm_body_index=palm_body_index,
            config=config,
        )
        support_body[batch_index] = body
        support_local[batch_index] = local
        support_target[batch_index] = target
        support_weight[batch_index] = weight
    return support_body, support_local, support_target, support_weight


def select_source_grasp(artifact: GraspRunArtifact, *, state_name: str, index: int) -> SourceGrasp:
    if state_name == "best":
        hand_pose_batch = np.asarray(artifact.state.best_hand_pose, dtype=np.float32)
        contact_indices_batch = np.asarray(artifact.state.best_contact_indices, dtype=np.int32)
        total = np.asarray(artifact.state.best_energy.total, dtype=np.float32)
    elif state_name == "current":
        hand_pose_batch = np.asarray(artifact.state.hand_pose, dtype=np.float32)
        contact_indices_batch = np.asarray(artifact.state.contact_indices, dtype=np.int32)
        total = np.asarray(artifact.state.energy.total, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported state_name: {state_name!r}")

    sample_index = int(np.argmin(total)) if index < 0 else int(index)
    if not 0 <= sample_index < hand_pose_batch.shape[0]:
        raise ValueError(f"sample index out of range: {sample_index}")

    return SourceGrasp(
        source_path=artifact.path,
        state_name=state_name,
        sample_index=sample_index,
        hand_side=str(artifact.metadata["hand"]["side"]),
        prop_meta=dict(artifact.metadata["prop"]),
        hand_pose=np.asarray(hand_pose_batch[sample_index], dtype=np.float32),
        contact_indices=np.asarray(contact_indices_batch[sample_index], dtype=np.int32),
    )


def select_batch_source(artifact: GraspRunArtifact, *, state_name: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    if state_name == "best":
        hand_pose = np.asarray(artifact.state.best_hand_pose, dtype=np.float32)
        contact_indices = np.asarray(artifact.state.best_contact_indices, dtype=np.int32)
        total = np.asarray(artifact.state.best_energy.total, dtype=np.float32)
    elif state_name == "current":
        hand_pose = np.asarray(artifact.state.hand_pose, dtype=np.float32)
        contact_indices = np.asarray(artifact.state.contact_indices, dtype=np.int32)
        total = np.asarray(artifact.state.energy.total, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported state_name: {state_name!r}")

    metadata = dict(artifact.metadata)
    metadata["source"] = {
        "result_path": str(artifact.path),
        "state": state_name,
    }
    return metadata, hand_pose, contact_indices, total


def build_runtime(metadata: dict[str, Any]) -> tuple[Hand, Any, GraspEnergyModel]:
    hand = Hand(str(metadata["hand"]["side"]))
    prop = prop_from_metadata(dict(metadata["prop"]))
    energy_model = GraspEnergyModel(
        hand,
        prop,
        contact_cfg=_contact_config_from_metadata(metadata),
        config=_energy_config_from_metadata(metadata),
    )
    return hand, prop, energy_model


def _ancestor_joint_masks(energy_model: GraspEnergyModel) -> np.ndarray:
    parent_indices = np.asarray(energy_model.hand_spec.parent_indices, dtype=np.int32)
    joint_qpos_indices = np.asarray(energy_model.hand_spec.joint_qpos_indices, dtype=np.int32)
    qpos_count = int(energy_model.hand_spec.qpos_lower.shape[0])
    body_count = len(parent_indices)
    masks = np.zeros((body_count, qpos_count), dtype=np.float32)
    for body_index in range(body_count):
        current = int(body_index)
        while current >= 0:
            qpos_index = int(joint_qpos_indices[current])
            if qpos_index >= 0:
                masks[body_index, qpos_index] = 1.0
            parent = int(parent_indices[current])
            if current <= 0 or parent == current:
                break
            current = parent
    return masks


def _active_pose_mask_single(
    hand_pose: np.ndarray,
    contact_indices: np.ndarray,
    energy_model: GraspEnergyModel,
    threshold: float,
    *,
    support_body_indices: np.ndarray,
    palm_body_index: int,
) -> np.ndarray:
    diagnostics = energy_model.diagnostics(
        jnp.asarray(hand_pose[None, :], dtype=jnp.float32),
        jnp.asarray(contact_indices[None, :], dtype=jnp.int32),
    )
    depths = np.asarray(diagnostics.penetration_depths[0], dtype=np.float32)
    cloud_body_indices = np.asarray(energy_model.hand_spec.cloud_body_indices, dtype=np.int32)
    penetrated_bodies = cloud_body_indices[depths > float(threshold)]
    support_bodies = np.asarray(support_body_indices, dtype=np.int32)
    support_bodies = support_bodies[support_bodies >= 0]
    bodies = np.unique(np.concatenate([penetrated_bodies, support_bodies], axis=0))

    parent_indices = np.asarray(energy_model.hand_spec.parent_indices, dtype=np.int32)
    joint_qpos_indices = np.asarray(energy_model.hand_spec.joint_qpos_indices, dtype=np.int32)
    mask = np.zeros((energy_model.pose_dim,), dtype=np.float32)
    for body_id in bodies.tolist():
        current = int(body_id)
        while current >= 0:
            qpos_index = int(joint_qpos_indices[current])
            if qpos_index >= 0:
                pose_index = 9 + qpos_index
                if 0 <= pose_index < len(mask):
                    mask[pose_index] = 1.0
            parent = int(parent_indices[current])
            if parent == current:
                break
            current = parent if current > 0 else -1
    if np.any(penetrated_bodies == int(palm_body_index)):
        mask[:3] = 1.0
        mask[3:9] = 1.0
    elif len(penetrated_bodies) == 0 and np.count_nonzero(mask[9:]) == 0 and len(support_bodies) > 0:
        mask[:3] = 1.0
    return mask


def _active_pose_masks_batch(
    hand_pose: np.ndarray,
    contact_indices: np.ndarray,
    energy_model: GraspEnergyModel,
    threshold: float,
    ancestor_joint_masks: np.ndarray,
    *,
    support_body_indices: np.ndarray,
    palm_body_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    diagnostics = energy_model.diagnostics(
        jnp.asarray(hand_pose, dtype=jnp.float32),
        jnp.asarray(contact_indices, dtype=jnp.int32),
    )
    depths = np.asarray(diagnostics.penetration_depths, dtype=np.float32)
    cloud_body_indices = np.asarray(energy_model.hand_spec.cloud_body_indices, dtype=np.int32)
    batch_size = hand_pose.shape[0]
    masks = np.zeros_like(hand_pose, dtype=np.float32)
    active_joint_count = np.zeros((batch_size,), dtype=np.int32)
    for batch_index in range(batch_size):
        penetrated_bodies = cloud_body_indices[depths[batch_index] > float(threshold)]
        support_bodies = np.asarray(support_body_indices[batch_index], dtype=np.int32)
        support_bodies = support_bodies[support_bodies >= 0]
        bodies = np.unique(np.concatenate([penetrated_bodies, support_bodies], axis=0))
        if len(bodies) == 0:
            continue
        joint_mask = np.max(ancestor_joint_masks[bodies], axis=0)
        masks[batch_index, 9:] = joint_mask
        active_joint_count[batch_index] = int(np.count_nonzero(joint_mask))
        if np.any(penetrated_bodies == int(palm_body_index)):
            masks[batch_index, :3] = 1.0
            masks[batch_index, 3:9] = 1.0
        elif len(penetrated_bodies) == 0 and active_joint_count[batch_index] == 0 and len(support_bodies) > 0:
            masks[batch_index, :3] = 1.0
    return masks, active_joint_count


def _ortho6d_to_matrix_np(ortho6d: np.ndarray) -> np.ndarray:
    ortho6d = np.asarray(ortho6d, dtype=np.float64).reshape(6)
    first = ortho6d[:3]
    first /= max(float(np.linalg.norm(first)), CONTACT_LOCAL_EPS)
    second = ortho6d[3:6] - first * float(np.dot(first, ortho6d[3:6]))
    second /= max(float(np.linalg.norm(second)), CONTACT_LOCAL_EPS)
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=1)


def _matrix_to_quat_np(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [0.25 * s, (rotation[2, 1] - rotation[1, 2]) / s, (rotation[0, 2] - rotation[2, 0]) / s, (rotation[1, 0] - rotation[0, 1]) / s],
            dtype=np.float64,
        )
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        quat = np.array(
            [(rotation[2, 1] - rotation[1, 2]) / s, 0.25 * s, (rotation[0, 1] + rotation[1, 0]) / s, (rotation[0, 2] + rotation[2, 0]) / s],
            dtype=np.float64,
        )
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        quat = np.array(
            [(rotation[0, 2] - rotation[2, 0]) / s, (rotation[0, 1] + rotation[1, 0]) / s, 0.25 * s, (rotation[1, 2] + rotation[2, 1]) / s],
            dtype=np.float64,
        )
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        quat = np.array(
            [(rotation[1, 0] - rotation[0, 1]) / s, (rotation[0, 2] + rotation[2, 0]) / s, (rotation[1, 2] + rotation[2, 1]) / s, 0.25 * s],
            dtype=np.float64,
        )
    quat /= max(float(np.linalg.norm(quat)), CONTACT_LOCAL_EPS)
    if quat[0] < 0.0:
        quat *= -1.0
    return quat.astype(np.float32)


def _pose_root(hand_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    root_pos = np.asarray(hand_pose[:3], dtype=np.float64)
    root_quat = _matrix_to_quat_np(_ortho6d_to_matrix_np(np.asarray(hand_pose[3:9], dtype=np.float64)))
    return root_pos, root_quat


def _actual_overlap_counts(source: SourceGrasp, hand_pose: np.ndarray, *, density: float) -> tuple[int, int, float, float]:
    hand, prop, _ = build_runtime({"hand": {"side": source.hand_side}, "prop": source.prop_meta})
    root_pos, root_quat = _pose_root(hand_pose)
    qpos = np.asarray(hand_pose[9:], dtype=np.float64)
    scene = build_physics_scene(
        hand,
        prop,
        source.prop_meta,
        root_pos=root_pos,
        root_quat=root_quat,
        qpos_target=qpos,
        timestep=0.005,
        density=density,
    )
    scene.reset(qpos_target=qpos)
    return scene.contact_counts()


def _actual_overlap_batch(metadata: dict[str, Any], hand_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hand, prop, _ = build_runtime(metadata)
    root_pos, root_quat = _pose_root(hand_pose[0])
    scene = build_physics_scene(
        hand,
        prop,
        dict(metadata["prop"]),
        root_pos=root_pos,
        root_quat=root_quat,
        qpos_target=np.asarray(hand_pose[0, 9:], dtype=np.float64),
        timestep=0.005,
        density=400.0,
    )
    contact_count = np.zeros((hand_pose.shape[0],), dtype=np.int32)
    penetration_count = np.zeros((hand_pose.shape[0],), dtype=np.int32)
    depth_sum = np.zeros((hand_pose.shape[0],), dtype=np.float32)
    max_depth = np.zeros((hand_pose.shape[0],), dtype=np.float32)
    for batch_index in range(hand_pose.shape[0]):
        root_pos, root_quat = _pose_root(hand_pose[batch_index])
        qpos = np.asarray(hand_pose[batch_index, 9:], dtype=np.float64)
        scene.base_root_pos = np.asarray(root_pos, dtype=np.float64)
        scene.base_root_quat = np.asarray(root_quat, dtype=np.float64)
        scene.base_qpos_target = np.asarray(qpos, dtype=np.float64)
        scene.reset(qpos_target=qpos)
        counts = scene.contact_counts()
        contact_count[batch_index] = int(counts[0])
        penetration_count[batch_index] = int(counts[1])
        depth_sum[batch_index] = float(counts[2])
        max_depth[batch_index] = float(counts[3])
    return contact_count, penetration_count, depth_sum, max_depth


def make_single_callbacks(source: SourceGrasp, *, metadata: dict[str, Any], config: RefineConfig) -> tuple[np.ndarray, SingleRefineCallbacks]:
    hand, _, energy_model = build_runtime(metadata)
    contact_indices = np.asarray(source.contact_indices, dtype=np.int32)
    contact_target_local = _initial_contact_targets_local_single(np.asarray(source.hand_pose, dtype=np.float32), contact_indices, energy_model)
    support_body_indices, support_local_positions, support_target_local, support_weights = _support_patch_single(
        np.asarray(source.hand_pose, dtype=np.float32),
        contact_indices,
        energy_model,
        palm_body_index=int(hand._kin.palm_body_index),
        config=config,
    )
    support_body_indices_jax = jnp.asarray(np.maximum(support_body_indices, 0), dtype=jnp.int32)
    support_local_positions_jax = jnp.asarray(support_local_positions, dtype=jnp.float32)
    support_target_local_jax = jnp.asarray(support_target_local, dtype=jnp.float32)
    support_weights_jax = jnp.asarray(support_weights, dtype=jnp.float32)

    def energy_terms(hand_pose: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        projected = energy_model.project(hand_pose[None, :])[0]
        diagnostics = energy_model.diagnostics(projected[None, :], jnp.asarray(contact_indices[None, :], dtype=jnp.int32))
        body_positions, body_rotations = _forward_kinematics_batch(energy_model.hand_spec, projected[None, :])
        selected_world_positions = _body_local_points_to_world(
            body_positions,
            body_rotations,
            energy_model.hand_spec.contact_body_indices[jnp.asarray(contact_indices, dtype=jnp.int32)],
            energy_model.hand_spec.contact_local_positions[jnp.asarray(contact_indices, dtype=jnp.int32)],
        )[0]
        selected_local_positions = _object_local(selected_world_positions, energy_model)
        target = jnp.asarray(contact_target_local, dtype=jnp.float32)
        contact_error = jnp.sum(jnp.square(selected_local_positions - target), axis=-1)
        contact_term = jnp.mean(contact_error)
        support_world_positions = _body_local_points_to_world(
            body_positions,
            body_rotations,
            support_body_indices_jax,
            support_local_positions_jax,
        )[0]
        support_local_current = _object_local(support_world_positions, energy_model)
        support_error = jnp.sum(jnp.square(support_local_current - support_target_local_jax), axis=-1)
        support_term = jnp.sum(support_weights_jax * support_error) / jnp.maximum(jnp.sum(support_weights_jax), CONTACT_LOCAL_EPS)
        penetration_term = jnp.mean(jnp.square(diagnostics.penetration_depths[0]))
        initial = jnp.asarray(source.hand_pose, dtype=jnp.float32)
        root_reg = jnp.mean(jnp.square(projected[:9] - initial[:9]))
        joint_reg = jnp.mean(jnp.square(projected[9:] - initial[9:]))
        distance = diagnostics.energy.distance[0]
        equilibrium = diagnostics.energy.equilibrium[0]
        total = (
            jnp.asarray(config.distance_weight, dtype=jnp.float32) * distance
            + jnp.asarray(config.equilibrium_weight, dtype=jnp.float32) * equilibrium
            + jnp.asarray(config.penetration_weight, dtype=jnp.float32) * penetration_term
            + jnp.asarray(config.contact_weight, dtype=jnp.float32) * contact_term
            + jnp.asarray(config.support_weight, dtype=jnp.float32) * support_term
            + jnp.asarray(config.root_reg_weight, dtype=jnp.float32) * root_reg
            + jnp.asarray(config.joint_reg_weight, dtype=jnp.float32) * joint_reg
        )
        return total, (distance, equilibrium, penetration_term, contact_term + support_term, root_reg, joint_reg, projected)

    grad_fn = jax.jit(jax.value_and_grad(energy_terms, argnums=0, has_aux=True))

    def evaluate_terms_with_grad(hand_pose: np.ndarray, contact_indices_: np.ndarray, contact_target_local_: np.ndarray, cfg: RefineConfig):
        ((total, aux), grad) = grad_fn(jnp.asarray(hand_pose, dtype=jnp.float32))
        projected = np.asarray(aux[-1], dtype=np.float32)
        terms = RefineEnergyTerms(
            total=float(total),
            distance=float(aux[0]),
            equilibrium=float(aux[1]),
            penetration=float(aux[2]),
            contact=float(aux[3]),
            root_reg=float(aux[4]),
            joint_reg=float(aux[5]),
        )
        return terms, projected, np.asarray(grad, dtype=np.float32)

    callbacks = SingleRefineCallbacks(
        evaluate_terms_with_grad=evaluate_terms_with_grad,
        active_pose_mask=lambda hand_pose, contact_indices_, threshold: _active_pose_mask_single(
            hand_pose,
            contact_indices_,
            energy_model,
            threshold,
            support_body_indices=support_body_indices,
            palm_body_index=int(hand._kin.palm_body_index),
        ),
        actual_overlap_counts=lambda hand_pose: _actual_overlap_counts(source, hand_pose, density=config.actual_object_density),
    )
    return contact_target_local, callbacks


def make_batch_callbacks(
    *,
    metadata: dict[str, Any],
    initial_hand_pose: np.ndarray,
    contact_indices: np.ndarray,
    config: RefineConfig,
) -> tuple[np.ndarray, BatchRefineCallbacks]:
    hand, _, energy_model = build_runtime(metadata)
    contact_target_local = initial_contact_targets_local_batch(initial_hand_pose, contact_indices, energy_model)
    support_body_indices, support_local_positions, support_target_local, support_weights = _support_patch_batch(
        initial_hand_pose,
        contact_indices,
        energy_model,
        palm_body_index=int(hand._kin.palm_body_index),
        config=config,
    )
    ancestor_joint_masks = _ancestor_joint_masks(energy_model)
    initial_hand_pose_jax = jnp.asarray(initial_hand_pose, dtype=jnp.float32)
    contact_indices_jax = jnp.asarray(contact_indices, dtype=jnp.int32)
    support_body_indices_jax = jnp.asarray(np.maximum(support_body_indices, 0), dtype=jnp.int32)
    support_local_positions_jax = jnp.asarray(support_local_positions, dtype=jnp.float32)
    support_target_local_jax = jnp.asarray(support_target_local, dtype=jnp.float32)
    support_weights_jax = jnp.asarray(support_weights, dtype=jnp.float32)

    def energy_fn(hand_pose: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        projected = energy_model.project(hand_pose)
        diagnostics = energy_model.diagnostics(projected, contact_indices_jax)
        body_positions, body_rotations = _forward_kinematics_batch(energy_model.hand_spec, projected)
        selected_world = _body_local_points_to_world_per_sample(
            body_positions,
            body_rotations,
            energy_model.hand_spec.contact_body_indices[contact_indices_jax],
            energy_model.hand_spec.contact_local_positions[contact_indices_jax],
        )
        selected_local = _object_local(selected_world, energy_model)
        target = jnp.asarray(contact_target_local, dtype=jnp.float32)
        contact_term = jnp.mean(jnp.sum(jnp.square(selected_local - target), axis=-1), axis=1)
        support_world = _body_local_points_to_world_per_sample(
            body_positions,
            body_rotations,
            support_body_indices_jax,
            support_local_positions_jax,
        )
        support_local_current = _object_local(support_world, energy_model)
        support_error = jnp.sum(jnp.square(support_local_current - support_target_local_jax), axis=-1)
        support_term = jnp.sum(support_weights_jax * support_error, axis=1) / jnp.maximum(
            jnp.sum(support_weights_jax, axis=1),
            CONTACT_LOCAL_EPS,
        )
        penetration_term = jnp.mean(jnp.square(diagnostics.penetration_depths), axis=1)
        root_reg = jnp.mean(jnp.square(projected[:, :9] - initial_hand_pose_jax[:, :9]), axis=1)
        joint_reg = jnp.mean(jnp.square(projected[:, 9:] - initial_hand_pose_jax[:, 9:]), axis=1)
        distance = diagnostics.energy.distance
        equilibrium = diagnostics.energy.equilibrium
        total = (
            jnp.asarray(config.distance_weight, dtype=jnp.float32) * distance
            + jnp.asarray(config.equilibrium_weight, dtype=jnp.float32) * equilibrium
            + jnp.asarray(config.penetration_weight, dtype=jnp.float32) * penetration_term
            + jnp.asarray(config.contact_weight, dtype=jnp.float32) * contact_term
            + jnp.asarray(config.support_weight, dtype=jnp.float32) * support_term
            + jnp.asarray(config.root_reg_weight, dtype=jnp.float32) * root_reg
            + jnp.asarray(config.joint_reg_weight, dtype=jnp.float32) * joint_reg
        )
        return jnp.sum(total), (
            total,
            distance,
            equilibrium,
            penetration_term,
            contact_term + support_term,
            root_reg,
            joint_reg,
            projected,
        )

    grad_fn = jax.jit(jax.value_and_grad(energy_fn, argnums=0, has_aux=True))

    def evaluate_terms_with_grad(hand_pose: np.ndarray, contact_indices_: np.ndarray, contact_target_local_: np.ndarray, cfg: RefineConfig):
        ((_, aux), grad) = grad_fn(jnp.asarray(hand_pose, dtype=jnp.float32))
        projected = np.asarray(aux[-1], dtype=np.float32)
        terms = {
            "total": np.asarray(aux[0], dtype=np.float32),
            "distance": np.asarray(aux[1], dtype=np.float32),
            "equilibrium": np.asarray(aux[2], dtype=np.float32),
            "penetration": np.asarray(aux[3], dtype=np.float32),
            "contact": np.asarray(aux[4], dtype=np.float32),
            "root_reg": np.asarray(aux[5], dtype=np.float32),
            "joint_reg": np.asarray(aux[6], dtype=np.float32),
        }
        return terms, projected, np.asarray(grad, dtype=np.float32)

    callbacks = BatchRefineCallbacks(
        evaluate_terms_with_grad=evaluate_terms_with_grad,
        active_pose_masks=lambda hand_pose, contact_indices_, threshold: _active_pose_masks_batch(
            hand_pose,
            contact_indices_,
            energy_model,
            threshold,
            ancestor_joint_masks,
            support_body_indices=support_body_indices,
            palm_body_index=int(hand._kin.palm_body_index),
        ),
        actual_overlap_batch=lambda hand_pose: _actual_overlap_batch(metadata, hand_pose),
    )
    return contact_target_local, callbacks


def load_source_artifact(path: str | Path) -> GraspRunArtifact:
    return load_grasp_run(path)
