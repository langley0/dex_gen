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


def _dga_step_scales(energy_model: GraspEnergyModel) -> np.ndarray:
    scales = np.ones((energy_model.pose_dim,), dtype=np.float32)
    scales[:3] = max(float(energy_model.config.root_position_margin), CONTACT_LOCAL_EPS)
    qpos_lower = np.asarray(energy_model.hand_spec.qpos_lower, dtype=np.float32)
    qpos_upper = np.asarray(energy_model.hand_spec.qpos_upper, dtype=np.float32)
    scales[9:] = np.maximum(0.5 * (qpos_upper - qpos_lower), CONTACT_LOCAL_EPS)
    return scales


def _self_repulsion_points(energy_model: GraspEnergyModel) -> tuple[np.ndarray, np.ndarray]:
    contact_bodies = np.unique(np.asarray(energy_model.hand_spec.contact_body_indices, dtype=np.int32))
    cloud_body_indices = np.asarray(energy_model.hand_spec.cloud_body_indices, dtype=np.int32)
    cloud_local_positions = np.asarray(energy_model.hand_spec.cloud_local_positions, dtype=np.float32)

    self_body_indices: list[int] = []
    self_local_positions: list[np.ndarray] = []
    for body_index in contact_bodies.tolist():
        body_points = cloud_local_positions[cloud_body_indices == int(body_index)]
        if body_points.size == 0:
            continue
        self_body_indices.append(int(body_index))
        self_local_positions.append(np.mean(body_points, axis=0, dtype=np.float32))
    if not self_body_indices:
        raise ValueError("Unable to construct self-repulsion points from hand surface cloud.")
    return np.asarray(self_body_indices, dtype=np.int32), np.asarray(self_local_positions, dtype=np.float32)


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


def _actual_overlap_batch(
    metadata: dict[str, Any],
    hand_pose: np.ndarray,
    *,
    density: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        density=density,
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
    _, _, energy_model = build_runtime(metadata)
    contact_indices = np.asarray(source.contact_indices, dtype=np.int32)
    contact_target_local = _initial_contact_targets_local_single(np.asarray(source.hand_pose, dtype=np.float32), contact_indices, energy_model)

    contact_indices_jax = jnp.asarray(contact_indices, dtype=jnp.int32)
    cloud_body_indices = energy_model.hand_spec.cloud_body_indices
    cloud_local_positions = energy_model.hand_spec.cloud_local_positions
    self_body_indices_np, self_local_positions_np = _self_repulsion_points(energy_model)
    self_body_indices = jnp.asarray(self_body_indices_np, dtype=jnp.int32)
    self_local_positions = jnp.asarray(self_local_positions_np, dtype=jnp.float32)

    surface_pull_threshold = jnp.asarray(float(config.surface_pull_threshold), dtype=jnp.float32)
    self_repulsion_threshold = jnp.asarray(float(config.self_repulsion_threshold), dtype=jnp.float32)
    surface_pull_weight = jnp.asarray(float(config.surface_pull_weight), dtype=jnp.float32)
    external_repulsion_weight = jnp.asarray(float(config.external_repulsion_weight), dtype=jnp.float32)
    self_repulsion_weight = jnp.asarray(float(config.self_repulsion_weight), dtype=jnp.float32)
    step_scales = _dga_step_scales(energy_model)

    def energy_terms(hand_pose: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        projected = energy_model.project(hand_pose[None, :])[0]
        body_positions, body_rotations = _forward_kinematics_batch(energy_model.hand_spec, projected[None, :])

        contact_world = _selected_contact_world_single(projected, contact_indices_jax, energy_model)
        contact_local = _object_local(contact_world[None, :, :], energy_model)
        surface_distance, _, _ = _closest_points_on_triangles(contact_local, energy_model.prop_mesh)
        surface_distance = surface_distance[0]
        surface_close_mask = surface_distance <= surface_pull_threshold
        surface_pull = jnp.sum(jnp.where(surface_close_mask, surface_distance, 0.0)) / (
            jnp.sum(surface_close_mask.astype(jnp.float32)) + CONTACT_LOCAL_EPS
        )

        cloud_world = _body_local_points_to_world(
            body_positions,
            body_rotations,
            cloud_body_indices,
            cloud_local_positions,
        )[0]
        cloud_local = _object_local(cloud_world[None, :, :], energy_model)
        external_unsigned, triangle_index, nearest_local = _closest_points_on_triangles(cloud_local, energy_model.prop_mesh)
        external_unsigned = external_unsigned[0]
        triangle_index = triangle_index[0]
        nearest_local = nearest_local[0]
        nearest_normals = energy_model.prop_mesh.triangle_normals_local[triangle_index]
        signed_distance = jnp.sum((nearest_local - cloud_local[0]) * nearest_normals, axis=-1)
        external_repulsion = jnp.max(jnp.where(signed_distance > 0.0, external_unsigned, 0.0))

        self_world = _body_local_points_to_world(
            body_positions,
            body_rotations,
            self_body_indices,
            self_local_positions,
        )[0]
        pairwise = jnp.linalg.norm(self_world[:, None, :] - self_world[None, :, :] + 1.0e-13, axis=-1)
        pairwise = jnp.where(jnp.eye(self_world.shape[0], dtype=bool), jnp.inf, pairwise)
        self_repulsion = jnp.sum(jnp.maximum(self_repulsion_threshold - pairwise, 0.0))

        total = (
            surface_pull_weight * surface_pull
            + external_repulsion_weight * external_repulsion
            + self_repulsion_weight * self_repulsion
        )
        return total, (surface_pull, external_repulsion, self_repulsion, projected)

    grad_fn = jax.jit(jax.value_and_grad(energy_terms, argnums=0, has_aux=True))

    def evaluate_terms_with_grad(hand_pose: np.ndarray, contact_indices_: np.ndarray, contact_target_local_: np.ndarray, cfg: RefineConfig):
        del contact_indices_, contact_target_local_, cfg
        ((total, aux), grad) = grad_fn(jnp.asarray(hand_pose, dtype=jnp.float32))
        projected = np.asarray(aux[-1], dtype=np.float32)
        terms = RefineEnergyTerms(
            total=float(total),
            distance=float(aux[0]),
            equilibrium=0.0,
            penetration=float(aux[1]),
            contact=float(aux[2]),
            root_reg=0.0,
            joint_reg=0.0,
        )
        return terms, projected, np.asarray(grad, dtype=np.float32)

    callbacks = SingleRefineCallbacks(
        evaluate_terms_with_grad=evaluate_terms_with_grad,
        sample_scales=lambda: step_scales,
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
    _, _, energy_model = build_runtime(metadata)
    contact_target_local = initial_contact_targets_local_batch(initial_hand_pose, contact_indices, energy_model)

    contact_indices_jax = jnp.asarray(contact_indices, dtype=jnp.int32)
    cloud_body_indices = energy_model.hand_spec.cloud_body_indices
    cloud_local_positions = energy_model.hand_spec.cloud_local_positions
    self_body_indices_np, self_local_positions_np = _self_repulsion_points(energy_model)
    self_body_indices = jnp.asarray(self_body_indices_np, dtype=jnp.int32)
    self_local_positions = jnp.asarray(self_local_positions_np, dtype=jnp.float32)

    surface_pull_threshold = jnp.asarray(float(config.surface_pull_threshold), dtype=jnp.float32)
    self_repulsion_threshold = jnp.asarray(float(config.self_repulsion_threshold), dtype=jnp.float32)
    surface_pull_weight = jnp.asarray(float(config.surface_pull_weight), dtype=jnp.float32)
    external_repulsion_weight = jnp.asarray(float(config.external_repulsion_weight), dtype=jnp.float32)
    self_repulsion_weight = jnp.asarray(float(config.self_repulsion_weight), dtype=jnp.float32)
    step_scales = _dga_step_scales(energy_model)

    def single_energy_fn(hand_pose: jax.Array, sample_contact_indices: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        projected = energy_model.project(hand_pose[None, :])[0]
        body_positions, body_rotations = _forward_kinematics_batch(energy_model.hand_spec, projected[None, :])

        contact_world = _selected_contact_world_single(projected, sample_contact_indices, energy_model)
        contact_local = _object_local(contact_world[None, :, :], energy_model)
        surface_distance, _, _ = _closest_points_on_triangles(contact_local, energy_model.prop_mesh)
        surface_distance = surface_distance[0]
        surface_close_mask = surface_distance <= surface_pull_threshold
        surface_pull = jnp.sum(jnp.where(surface_close_mask, surface_distance, 0.0)) / (
            jnp.sum(surface_close_mask.astype(jnp.float32)) + CONTACT_LOCAL_EPS
        )

        cloud_world = _body_local_points_to_world(
            body_positions,
            body_rotations,
            cloud_body_indices,
            cloud_local_positions,
        )[0]
        cloud_local = _object_local(cloud_world[None, :, :], energy_model)
        external_unsigned, triangle_index, nearest_local = _closest_points_on_triangles(cloud_local, energy_model.prop_mesh)
        external_unsigned = external_unsigned[0]
        triangle_index = triangle_index[0]
        nearest_local = nearest_local[0]
        nearest_normals = energy_model.prop_mesh.triangle_normals_local[triangle_index]
        signed_distance = jnp.sum((nearest_local - cloud_local[0]) * nearest_normals, axis=-1)
        external_repulsion = jnp.max(jnp.where(signed_distance > 0.0, external_unsigned, 0.0))

        self_world = _body_local_points_to_world(
            body_positions,
            body_rotations,
            self_body_indices,
            self_local_positions,
        )[0]
        pairwise = jnp.linalg.norm(self_world[:, None, :] - self_world[None, :, :] + 1.0e-13, axis=-1)
        pairwise = jnp.where(jnp.eye(self_world.shape[0], dtype=bool), jnp.inf, pairwise)
        self_repulsion = jnp.sum(jnp.maximum(self_repulsion_threshold - pairwise, 0.0))

        total = (
            surface_pull_weight * surface_pull
            + external_repulsion_weight * external_repulsion
            + self_repulsion_weight * self_repulsion
        )
        return total, (surface_pull, external_repulsion, self_repulsion, projected)

    grad_fn = jax.jit(jax.value_and_grad(single_energy_fn, argnums=0, has_aux=True))

    def evaluate_terms_with_grad(hand_pose: np.ndarray, contact_indices_: np.ndarray, contact_target_local_: np.ndarray, cfg: RefineConfig):
        del contact_target_local_, cfg
        batch_size = int(hand_pose.shape[0])
        total = np.zeros((batch_size,), dtype=np.float32)
        distance = np.zeros((batch_size,), dtype=np.float32)
        penetration = np.zeros((batch_size,), dtype=np.float32)
        contact = np.zeros((batch_size,), dtype=np.float32)
        projected = np.zeros_like(hand_pose, dtype=np.float32)
        grad = np.zeros_like(hand_pose, dtype=np.float32)
        for batch_index in range(batch_size):
            ((sample_total, aux), sample_grad) = grad_fn(
                jnp.asarray(hand_pose[batch_index], dtype=jnp.float32),
                jnp.asarray(contact_indices_[batch_index], dtype=jnp.int32),
            )
            total[batch_index] = float(sample_total)
            distance[batch_index] = float(aux[0])
            penetration[batch_index] = float(aux[1])
            contact[batch_index] = float(aux[2])
            projected[batch_index] = np.asarray(aux[3], dtype=np.float32)
            grad[batch_index] = np.asarray(sample_grad, dtype=np.float32)
        terms = {
            "total": total,
            "distance": distance,
            "equilibrium": np.zeros_like(total),
            "penetration": penetration,
            "contact": contact,
            "root_reg": np.zeros_like(total),
            "joint_reg": np.zeros_like(total),
        }
        return terms, projected, grad

    callbacks = BatchRefineCallbacks(
        evaluate_terms_with_grad=evaluate_terms_with_grad,
        sample_scales=lambda: step_scales,
        actual_overlap_batch=lambda hand_pose: _actual_overlap_batch(
            metadata,
            hand_pose,
            density=config.actual_object_density,
        ),
    )
    return contact_target_local, callbacks


def load_source_artifact(path: str | Path) -> GraspRunArtifact:
    return load_grasp_run(path)
