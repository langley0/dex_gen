from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from .hand import Hand, InitConfig
from .hand_contacts import ContactConfig
from .prop import Prop
from .grasp_equilibrium import (
    EquilibriumTerms,
    mesh_scale_np,
    simple_terms,
    torque_terms,
    triangle_normals_local_np,
    validate_mode,
    wrench_terms,
    zero_terms,
)
from .grasp_optimizer_state import GraspBatchEnergy


ORTHO6D_EPS = 1.0e-8
TRIANGLE_EPS = 1.0e-12


class HandEnergySpec(NamedTuple):
    parent_indices: np.ndarray
    body_local_pos: jax.Array
    body_local_rot: jax.Array
    joint_qpos_indices: np.ndarray
    joint_axes: jax.Array
    qpos_lower: jax.Array
    qpos_upper: jax.Array
    contact_body_indices: jax.Array
    contact_local_positions: jax.Array


class PropMeshSpec(NamedTuple):
    triangles_local: jax.Array
    triangle_normals_local: jax.Array
    origin_world: jax.Array
    com_world: jax.Array
    rotation_world: jax.Array
    scale: jax.Array


class GraspBatchDiagnostics(NamedTuple):
    energy: GraspBatchEnergy
    nearest_world_positions: jax.Array
    nearest_world_normals: jax.Array
    sum_force: jax.Array
    sum_torque: jax.Array
    contact_weights: jax.Array


@dataclass(frozen=True)
class GraspEnergyConfig:
    distance_weight: float = 1.0
    equilibrium_mode: str = "none"
    equilibrium_weight: float = 1.0
    wrench_iters: int = 24
    root_position_margin: float = 0.35
    root_height_floor: float = 0.03


def _quat_to_matrix_np(quat: np.ndarray) -> np.ndarray:
    matrix = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=float))
    return matrix.reshape(3, 3)


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


def _forward_kinematics_batch(spec: HandEnergySpec, hand_pose: jax.Array) -> tuple[jax.Array, jax.Array]:
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


def _dot(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.sum(a * b, axis=-1)


def _closest_points_on_triangles(points_local: jax.Array, mesh: PropMeshSpec) -> tuple[jax.Array, jax.Array, jax.Array]:
    points = points_local[:, :, None, :]
    triangles = mesh.triangles_local[None, None, :, :, :]
    a = triangles[..., 0, :]
    b = triangles[..., 1, :]
    c = triangles[..., 2, :]

    ab = b - a
    ac = c - a
    bc = c - b
    ap = points - a
    bp = points - b
    cp = points - c

    d1 = _dot(ab, ap)
    d2 = _dot(ac, ap)
    d3 = _dot(ab, bp)
    d4 = _dot(ac, bp)
    d5 = _dot(ab, cp)
    d6 = _dot(ac, cp)

    a_mask = (d1 <= 0.0) & (d2 <= 0.0)
    b_mask = (d3 >= 0.0) & (d4 <= d3)
    c_mask = (d6 >= 0.0) & (d5 <= d6)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    ab_mask = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    ac_mask = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    bc_mask = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)

    ab_v = d1 / jnp.maximum(d1 - d3, TRIANGLE_EPS)
    ac_w = d2 / jnp.maximum(d2 - d6, TRIANGLE_EPS)
    bc_w = (d4 - d3) / jnp.maximum((d4 - d3) + (d5 - d6), TRIANGLE_EPS)
    face_denom = jnp.maximum(va + vb + vc, TRIANGLE_EPS)
    face_v = vb / face_denom
    face_w = vc / face_denom

    closest = a + face_v[..., None] * ab + face_w[..., None] * ac
    closest = jnp.where(a_mask[..., None], a, closest)
    closest = jnp.where(b_mask[..., None], b, closest)
    closest = jnp.where(c_mask[..., None], c, closest)
    closest = jnp.where(ab_mask[..., None], a + ab_v[..., None] * ab, closest)
    closest = jnp.where(ac_mask[..., None], a + ac_w[..., None] * ac, closest)
    closest = jnp.where(bc_mask[..., None], b + bc_w[..., None] * bc, closest)

    distance_sq = _dot(points - closest, points - closest)
    triangle_index = jnp.argmin(distance_sq, axis=2)
    unsigned = jnp.sqrt(jnp.maximum(jnp.take_along_axis(distance_sq, triangle_index[..., None], axis=2)[..., 0], 0.0))
    nearest_points = jnp.take_along_axis(closest, triangle_index[..., None, None], axis=2)[..., 0, :]
    return unsigned, triangle_index, nearest_points


def _hand_qpos_limits(hand: Hand) -> tuple[jax.Array, jax.Array]:
    lower = np.full(hand.model.nq, -np.inf, dtype=np.float32)
    upper = np.full(hand.model.nq, np.inf, dtype=np.float32)
    for actuator in hand.actuators:
        lower[actuator.qpos_index] = float(actuator.lo)
        upper[actuator.qpos_index] = float(actuator.hi)
    return jnp.asarray(lower, dtype=jnp.float32), jnp.asarray(upper, dtype=jnp.float32)


def _extract_hand_spec(hand: Hand, contact_cfg: ContactConfig) -> HandEnergySpec:
    contact_init_cfg = InitConfig(
        n_per_seg=contact_cfg.n_per_seg,
        thumb_weight=contact_cfg.thumb_weight,
        palm_clearance=contact_cfg.palm_clearance,
    )
    contact_batch = hand._contact_batch(contact_init_cfg)
    qpos_lower, qpos_upper = _hand_qpos_limits(hand)
    return HandEnergySpec(
        parent_indices=hand._kin.parent_indices.copy(),
        body_local_pos=hand._kin.body_local_pos,
        body_local_rot=hand._kin.body_local_rot,
        joint_qpos_indices=hand._kin.joint_qpos_indices.copy(),
        joint_axes=hand._kin.joint_axes,
        qpos_lower=qpos_lower,
        qpos_upper=qpos_upper,
        contact_body_indices=contact_batch.body_indices,
        contact_local_positions=contact_batch.local_positions,
    )


def _extract_prop_mesh(prop: Prop) -> PropMeshSpec:
    triangles_local = np.asarray(prop.vertices[prop.faces], dtype=np.float32)
    triangle_normals_local = triangle_normals_local_np(triangles_local)
    rotation_world = _quat_to_matrix_np(prop.quat)
    origin_world = np.asarray(prop.pos, dtype=np.float32)
    com_world = origin_world + rotation_world @ np.asarray(prop.com_local, dtype=np.float32)
    return PropMeshSpec(
        triangles_local=jnp.asarray(triangles_local, dtype=jnp.float32),
        triangle_normals_local=jnp.asarray(triangle_normals_local, dtype=jnp.float32),
        origin_world=jnp.asarray(origin_world, dtype=jnp.float32),
        com_world=jnp.asarray(com_world, dtype=jnp.float32),
        rotation_world=jnp.asarray(rotation_world, dtype=jnp.float32),
        scale=jnp.asarray(mesh_scale_np(prop.vertices), dtype=jnp.float32),
    )


class GraspEnergyModel:
    def __init__(
        self,
        hand: Hand,
        prop: Prop,
        *,
        contact_cfg: ContactConfig | None = None,
        config: GraspEnergyConfig | None = None,
    ):
        self.hand = hand
        self.prop = prop
        self.contact_cfg = ContactConfig() if contact_cfg is None else contact_cfg
        self.config = GraspEnergyConfig() if config is None else config
        self._equilibrium_mode = validate_mode(self.config.equilibrium_mode)
        if self.config.equilibrium_weight < 0.0:
            raise ValueError("equilibrium_weight must be non-negative.")
        if self.config.wrench_iters <= 0:
            raise ValueError("wrench_iters must be positive.")
        self.hand_spec = _extract_hand_spec(hand, self.contact_cfg)
        self.prop_mesh = _extract_prop_mesh(prop)
        self.pose_dim = 9 + int(hand.model.nq)
        self.point_count = int(self.hand_spec.contact_local_positions.shape[0])

    def project(self, hand_pose: jax.Array) -> jax.Array:
        hand_pose = jnp.asarray(hand_pose, dtype=jnp.float32)
        root_pos = hand_pose[:, :3]
        root_rot6d = hand_pose[:, 3:9]
        hand_qpos = hand_pose[:, 9:]

        offset = root_pos - self.prop_mesh.origin_world[None, :]
        offset_norm = jnp.linalg.norm(offset, axis=1, keepdims=True)
        margin = jnp.asarray(self.config.root_position_margin, dtype=jnp.float32)
        clipped_offset = jnp.where(
            offset_norm > margin,
            offset * (margin / jnp.maximum(offset_norm, ORTHO6D_EPS)),
            offset,
        )
        root_pos = self.prop_mesh.origin_world[None, :] + clipped_offset
        root_pos = root_pos.at[:, 2].set(jnp.maximum(root_pos[:, 2], jnp.asarray(self.config.root_height_floor, dtype=jnp.float32)))
        hand_qpos = jnp.clip(hand_qpos, self.hand_spec.qpos_lower[None, :], self.hand_spec.qpos_upper[None, :])
        return jnp.concatenate([root_pos, root_rot6d, hand_qpos], axis=1)

    def _equilibrium_terms(
        self,
        nearest_world_positions: jax.Array,
        nearest_world_normals: jax.Array,
    ) -> EquilibriumTerms:
        batch_size = int(nearest_world_positions.shape[0])
        contact_count = int(nearest_world_positions.shape[1])
        if self._equilibrium_mode == "none":
            return zero_terms(batch_size, contact_count, dtype=nearest_world_positions)
        if self._equilibrium_mode == "torque":
            return torque_terms(
                nearest_world_positions,
                nearest_world_normals,
                self.prop_mesh.com_world,
                jnp.broadcast_to(self.prop_mesh.scale, (batch_size,)),
            )
        if self._equilibrium_mode == "simple":
            return simple_terms(
                nearest_world_positions,
                nearest_world_normals,
                self.prop_mesh.com_world,
                jnp.broadcast_to(self.prop_mesh.scale, (batch_size,)),
            )
        return wrench_terms(
            nearest_world_positions,
            nearest_world_normals,
            self.prop_mesh.com_world,
            jnp.broadcast_to(self.prop_mesh.scale, (batch_size,)),
            iterations=self.config.wrench_iters,
        )

    def diagnostics(self, hand_pose: jax.Array, contact_indices: jax.Array) -> GraspBatchDiagnostics:
        hand_pose = jnp.asarray(hand_pose, dtype=jnp.float32)
        body_positions, body_rotations = _forward_kinematics_batch(self.hand_spec, hand_pose)
        contact_world_positions = _body_local_points_to_world(
            body_positions,
            body_rotations,
            self.hand_spec.contact_body_indices,
            self.hand_spec.contact_local_positions,
        )
        batch_indices = jnp.arange(contact_world_positions.shape[0], dtype=jnp.int32)[:, None]
        selected_world_positions = contact_world_positions[batch_indices, contact_indices]
        selected_local_positions = jnp.einsum(
            "bkj,jm->bkm",
            selected_world_positions - self.prop_mesh.origin_world[None, None, :],
            self.prop_mesh.rotation_world,
        )
        distance_to_surface, triangle_index, nearest_local_positions = _closest_points_on_triangles(
            selected_local_positions,
            self.prop_mesh,
        )
        nearest_world_positions = self.prop_mesh.origin_world[None, None, :] + jnp.einsum(
            "ij,bkj->bki",
            self.prop_mesh.rotation_world,
            nearest_local_positions,
        )
        nearest_local_normals = self.prop_mesh.triangle_normals_local[triangle_index]
        nearest_world_normals = jnp.einsum(
            "ij,bkj->bki",
            self.prop_mesh.rotation_world,
            nearest_local_normals,
        )
        distance_energy = jnp.asarray(self.config.distance_weight, dtype=jnp.float32) * jnp.sum(distance_to_surface, axis=1)
        equilibrium_terms = self._equilibrium_terms(nearest_world_positions, nearest_world_normals)
        equilibrium_energy = jnp.asarray(self.config.equilibrium_weight, dtype=jnp.float32) * equilibrium_terms.energy
        energy = GraspBatchEnergy(
            total=distance_energy + equilibrium_energy,
            distance=distance_energy,
            equilibrium=equilibrium_energy,
            force=equilibrium_terms.force_residual,
            torque=equilibrium_terms.torque_residual,
        )
        return GraspBatchDiagnostics(
            energy=energy,
            nearest_world_positions=nearest_world_positions,
            nearest_world_normals=nearest_world_normals,
            sum_force=equilibrium_terms.sum_force,
            sum_torque=equilibrium_terms.sum_torque,
            contact_weights=equilibrium_terms.contact_weights,
        )

    def energy(self, hand_pose: jax.Array, contact_indices: jax.Array) -> GraspBatchEnergy:
        return self.diagnostics(hand_pose, contact_indices).energy
