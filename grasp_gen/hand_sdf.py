from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from .hand import Hand


SDF_EPS = 1.0e-8


class HandSdfSpec(NamedTuple):
    box_body_indices: jax.Array
    box_local_positions: jax.Array
    box_local_rotations: jax.Array
    box_half_extents: jax.Array
    capsule_body_indices: jax.Array
    capsule_local_positions: jax.Array
    capsule_local_rotations: jax.Array
    capsule_radii: jax.Array
    capsule_half_lengths: jax.Array


def _quat_to_matrix_np(quat: np.ndarray) -> np.ndarray:
    matrix = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=float))
    return matrix.reshape(3, 3)


def _empty_rows(shape: tuple[int, ...]) -> jax.Array:
    return jnp.zeros(shape, dtype=jnp.float32)


def build_hand_sdf_spec(hand: Hand) -> HandSdfSpec:
    box_body_indices: list[int] = []
    box_local_positions: list[np.ndarray] = []
    box_local_rotations: list[np.ndarray] = []
    box_half_extents: list[np.ndarray] = []
    capsule_body_indices: list[int] = []
    capsule_local_positions: list[np.ndarray] = []
    capsule_local_rotations: list[np.ndarray] = []
    capsule_radii: list[float] = []
    capsule_half_lengths: list[float] = []

    prefix = f"collision_hand_{hand.side}_"
    for geom_id in range(hand.model.ngeom):
        geom_name = mujoco.mj_id2name(hand.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not geom_name.startswith(prefix):
            continue

        body_id = int(hand.model.geom_bodyid[geom_id])
        if body_id not in hand._body_id_map:
            raise ValueError(f"Collision geom body {body_id} is outside the compact hand tree.")
        compact_body_id = int(hand._body_id_map[body_id])
        geom_pos = np.asarray(hand.model.geom_pos[geom_id], dtype=np.float32)
        geom_rot = _quat_to_matrix_np(hand.model.geom_quat[geom_id]).astype(np.float32)
        geom_type = int(hand.model.geom_type[geom_id])
        geom_size = np.asarray(hand.model.geom_size[geom_id], dtype=np.float32)

        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            box_body_indices.append(compact_body_id)
            box_local_positions.append(geom_pos)
            box_local_rotations.append(geom_rot)
            box_half_extents.append(np.asarray(geom_size[:3], dtype=np.float32))
            continue
        if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            capsule_body_indices.append(compact_body_id)
            capsule_local_positions.append(geom_pos)
            capsule_local_rotations.append(geom_rot)
            capsule_radii.append(float(geom_size[0]))
            capsule_half_lengths.append(float(geom_size[1]))
            continue
        raise ValueError(f"Unsupported hand collision geom type for SDF: {geom_type}")

    return HandSdfSpec(
        box_body_indices=jnp.asarray(box_body_indices, dtype=jnp.int32),
        box_local_positions=(
            jnp.asarray(np.stack(box_local_positions, axis=0), dtype=jnp.float32)
            if box_local_positions
            else _empty_rows((0, 3))
        ),
        box_local_rotations=(
            jnp.asarray(np.stack(box_local_rotations, axis=0), dtype=jnp.float32)
            if box_local_rotations
            else _empty_rows((0, 3, 3))
        ),
        box_half_extents=(
            jnp.asarray(np.stack(box_half_extents, axis=0), dtype=jnp.float32)
            if box_half_extents
            else _empty_rows((0, 3))
        ),
        capsule_body_indices=jnp.asarray(capsule_body_indices, dtype=jnp.int32),
        capsule_local_positions=(
            jnp.asarray(np.stack(capsule_local_positions, axis=0), dtype=jnp.float32)
            if capsule_local_positions
            else _empty_rows((0, 3))
        ),
        capsule_local_rotations=(
            jnp.asarray(np.stack(capsule_local_rotations, axis=0), dtype=jnp.float32)
            if capsule_local_rotations
            else _empty_rows((0, 3, 3))
        ),
        capsule_radii=(
            jnp.asarray(np.asarray(capsule_radii, dtype=np.float32), dtype=jnp.float32)
            if capsule_radii
            else _empty_rows((0,))
        ),
        capsule_half_lengths=(
            jnp.asarray(np.asarray(capsule_half_lengths, dtype=np.float32), dtype=jnp.float32)
            if capsule_half_lengths
            else _empty_rows((0,))
        ),
    )


def _geom_world_pose(
    body_positions: jax.Array,
    body_rotations: jax.Array,
    body_indices: jax.Array,
    local_positions: jax.Array,
    local_rotations: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    batch_indices = jnp.arange(body_positions.shape[0], dtype=jnp.int32)[:, None]
    selected_body_positions = body_positions[batch_indices, body_indices]
    selected_body_rotations = body_rotations[batch_indices, body_indices]
    world_positions = selected_body_positions + jnp.einsum("bgij,gj->bgi", selected_body_rotations, local_positions)
    world_rotations = jnp.einsum("bgij,gjk->bgik", selected_body_rotations, local_rotations)
    return world_positions, world_rotations


def _points_in_geom_frame(
    points_world: jax.Array,
    geom_world_positions: jax.Array,
    geom_world_rotations: jax.Array,
) -> jax.Array:
    deltas = points_world[:, :, None, :] - geom_world_positions[:, None, :, :]
    return jnp.einsum("bpgi,bgij->bpgj", deltas, geom_world_rotations)


def _box_signed_distance(points_local: jax.Array, half_extents: jax.Array) -> jax.Array:
    q = jnp.abs(points_local) - half_extents[None, None, :, :]
    outside = jnp.linalg.norm(jnp.maximum(q, 0.0), axis=-1)
    inside = jnp.minimum(jnp.max(q, axis=-1), 0.0)
    return outside + inside


def _capsule_signed_distance(points_local: jax.Array, radii: jax.Array, half_lengths: jax.Array) -> jax.Array:
    closest_z = jnp.clip(points_local[..., 2], -half_lengths[None, None, :], half_lengths[None, None, :])
    closest = jnp.stack(
        [
            jnp.zeros_like(closest_z),
            jnp.zeros_like(closest_z),
            closest_z,
        ],
        axis=-1,
    )
    return jnp.linalg.norm(points_local - closest, axis=-1) - radii[None, None, :]


def hand_signed_distance(
    spec: HandSdfSpec,
    body_positions: jax.Array,
    body_rotations: jax.Array,
    points_world: jax.Array,
) -> jax.Array:
    batch_size = int(points_world.shape[0])
    point_count = int(points_world.shape[1])
    sdf = jnp.full((batch_size, point_count), jnp.inf, dtype=jnp.float32)

    if int(spec.box_body_indices.shape[0]) > 0:
        box_world_positions, box_world_rotations = _geom_world_pose(
            body_positions,
            body_rotations,
            spec.box_body_indices,
            spec.box_local_positions,
            spec.box_local_rotations,
        )
        box_points_local = _points_in_geom_frame(points_world, box_world_positions, box_world_rotations)
        sdf = jnp.minimum(sdf, jnp.min(_box_signed_distance(box_points_local, spec.box_half_extents), axis=2))

    if int(spec.capsule_body_indices.shape[0]) > 0:
        capsule_world_positions, capsule_world_rotations = _geom_world_pose(
            body_positions,
            body_rotations,
            spec.capsule_body_indices,
            spec.capsule_local_positions,
            spec.capsule_local_rotations,
        )
        capsule_points_local = _points_in_geom_frame(points_world, capsule_world_positions, capsule_world_rotations)
        sdf = jnp.minimum(
            sdf,
            jnp.min(
                _capsule_signed_distance(
                    capsule_points_local,
                    spec.capsule_radii,
                    spec.capsule_half_lengths,
                ),
                axis=2,
            ),
        )

    return sdf
