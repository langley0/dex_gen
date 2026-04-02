from __future__ import annotations

import jax
import jax.numpy as jnp

from .inspire_hand import InspireHandSpec


def _weighted_mean(values: jax.Array, sample_weight: jax.Array | None = None) -> jax.Array:
    if sample_weight is None:
        return jnp.mean(values)
    weights = jnp.asarray(sample_weight, dtype=values.dtype)
    denom = jnp.maximum(jnp.sum(weights), jnp.asarray(1.0, dtype=values.dtype))
    return jnp.sum(values * weights) / denom


def joint_limit_loss(
    denormalized_pose: jax.Array,
    hand_spec: InspireHandSpec,
    sample_weight: jax.Array | None = None,
) -> jax.Array:
    joint_lower = jnp.asarray(hand_spec.joint_lower, dtype=denormalized_pose.dtype)
    joint_upper = jnp.asarray(hand_spec.joint_upper, dtype=denormalized_pose.dtype)
    joints = denormalized_pose[..., 9:]
    below = jnp.maximum(joint_lower - joints, 0.0)
    above = jnp.maximum(joints - joint_upper, 0.0)
    per_sample = jnp.mean(below + above, axis=-1)
    return _weighted_mean(per_sample, sample_weight)


def root_distance_loss(
    denormalized_pose: jax.Array,
    object_points: jax.Array,
    threshold: float,
    sample_weight: jax.Array | None = None,
) -> jax.Array:
    root = denormalized_pose[..., :3]
    diff = root[:, None, :] - object_points
    distances = jnp.linalg.norm(diff, axis=-1)
    nearest = jnp.min(distances, axis=1)
    per_sample = jnp.maximum(nearest - float(threshold), 0.0)
    return _weighted_mean(per_sample, sample_weight)
