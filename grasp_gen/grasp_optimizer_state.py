from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class GraspBatchEnergy(NamedTuple):
    total: jax.Array
    distance: jax.Array
    penetration: jax.Array
    penetration_depth: jax.Array
    selected_penetration: jax.Array

    @classmethod
    def from_total(cls, total: jax.Array) -> "GraspBatchEnergy":
        total = jnp.asarray(total, dtype=jnp.float32)
        zeros = jnp.zeros_like(total)
        flags = jnp.zeros(total.shape, dtype=bool)
        return cls(
            total=total,
            distance=zeros,
            penetration=zeros,
            penetration_depth=zeros,
            selected_penetration=flags,
        )

    @property
    def batch_size(self) -> int:
        return int(self.total.shape[0])


class GraspBatchState(NamedTuple):
    hand_pose: jax.Array
    contact_indices: jax.Array
    energy: GraspBatchEnergy
    best_hand_pose: jax.Array
    best_contact_indices: jax.Array
    best_energy: GraspBatchEnergy
    ema_grad: jax.Array
    accepted_steps: jax.Array
    rejected_steps: jax.Array
    step_index: jax.Array
    rng_key: jax.Array

    @property
    def batch_size(self) -> int:
        return int(self.hand_pose.shape[0])

    @property
    def pose_dim(self) -> int:
        return int(self.hand_pose.shape[1])

    @property
    def contact_count(self) -> int:
        return int(self.contact_indices.shape[1])


class GraspBatchSnapshot(NamedTuple):
    step_index: jax.Array
    hand_pose: jax.Array
    contact_indices: jax.Array
    energy: GraspBatchEnergy
    accepted: jax.Array
    temperature: jax.Array
    step_size: jax.Array

