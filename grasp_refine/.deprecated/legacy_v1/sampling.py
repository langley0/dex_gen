from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .diffusion import DiffusionSchedule, predict_x0
from .model_factory import apply_model
from .normalization import PoseNormalizer
from .types import DiffusionConfig, ModelConfig


@dataclass(frozen=True)
class SampleBatch:
    normalized_pose: jax.Array
    pose: jax.Array


def _broadcast_scene_tensor(values: jax.Array, batch_size: int) -> jax.Array:
    if values.ndim == 3:
        if values.shape[0] != batch_size:
            raise ValueError(f"Expected scene batch size {batch_size}, got {values.shape[0]}.")
        return values
    if values.ndim != 2:
        raise ValueError(f"Expected scene tensor rank 2 or 3, got shape {values.shape}.")
    return jnp.broadcast_to(values[None, ...], (batch_size, values.shape[0], values.shape[1]))


def sample_grasp_poses(
    params,
    *,
    object_points: jax.Array,
    object_normals: jax.Array,
    rng_key: jax.Array,
    model_config: ModelConfig,
    diffusion_config: DiffusionConfig,
    normalizer: PoseNormalizer,
    schedule: DiffusionSchedule,
    num_samples: int = 1,
) -> SampleBatch:
    batch_size = int(num_samples)
    if batch_size <= 0:
        raise ValueError("num_samples must be positive.")

    scene_points = _broadcast_scene_tensor(jnp.asarray(object_points, dtype=jnp.float32), batch_size)
    scene_normals = _broadcast_scene_tensor(jnp.asarray(object_normals, dtype=jnp.float32), batch_size)
    current = jax.random.normal(rng_key, (batch_size, model_config.pose_dim), dtype=jnp.float32)

    for step in range(diffusion_config.steps - 1, -1, -1):
        timesteps = jnp.full((batch_size,), step, dtype=jnp.int32)
        pred_noise = apply_model(
            params,
            current,
            timesteps,
            scene_points,
            scene_normals,
            model_config,
        )
        pred_x0 = predict_x0(schedule, current, timesteps, pred_noise)
        if step == 0:
            current = pred_x0
            continue
        alpha_prev = schedule.alphas_cumprod[step - 1]
        current = jnp.sqrt(alpha_prev) * pred_x0 + jnp.sqrt(1.0 - alpha_prev) * pred_noise

    return SampleBatch(
        normalized_pose=current,
        pose=normalizer.denormalize_jax(current),
    )
