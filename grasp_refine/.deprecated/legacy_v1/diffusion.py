from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .inspire_hand import InspireHandSpec
from .losses import joint_limit_loss, root_distance_loss
from .model_factory import apply_model
from .normalization import PoseNormalizer
from .types import DiffusionConfig, LossConfig, ModelConfig


class DiffusionSchedule(NamedTuple):
    betas: jax.Array
    alphas_cumprod: jax.Array
    sqrt_alphas_cumprod: jax.Array
    sqrt_one_minus_alphas_cumprod: jax.Array
    sqrt_recip_alphas_cumprod: jax.Array
    sqrt_recipm1_alphas_cumprod: jax.Array


def make_diffusion_schedule(config: DiffusionConfig) -> DiffusionSchedule:
    betas = jnp.linspace(config.beta_start, config.beta_end, config.steps, dtype=jnp.float32)
    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas, axis=0)
    return DiffusionSchedule(
        betas=betas,
        alphas_cumprod=alphas_cumprod,
        sqrt_alphas_cumprod=jnp.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=jnp.sqrt(1.0 - alphas_cumprod),
        sqrt_recip_alphas_cumprod=jnp.sqrt(1.0 / alphas_cumprod),
        sqrt_recipm1_alphas_cumprod=jnp.sqrt(1.0 / alphas_cumprod - 1.0),
    )


def sample_timesteps(rng_key: jax.Array, batch_size: int, config: DiffusionConfig) -> jax.Array:
    if config.rand_t_type == "all":
        return jax.random.randint(rng_key, (batch_size,), minval=0, maxval=config.steps, dtype=jnp.int32)
    half = jax.random.randint(rng_key, ((batch_size + 1) // 2,), minval=0, maxval=config.steps, dtype=jnp.int32)
    mirrored = config.steps - half - 1
    if batch_size % 2 == 1:
        mirrored = mirrored[:-1]
    return jnp.concatenate([half, mirrored], axis=0)


def q_sample(schedule: DiffusionSchedule, x0: jax.Array, timesteps: jax.Array, noise: jax.Array) -> jax.Array:
    return (
        schedule.sqrt_alphas_cumprod[timesteps][:, None] * x0
        + schedule.sqrt_one_minus_alphas_cumprod[timesteps][:, None] * noise
    )


def predict_x0(schedule: DiffusionSchedule, x_t: jax.Array, timesteps: jax.Array, pred_noise: jax.Array) -> jax.Array:
    return (
        schedule.sqrt_recip_alphas_cumprod[timesteps][:, None] * x_t
        - schedule.sqrt_recipm1_alphas_cumprod[timesteps][:, None] * pred_noise
    )


def _weighted_mean(values: jax.Array, sample_weight: jax.Array | None = None) -> jax.Array:
    if sample_weight is None:
        return jnp.mean(values)
    weights = jnp.asarray(sample_weight, dtype=values.dtype)
    denom = jnp.maximum(jnp.sum(weights), jnp.asarray(1.0, dtype=values.dtype))
    return jnp.sum(values * weights) / denom


def diffusion_loss(
    params,
    batch: dict[str, jax.Array],
    rng_key: jax.Array,
    *,
    model_config: ModelConfig,
    diffusion_config: DiffusionConfig,
    loss_config: LossConfig,
    normalizer: PoseNormalizer,
    hand_spec: InspireHandSpec,
    schedule: DiffusionSchedule,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    t_key, noise_key = jax.random.split(rng_key)
    sample_weight = batch.get("sample_weight")
    timesteps = sample_timesteps(t_key, batch["pose"].shape[0], diffusion_config)
    noise = jax.random.normal(noise_key, batch["pose"].shape, dtype=batch["pose"].dtype)
    noisy_pose = q_sample(schedule, batch["pose"], timesteps, noise)
    pred_noise = apply_model(
        params,
        noisy_pose,
        timesteps,
        batch["object_points"],
        batch["object_normals"],
        model_config,
        object_index=batch.get("object_index"),
    )
    pred_x0 = predict_x0(schedule, noisy_pose, timesteps, pred_noise)
    denormalized_pose = normalizer.denormalize_jax(pred_x0)

    if diffusion_config.loss_type == "l1":
        noise_per_sample = jnp.mean(jnp.abs(pred_noise - noise), axis=-1)
    else:
        noise_per_sample = jnp.mean(jnp.square(pred_noise - noise), axis=-1)
    noise_loss = _weighted_mean(noise_per_sample, sample_weight)
    joint_loss = joint_limit_loss(denormalized_pose, hand_spec, sample_weight=sample_weight)
    root_loss = root_distance_loss(
        denormalized_pose,
        batch["object_points"],
        loss_config.root_distance_threshold,
        sample_weight=sample_weight,
    )
    total_loss = (
        float(loss_config.noise_weight) * noise_loss
        + float(loss_config.joint_limit_weight) * joint_loss
        + float(loss_config.root_distance_weight) * root_loss
    )
    metrics = {
        "loss": total_loss,
        "noise_loss": noise_loss,
        "joint_limit_loss": joint_loss,
        "root_distance_loss": root_loss,
    }
    return total_loss, metrics
