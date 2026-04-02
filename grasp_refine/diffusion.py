from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .hand_points import DgaHandPointSpec, full_pose_distance_points, full_pose_key_points, full_pose_surface_points
from .losses import erf_loss, spf_loss, srf_loss
from .model_factory import apply_model
from .normalization import DgaPoseNormalizer
from .training_types import DiffusionConfig, LossConfig, ModelConfig


IDENTITY_ROT6D = jnp.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=jnp.float32)


class DiffusionSchedule(NamedTuple):
    betas: jax.Array
    alphas: jax.Array
    alphas_cumprod: jax.Array
    alphas_cumprod_prev: jax.Array
    sqrt_alphas_cumprod: jax.Array
    sqrt_one_minus_alphas_cumprod: jax.Array
    sqrt_recip_alphas_cumprod: jax.Array
    sqrt_recipm1_alphas_cumprod: jax.Array
    posterior_variance: jax.Array
    posterior_log_variance_clipped: jax.Array
    posterior_mean_coef1: jax.Array
    posterior_mean_coef2: jax.Array


def make_diffusion_schedule(config: DiffusionConfig) -> DiffusionSchedule:
    betas = jnp.linspace(config.beta_start, config.beta_end, config.steps, dtype=jnp.float32)
    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas, axis=0)
    alphas_cumprod_prev = jnp.concatenate([jnp.ones((1,), dtype=jnp.float32), alphas_cumprod[:-1]], axis=0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / jnp.maximum(1.0 - alphas_cumprod, 1.0e-8)
    posterior_log_variance_clipped = jnp.log(jnp.maximum(posterior_variance, 1.0e-20))
    posterior_mean_coef1 = betas * jnp.sqrt(alphas_cumprod_prev) / jnp.maximum(1.0 - alphas_cumprod, 1.0e-8)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * jnp.sqrt(alphas) / jnp.maximum(1.0 - alphas_cumprod, 1.0e-8)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        sqrt_alphas_cumprod=jnp.sqrt(alphas_cumprod),
        sqrt_one_minus_alphas_cumprod=jnp.sqrt(1.0 - alphas_cumprod),
        sqrt_recip_alphas_cumprod=jnp.sqrt(1.0 / alphas_cumprod),
        sqrt_recipm1_alphas_cumprod=jnp.sqrt(1.0 / alphas_cumprod - 1.0),
        posterior_variance=posterior_variance,
        posterior_log_variance_clipped=posterior_log_variance_clipped,
        posterior_mean_coef1=posterior_mean_coef1,
        posterior_mean_coef2=posterior_mean_coef2,
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


def _dga_full_pose(denormalized_pose: jax.Array) -> jax.Array:
    leading_shape = denormalized_pose.shape[:-1]
    identity_rot = jnp.broadcast_to(IDENTITY_ROT6D, leading_shape + (6,))
    return jnp.concatenate([denormalized_pose[..., :3], identity_rot, denormalized_pose[..., 3:]], axis=-1)


def diffusion_loss(
    params,
    batch: dict[str, jax.Array],
    rng_key: jax.Array,
    *,
    model_config: ModelConfig,
    diffusion_config: DiffusionConfig,
    loss_config: LossConfig,
    normalizer: DgaPoseNormalizer,
    hand_point_spec: DgaHandPointSpec,
    schedule: DiffusionSchedule,
    training: bool = True,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    t_key, noise_key, model_key = jax.random.split(rng_key, 3)
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
        rng_key=model_key,
        training=training,
    )
    pred_x0 = predict_x0(schedule, noisy_pose, timesteps, pred_noise)
    denormalized_pose = normalizer.denormalize_jax(pred_x0)
    full_pose = _dga_full_pose(denormalized_pose)

    hand_surface_points = full_pose_surface_points(hand_point_spec, full_pose)
    hand_distance_points = full_pose_distance_points(hand_point_spec, full_pose)
    hand_key_points = full_pose_key_points(hand_point_spec, full_pose)
    object_pcd_nor = jnp.concatenate([batch["object_points"], batch["object_normals"]], axis=-1)

    if diffusion_config.loss_type == "l1":
        noise_loss = jnp.mean(jnp.abs(pred_noise - noise))
    else:
        noise_loss = jnp.mean(jnp.square(pred_noise - noise))
    erf_value = erf_loss(object_pcd_nor, hand_surface_points)
    spf_value = spf_loss(hand_distance_points, batch["object_points"], threshold=loss_config.spf_threshold)
    srf_value = srf_loss(hand_key_points, threshold=loss_config.srf_threshold)
    total_loss = (
        float(loss_config.noise_weight) * noise_loss
        + float(loss_config.erf_weight) * erf_value
        + float(loss_config.spf_weight) * spf_value
        + float(loss_config.srf_weight) * srf_value
    )
    metrics = {
        "loss": total_loss,
        "noise_loss": noise_loss,
        "erf_loss": erf_value,
        "spf_loss": spf_value,
        "srf_loss": srf_value,
    }
    return total_loss, metrics
