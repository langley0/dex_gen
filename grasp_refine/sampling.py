from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .checkpoint import load_checkpoint
from .diffusion import DiffusionConfig, DiffusionSchedule, _dga_full_pose, make_diffusion_schedule, predict_x0
from .hand_points import DgaHandPointSpec, full_pose_distance_points, full_pose_key_points, full_pose_surface_points, load_dga_hand_point_spec
from .loader import DgaBatch, LoadedDgaDataset, iterate_dga_batches, load_saved_dga_dataset
from .losses import erf_loss, spf_loss, srf_loss
from .model_factory import apply_model, condition_model
from .training_types import ModelConfig


@dataclass(frozen=True)
class SamplingOutput:
    samples: np.ndarray
    samples_full: np.ndarray
    trajectory: np.ndarray | None


@dataclass(frozen=True)
class DpmSolverConfig:
    use_dpmsolver: bool = False
    algorithm_type: str = "dpmsolver++"
    steps: int = 10
    order: int = 1
    skip_type: str = "time_uniform"
    t_start: float = 1.0
    t_end: float = 0.01
    method: str = "singlestep"
    lower_order_final: bool = True
    dynamic_thresholding_ratio: float = 0.995
    thresholding_max_val: float = 1.0


@dataclass(frozen=True)
class GuidanceConfig:
    enabled: bool = False
    guidance_scale: float = 1.0
    grad_scale: float = 0.1
    clip_grad_min: float = -0.1
    clip_grad_max: float = 0.1
    erf_weight: float = 0.3
    spf_weight: float = 1.0
    srf_weight: float = 1.0
    opt_interval: int = 1


def _posterior_mean_variance(
    schedule: DiffusionSchedule,
    x_t: jax.Array,
    timesteps: jax.Array,
    pred_noise: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    pred_x0 = predict_x0(schedule, x_t, timesteps, pred_noise)
    model_mean = (
        schedule.posterior_mean_coef1[timesteps][:, None] * pred_x0
        + schedule.posterior_mean_coef2[timesteps][:, None] * x_t
    )
    model_variance = schedule.posterior_variance[timesteps][:, None]
    model_log_variance = schedule.posterior_log_variance_clipped[timesteps][:, None]
    return model_mean, model_variance, model_log_variance, pred_x0


def p_sample(
    params,
    x_t: jax.Array,
    timestep: int,
    *,
    cond: jax.Array,
    model_config: ModelConfig,
    schedule: DiffusionSchedule,
    rng_key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    batch_size = int(x_t.shape[0])
    timesteps = jnp.full((batch_size,), int(timestep), dtype=jnp.int32)
    pred_noise = apply_model(
        params,
        x_t,
        timesteps,
        jnp.zeros((batch_size, 1, 3), dtype=x_t.dtype),
        jnp.zeros((batch_size, 1, 3), dtype=x_t.dtype),
        model_config,
        rng_key=rng_key,
        training=False,
        condition_override=cond,
    )
    model_mean, _, model_log_variance, pred_x0 = _posterior_mean_variance(schedule, x_t, timesteps, pred_noise)
    noise = jax.random.normal(rng_key, x_t.shape, dtype=x_t.dtype) if int(timestep) > 0 else jnp.zeros_like(x_t)
    x_prev = model_mean + jnp.exp(0.5 * model_log_variance) * noise
    return x_prev, pred_x0


def _guidance_objective_from_full_pose(
    full_pose: jax.Array,
    *,
    hand_point_spec: DgaHandPointSpec,
    object_points: jax.Array,
    object_normals: jax.Array,
    guidance_config: GuidanceConfig,
) -> jax.Array:
    hand_surface = full_pose_surface_points(hand_point_spec, full_pose)
    hand_distance = full_pose_distance_points(hand_point_spec, full_pose)
    hand_key = full_pose_key_points(hand_point_spec, full_pose)
    object_pcd_nor = jnp.concatenate([object_points, object_normals], axis=-1)
    erf_value = erf_loss(object_pcd_nor, hand_surface)
    spf_value = spf_loss(hand_distance, object_points)
    srf_value = srf_loss(hand_key)
    return (
        float(guidance_config.erf_weight) * erf_value
        + float(guidance_config.spf_weight) * spf_value
        + float(guidance_config.srf_weight) * srf_value
    )


def _guidance_mix_step(
    x_current: jax.Array,
    x_sample: jax.Array,
    *,
    gradient: jax.Array,
    std: jax.Array,
    guidance_config: GuidanceConfig,
    x_mean: jax.Array | None = None,
    sample_direction: jax.Array | None = None,
) -> jax.Array:
    eps = 1.0e-8
    grad = gradient * float(guidance_config.grad_scale)
    grad_norm = jnp.linalg.norm(grad, axis=1, keepdims=True)
    std_value = std if std.ndim == 2 else std[:, None]
    radius = jnp.sqrt(float(x_current.shape[-1])) * std_value
    d_star = -radius * grad / (grad_norm + eps)
    d_sample = x_sample - x_mean if x_mean is not None else sample_direction
    if d_sample is None:
        raise ValueError("sample_direction is required when x_mean is None.")
    mix_direction = d_sample + float(guidance_config.guidance_scale) * (d_star - d_sample)
    mix_norm = jnp.linalg.norm(mix_direction, axis=1, keepdims=True)
    mix_step = mix_direction / (mix_norm + eps) * radius
    return (x_mean + mix_step) if x_mean is not None else (x_sample + mix_step)


def _guided_ddpm_step(
    params,
    x_t: jax.Array,
    timestep: int,
    *,
    cond: jax.Array,
    model_config: ModelConfig,
    schedule: DiffusionSchedule,
    dataset: LoadedDgaDataset,
    hand_point_spec: DgaHandPointSpec,
    object_points: jax.Array,
    object_normals: jax.Array,
    guidance_config: GuidanceConfig,
    rng_key: jax.Array,
) -> jax.Array:
    batch_size = int(x_t.shape[0])
    timesteps = jnp.full((batch_size,), int(timestep), dtype=jnp.int32)

    def objective_fn(x_current: jax.Array) -> jax.Array:
        pred_noise = apply_model(
            params,
            x_current,
            timesteps,
            jnp.zeros((batch_size, 1, 3), dtype=x_current.dtype),
            jnp.zeros((batch_size, 1, 3), dtype=x_current.dtype),
            model_config,
            rng_key=rng_key,
            training=False,
            condition_override=cond,
        )
        pred_x0 = predict_x0(schedule, x_current, timesteps, pred_noise)
        denorm = dataset.normalizer.denormalize_jax(pred_x0)
        full_pose = _dga_full_pose(denorm)
        objective_value = _guidance_objective_from_full_pose(
            full_pose,
            hand_point_spec=hand_point_spec,
            object_points=object_points,
            object_normals=object_normals,
            guidance_config=guidance_config,
        )
        return objective_value

    def sample_from_current(x_current: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        pred_noise = apply_model(
            params,
            x_current,
            timesteps,
            jnp.zeros((batch_size, 1, 3), dtype=x_current.dtype),
            jnp.zeros((batch_size, 1, 3), dtype=x_current.dtype),
            model_config,
            rng_key=rng_key,
            training=False,
            condition_override=cond,
        )
        model_mean, _, model_log_variance, pred_x0 = _posterior_mean_variance(schedule, x_current, timesteps, pred_noise)
        noise = jax.random.normal(rng_key, x_current.shape, dtype=x_current.dtype) if int(timestep) > 0 else jnp.zeros_like(x_current)
        x_sample = model_mean + jnp.exp(0.5 * model_log_variance) * noise
        sample_std = jnp.exp(0.5 * model_log_variance)
        return x_sample, pred_x0, sample_std, model_mean

    x_sample, pred_x0, sample_std, model_mean = sample_from_current(x_t)
    gradient = jax.grad(objective_fn)(x_t)
    return _guidance_mix_step(
        x_t,
        x_sample,
        gradient=gradient,
        std=sample_std,
        guidance_config=guidance_config,
        x_mean=model_mean,
    )


def _interpolate_1d(x: jax.Array, xp: jax.Array, fp: jax.Array) -> jax.Array:
    indices = jnp.clip(jnp.searchsorted(xp, x, side="left"), 1, xp.shape[0] - 1)
    x0 = xp[indices - 1]
    x1 = xp[indices]
    y0 = fp[indices - 1]
    y1 = fp[indices]
    weight = (x - x0) / jnp.maximum(x1 - x0, 1.0e-12)
    return y0 + weight * (y1 - y0)


def _marginal_log_mean_coeff(schedule: DiffusionSchedule, t_continuous: jax.Array) -> jax.Array:
    total_n = int(schedule.betas.shape[0])
    t_array = jnp.linspace(0.0, 1.0, total_n + 1, dtype=jnp.float32)[1:]
    log_alpha_array = 0.5 * jnp.log(schedule.alphas_cumprod)
    clipped_t = jnp.clip(t_continuous.astype(jnp.float32), t_array[0], t_array[-1])
    return _interpolate_1d(clipped_t, t_array, log_alpha_array)


def _marginal_alpha(schedule: DiffusionSchedule, t_continuous: jax.Array) -> jax.Array:
    return jnp.exp(_marginal_log_mean_coeff(schedule, t_continuous))


def _marginal_std(schedule: DiffusionSchedule, t_continuous: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.maximum(1.0 - jnp.exp(2.0 * _marginal_log_mean_coeff(schedule, t_continuous)), 1.0e-20))


def _marginal_lambda(schedule: DiffusionSchedule, t_continuous: jax.Array) -> jax.Array:
    log_mean = _marginal_log_mean_coeff(schedule, t_continuous)
    log_std = 0.5 * jnp.log(jnp.maximum(1.0 - jnp.exp(2.0 * log_mean), 1.0e-20))
    return log_mean - log_std


def _inverse_lambda(schedule: DiffusionSchedule, lambda_value: jax.Array) -> jax.Array:
    total_n = int(schedule.betas.shape[0])
    t_array = jnp.linspace(0.0, 1.0, total_n + 1, dtype=jnp.float32)[1:]
    lambda_array = _marginal_lambda(schedule, t_array)
    return _interpolate_1d(lambda_value.astype(jnp.float32), lambda_array[::-1], t_array[::-1])


def _get_model_input_time(schedule: DiffusionSchedule, t_continuous: jax.Array) -> jax.Array:
    total_n = int(schedule.betas.shape[0])
    return (t_continuous - 1.0 / total_n) * total_n


def _dynamic_thresholding(
    x0: jax.Array,
    *,
    ratio: float,
    max_val: float,
) -> jax.Array:
    batch_size = int(x0.shape[0])
    flat = jnp.abs(x0).reshape(batch_size, -1)
    quantile = jnp.quantile(flat, float(ratio), axis=1)
    scale = jnp.maximum(quantile, float(max_val))
    scale = scale.reshape((batch_size,) + (1,) * (x0.ndim - 1))
    return jnp.clip(x0, -scale, scale) / scale


def _data_prediction_from_continuous_model(
    params,
    x_t: jax.Array,
    t_continuous: jax.Array,
    *,
    cond: jax.Array,
    model_config: ModelConfig,
    schedule: DiffusionSchedule,
    dpm_config: DpmSolverConfig,
) -> jax.Array:
    input_rank = x_t.ndim
    x_model = x_t[None, :] if input_rank == 1 else x_t
    t_model = t_continuous[None] if t_continuous.ndim == 0 else t_continuous
    t_input = _get_model_input_time(schedule, t_model)
    pred_noise = apply_model(
        params,
        x_model,
        t_input,
        jnp.zeros((x_model.shape[0], 1, 3), dtype=x_model.dtype),
        jnp.zeros((x_model.shape[0], 1, 3), dtype=x_model.dtype),
        model_config,
        training=False,
        condition_override=cond,
    )
    alpha_t = _marginal_alpha(schedule, t_model)[:, None]
    sigma_t = _marginal_std(schedule, t_model)[:, None]
    x0 = (x_model - sigma_t * pred_noise) / alpha_t
    if dpm_config.algorithm_type == "dpmsolver++":
        x0 = _dynamic_thresholding(
            x0,
            ratio=dpm_config.dynamic_thresholding_ratio,
            max_val=dpm_config.thresholding_max_val,
        )
    return x0[0] if input_rank == 1 else x0


def _dpm_solver_first_update(
    params,
    x_t: jax.Array,
    s: jax.Array,
    t: jax.Array,
    *,
    cond: jax.Array,
    model_config: ModelConfig,
    schedule: DiffusionSchedule,
    dpm_config: DpmSolverConfig,
    rng_key: jax.Array,
    guidance_context: tuple[LoadedDgaDataset, DgaHandPointSpec, jax.Array, jax.Array, GuidanceConfig] | None = None,
) -> jax.Array:
    x_t = jnp.reshape(x_t, (-1, x_t.shape[-1]))
    lambda_s = _marginal_lambda(schedule, s)
    lambda_t = _marginal_lambda(schedule, t)
    h = lambda_t - lambda_s
    sigma_s = _marginal_std(schedule, s)[:, None]
    sigma_t = _marginal_std(schedule, t)[:, None]
    alpha_t = _marginal_alpha(schedule, t)[:, None]
    model_s = _data_prediction_from_continuous_model(
        params,
        x_t,
        s,
        cond=cond,
        model_config=model_config,
        schedule=schedule,
        dpm_config=dpm_config,
    )
    phi_1 = jnp.expm1(-h)[:, None]
    x_next = sigma_t / sigma_s * x_t - alpha_t * phi_1 * model_s
    if guidance_context is not None:
        dataset, hand_point_spec, object_points, object_normals, guidance_config = guidance_context
        def objective_fn(x_current: jax.Array) -> jax.Array:
            x0_pred = _data_prediction_from_continuous_model(
                params,
                x_current,
                s,
                cond=cond,
                model_config=model_config,
                schedule=schedule,
                dpm_config=dpm_config,
            )
            denorm = dataset.normalizer.denormalize_jax(x0_pred)
            full_pose = _dga_full_pose(denorm)
            objective_value = _guidance_objective_from_full_pose(
                full_pose,
                hand_point_spec=hand_point_spec,
                object_points=object_points,
                object_normals=object_normals,
                guidance_config=guidance_config,
            )
            return objective_value

        std = _marginal_std(schedule, s)[:, None] * jnp.sqrt(jnp.maximum(-phi_1, 0.0))[:, None]
        gradient = jax.grad(objective_fn)(x_t)
        sample_direction = std * jax.random.normal(rng_key, x_t.shape, dtype=x_t.dtype)
        x_next = _guidance_mix_step(
            x_t,
            x_next,
            gradient=gradient,
            std=std,
            guidance_config=guidance_config,
            x_mean=None,
            sample_direction=sample_direction,
        )
    return x_next


def _guided_dpm_solver_correction(
    params,
    x_current: jax.Array,
    x_sample: jax.Array,
    *,
    time_for_objective: jax.Array,
    cond: jax.Array,
    model_config: ModelConfig,
    schedule: DiffusionSchedule,
    dpm_config: DpmSolverConfig,
    rng_key: jax.Array,
    guidance_context: tuple[LoadedDgaDataset, DgaHandPointSpec, jax.Array, jax.Array, GuidanceConfig],
    std: jax.Array,
    x0_fn,
) -> jax.Array:
    dataset, hand_point_spec, object_points, object_normals, guidance_config = guidance_context

    def objective_fn(x_variable: jax.Array) -> jax.Array:
        x0_pred = x0_fn(x_variable, time_for_objective)
        denorm = dataset.normalizer.denormalize_jax(x0_pred)
        full_pose = _dga_full_pose(denorm)
        return _guidance_objective_from_full_pose(
            full_pose,
            hand_point_spec=hand_point_spec,
            object_points=object_points,
            object_normals=object_normals,
            guidance_config=guidance_config,
        )

    gradient = jax.grad(objective_fn)(x_current)
    sample_direction = std * jax.random.normal(rng_key, x_current.shape, dtype=x_current.dtype)
    return _guidance_mix_step(
        x_current,
        x_sample,
        gradient=gradient,
        std=std,
        guidance_config=guidance_config,
        x_mean=None,
        sample_direction=sample_direction,
    )


def _singlestep_second_update(
    params,
    x_t: jax.Array,
    s: jax.Array,
    t: jax.Array,
    *,
    cond: jax.Array,
    model_config: ModelConfig,
    schedule: DiffusionSchedule,
    dpm_config: DpmSolverConfig,
    rng_key: jax.Array,
    guidance_context: tuple[LoadedDgaDataset, DgaHandPointSpec, jax.Array, jax.Array, GuidanceConfig] | None = None,
    r1: float = 0.5,
) -> jax.Array:
    lambda_s = _marginal_lambda(schedule, s)
    lambda_t = _marginal_lambda(schedule, t)
    h = lambda_t - lambda_s
    lambda_s1 = lambda_s + float(r1) * h
    s1 = _inverse_lambda(schedule, lambda_s1)
    sigma_s = _marginal_std(schedule, s)[:, None]
    sigma_s1 = _marginal_std(schedule, s1)[:, None]
    sigma_t = _marginal_std(schedule, t)[:, None]
    alpha_s1 = _marginal_alpha(schedule, s1)[:, None]
    alpha_t = _marginal_alpha(schedule, t)[:, None]
    phi_11 = jnp.expm1(-float(r1) * h)[:, None]
    phi_1 = jnp.expm1(-h)[:, None]
    model_s = _data_prediction_from_continuous_model(
        params,
        x_t,
        s,
        cond=cond,
        model_config=model_config,
        schedule=schedule,
        dpm_config=dpm_config,
    )
    x_s1 = sigma_s1 / sigma_s * x_t - alpha_s1 * phi_11 * model_s
    model_s1 = _data_prediction_from_continuous_model(
        params,
        x_s1,
        s1,
        cond=cond,
        model_config=model_config,
        schedule=schedule,
        dpm_config=dpm_config,
    )
    x_next = (
        sigma_t / sigma_s * x_t
        - alpha_t * phi_1 * model_s
        - (0.5 / float(r1)) * alpha_t * phi_1 * (model_s1 - model_s)
    )
    if guidance_context is not None:
        std = _marginal_std(schedule, s)[:, None] * jnp.sqrt(jnp.maximum(-phi_1, 0.0))
        x_next = _guided_dpm_solver_correction(
            params,
            x_t,
            x_next,
            time_for_objective=s,
            cond=cond,
            model_config=model_config,
            schedule=schedule,
            dpm_config=dpm_config,
            rng_key=rng_key,
            guidance_context=guidance_context,
            std=std,
            x0_fn=lambda x_var, time_var: _data_prediction_from_continuous_model(
                params,
                x_var,
                time_var,
                cond=cond,
                model_config=model_config,
                schedule=schedule,
                dpm_config=dpm_config,
            ),
        )
    return x_next


def _singlestep_third_update(
    params,
    x_t: jax.Array,
    s: jax.Array,
    t: jax.Array,
    *,
    cond: jax.Array,
    model_config: ModelConfig,
    schedule: DiffusionSchedule,
    dpm_config: DpmSolverConfig,
    rng_key: jax.Array,
    guidance_context: tuple[LoadedDgaDataset, DgaHandPointSpec, jax.Array, jax.Array, GuidanceConfig] | None = None,
    r1: float = 1.0 / 3.0,
    r2: float = 2.0 / 3.0,
) -> jax.Array:
    lambda_s = _marginal_lambda(schedule, s)
    lambda_t = _marginal_lambda(schedule, t)
    h = lambda_t - lambda_s
    s1 = _inverse_lambda(schedule, lambda_s + float(r1) * h)
    s2 = _inverse_lambda(schedule, lambda_s + float(r2) * h)
    sigma_s = _marginal_std(schedule, s)[:, None]
    sigma_s1 = _marginal_std(schedule, s1)[:, None]
    sigma_s2 = _marginal_std(schedule, s2)[:, None]
    sigma_t = _marginal_std(schedule, t)[:, None]
    alpha_s1 = _marginal_alpha(schedule, s1)[:, None]
    alpha_s2 = _marginal_alpha(schedule, s2)[:, None]
    alpha_t = _marginal_alpha(schedule, t)[:, None]
    phi_11 = jnp.expm1(-float(r1) * h)[:, None]
    phi_12 = jnp.expm1(-float(r2) * h)[:, None]
    phi_1 = jnp.expm1(-h)[:, None]
    phi_22 = phi_12 / (float(r2) * h)[:, None] + 1.0
    phi_2 = phi_1 / h[:, None] + 1.0

    model_s = _data_prediction_from_continuous_model(
        params,
        x_t,
        s,
        cond=cond,
        model_config=model_config,
        schedule=schedule,
        dpm_config=dpm_config,
    )
    x_s1 = sigma_s1 / sigma_s * x_t - alpha_s1 * phi_11 * model_s
    model_s1 = _data_prediction_from_continuous_model(
        params,
        x_s1,
        s1,
        cond=cond,
        model_config=model_config,
        schedule=schedule,
        dpm_config=dpm_config,
    )
    x_s2 = sigma_s2 / sigma_s * x_t - alpha_s2 * phi_12 * model_s + float(r2 / r1) * alpha_s2 * phi_22 * (model_s1 - model_s)
    model_s2 = _data_prediction_from_continuous_model(
        params,
        x_s2,
        s2,
        cond=cond,
        model_config=model_config,
        schedule=schedule,
        dpm_config=dpm_config,
    )
    x_next = sigma_t / sigma_s * x_t - alpha_t * phi_1 * model_s + (1.0 / float(r2)) * alpha_t * phi_2 * (model_s2 - model_s)
    if guidance_context is not None:
        std = _marginal_std(schedule, s)[:, None] * jnp.sqrt(jnp.maximum(-phi_1, 0.0))
        x_next = _guided_dpm_solver_correction(
            params,
            x_t,
            x_next,
            time_for_objective=s,
            cond=cond,
            model_config=model_config,
            schedule=schedule,
            dpm_config=dpm_config,
            rng_key=rng_key,
            guidance_context=guidance_context,
            std=std,
            x0_fn=lambda x_var, time_var: _data_prediction_from_continuous_model(
                params,
                x_var,
                time_var,
                cond=cond,
                model_config=model_config,
                schedule=schedule,
                dpm_config=dpm_config,
            ),
        )
    return x_next


def _multistep_second_update(
    x_t: jax.Array,
    model_prev_list: list[jax.Array],
    t_prev_list: list[jax.Array],
    t: jax.Array,
    *,
    schedule: DiffusionSchedule,
    rng_key: jax.Array,
    params,
    cond: jax.Array,
    model_config: ModelConfig,
    dpm_config: DpmSolverConfig,
    guidance_context: tuple[LoadedDgaDataset, DgaHandPointSpec, jax.Array, jax.Array, GuidanceConfig] | None = None,
) -> jax.Array:
    model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
    t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
    lambda_prev_1 = _marginal_lambda(schedule, t_prev_1)
    lambda_prev_0 = _marginal_lambda(schedule, t_prev_0)
    lambda_t = _marginal_lambda(schedule, t)
    sigma_prev_0 = _marginal_std(schedule, t_prev_0)[:, None]
    sigma_t = _marginal_std(schedule, t)[:, None]
    alpha_t = _marginal_alpha(schedule, t)[:, None]
    h_0 = lambda_prev_0 - lambda_prev_1
    h = lambda_t - lambda_prev_0
    r0 = h_0 / h
    d1_0 = (1.0 / r0)[:, None] * (model_prev_0 - model_prev_1)
    phi_1 = jnp.expm1(-h)[:, None]
    x_next = sigma_t / sigma_prev_0 * x_t - alpha_t * phi_1 * model_prev_0 - 0.5 * alpha_t * phi_1 * d1_0
    if guidance_context is not None:
        std = _marginal_std(schedule, t_prev_0)[:, None] * jnp.sqrt(jnp.maximum(-phi_1, 0.0))
        x_next = _guided_dpm_solver_correction(
            params,
            x_t,
            x_next,
            time_for_objective=t_prev_0,
            cond=cond,
            model_config=model_config,
            schedule=schedule,
            dpm_config=dpm_config,
            rng_key=rng_key,
            guidance_context=guidance_context,
            std=std,
            x0_fn=lambda x_var, time_var: _data_prediction_from_continuous_model(
                params,
                x_var,
                time_var,
                cond=cond,
                model_config=model_config,
                schedule=schedule,
                dpm_config=dpm_config,
            ),
        )
    return x_next


def _multistep_third_update(
    x_t: jax.Array,
    model_prev_list: list[jax.Array],
    t_prev_list: list[jax.Array],
    t: jax.Array,
    *,
    schedule: DiffusionSchedule,
    rng_key: jax.Array,
    params,
    cond: jax.Array,
    model_config: ModelConfig,
    dpm_config: DpmSolverConfig,
    guidance_context: tuple[LoadedDgaDataset, DgaHandPointSpec, jax.Array, jax.Array, GuidanceConfig] | None = None,
) -> jax.Array:
    model_prev_2, model_prev_1, model_prev_0 = model_prev_list[-3], model_prev_list[-2], model_prev_list[-1]
    t_prev_2, t_prev_1, t_prev_0 = t_prev_list[-3], t_prev_list[-2], t_prev_list[-1]
    lambda_prev_2 = _marginal_lambda(schedule, t_prev_2)
    lambda_prev_1 = _marginal_lambda(schedule, t_prev_1)
    lambda_prev_0 = _marginal_lambda(schedule, t_prev_0)
    lambda_t = _marginal_lambda(schedule, t)
    sigma_prev_0 = _marginal_std(schedule, t_prev_0)[:, None]
    sigma_t = _marginal_std(schedule, t)[:, None]
    alpha_t = _marginal_alpha(schedule, t)[:, None]
    h_1 = lambda_prev_1 - lambda_prev_2
    h_0 = lambda_prev_0 - lambda_prev_1
    h = lambda_t - lambda_prev_0
    r0 = h_0 / h
    r1 = h_1 / h
    d1_0 = (1.0 / r0)[:, None] * (model_prev_0 - model_prev_1)
    d1_1 = (1.0 / r1)[:, None] * (model_prev_1 - model_prev_2)
    d1 = d1_0 + (r0 / (r0 + r1))[:, None] * (d1_0 - d1_1)
    d2 = (1.0 / (r0 + r1))[:, None] * (d1_0 - d1_1)
    phi_1 = jnp.expm1(-h)[:, None]
    phi_2 = phi_1 / h[:, None] + 1.0
    phi_3 = phi_2 / h[:, None] - 0.5
    x_next = sigma_t / sigma_prev_0 * x_t - alpha_t * phi_1 * model_prev_0 + alpha_t * phi_2 * d1 - alpha_t * phi_3 * d2
    if guidance_context is not None:
        std = _marginal_std(schedule, t_prev_0)[:, None] * jnp.sqrt(jnp.maximum(-phi_1, 0.0))
        x_next = _guided_dpm_solver_correction(
            params,
            x_t,
            x_next,
            time_for_objective=t_prev_0,
            cond=cond,
            model_config=model_config,
            schedule=schedule,
            dpm_config=dpm_config,
            rng_key=rng_key,
            guidance_context=guidance_context,
            std=std,
            x0_fn=lambda x_var, time_var: _data_prediction_from_continuous_model(
                params,
                x_var,
                time_var,
                cond=cond,
                model_config=model_config,
                schedule=schedule,
                dpm_config=dpm_config,
            ),
        )
    return x_next


def _dpm_time_steps(config: DpmSolverConfig) -> jax.Array:
    if config.skip_type != "time_uniform":
        raise ValueError(f"Currently only time_uniform is supported, got {config.skip_type!r}")
    return jnp.linspace(float(config.t_start), float(config.t_end), int(config.steps) + 1, dtype=jnp.float32)


def _get_orders_and_timesteps_for_singlestep(config: DpmSolverConfig) -> tuple[jax.Array, list[int]]:
    order = int(config.order)
    steps = int(config.steps)
    if order == 3:
        k = steps // 3 + 1
        if steps % 3 == 0:
            orders = [3] * (k - 2) + [2, 1]
        elif steps % 3 == 1:
            orders = [3] * (k - 1) + [1]
        else:
            orders = [3] * (k - 1) + [2]
    elif order == 2:
        if steps % 2 == 0:
            k = steps // 2
            orders = [2] * k
        else:
            k = steps // 2 + 1
            orders = [2] * (k - 1) + [1]
    elif order == 1:
        orders = [1] * steps
    else:
        raise ValueError(f"Unsupported DPM order: {order}")
    cumulative = np.cumsum([0] + orders)
    full = np.linspace(float(config.t_start), float(config.t_end), int(config.steps) + 1, dtype=np.float32)
    return jnp.asarray(full[cumulative], dtype=jnp.float32), orders


def dpm_solver_sample_loop(
    params,
    object_points: jax.Array,
    object_normals: jax.Array,
    *,
    pose_dim: int,
    model_config: ModelConfig,
    schedule: DiffusionSchedule,
    rng_key: jax.Array,
    dpm_config: DpmSolverConfig,
    return_trajectory: bool = False,
    guidance_context: tuple[LoadedDgaDataset, DgaHandPointSpec, jax.Array, jax.Array, GuidanceConfig] | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    if dpm_config.algorithm_type != "dpmsolver++":
        raise ValueError(f"Currently only algorithm_type='dpmsolver++' is supported, got {dpm_config.algorithm_type!r}.")
    batch_size = int(object_points.shape[0])
    cond = condition_model(params, object_points, object_normals, model_config)
    x_t = jax.random.normal(rng_key, (batch_size, pose_dim), dtype=jnp.float32)
    steps = [x_t] if return_trajectory else None
    loop_keys = list(jax.random.split(rng_key, max(int(dpm_config.steps) * 3, 1)))

    if dpm_config.method == "singlestep":
        time_steps_outer, orders = _get_orders_and_timesteps_for_singlestep(dpm_config)
        for step_index, order in enumerate(orders):
            s = jnp.full((batch_size,), time_steps_outer[step_index], dtype=jnp.float32)
            t = jnp.full((batch_size,), time_steps_outer[step_index + 1], dtype=jnp.float32)
            if order == 1:
                x_t = _dpm_solver_first_update(
                    params,
                    x_t,
                    s,
                    t,
                    cond=cond,
                    model_config=model_config,
                    schedule=schedule,
                    dpm_config=dpm_config,
                    rng_key=loop_keys[step_index],
                    guidance_context=guidance_context,
                )
            elif order == 2:
                x_t = _singlestep_second_update(
                    params,
                    x_t,
                    s,
                    t,
                    cond=cond,
                    model_config=model_config,
                    schedule=schedule,
                    dpm_config=dpm_config,
                    rng_key=loop_keys[step_index],
                    guidance_context=guidance_context,
                )
            else:
                x_t = _singlestep_third_update(
                    params,
                    x_t,
                    s,
                    t,
                    cond=cond,
                    model_config=model_config,
                    schedule=schedule,
                    dpm_config=dpm_config,
                    rng_key=loop_keys[step_index],
                    guidance_context=guidance_context,
                )
            x_t = jnp.reshape(x_t, (batch_size, pose_dim))
            if return_trajectory:
                steps.append(x_t)
    elif dpm_config.method == "multistep":
        if int(dpm_config.steps) < int(dpm_config.order):
            raise ValueError("Multistep DPM-Solver requires steps >= order.")
        time_steps = _dpm_time_steps(dpm_config)
        t_prev_list = [jnp.full((batch_size,), time_steps[0], dtype=jnp.float32)]
        model_prev_list = [
            _data_prediction_from_continuous_model(
                params,
                x_t,
                t_prev_list[0],
                cond=cond,
                model_config=model_config,
                schedule=schedule,
                dpm_config=dpm_config,
            )
        ]
        for step_index in range(1, int(dpm_config.order)):
            t = jnp.full((batch_size,), time_steps[step_index], dtype=jnp.float32)
            if step_index == 1:
                x_t = _dpm_solver_first_update(
                    params,
                    x_t,
                    t_prev_list[-1],
                    t,
                    cond=cond,
                    model_config=model_config,
                    schedule=schedule,
                    dpm_config=dpm_config,
                    rng_key=loop_keys[step_index - 1],
                    guidance_context=guidance_context,
                )
            else:
                x_t = _multistep_second_update(
                    x_t,
                    model_prev_list,
                    t_prev_list,
                    t,
                    schedule=schedule,
                    rng_key=loop_keys[step_index - 1],
                    params=params,
                    cond=cond,
                    model_config=model_config,
                    dpm_config=dpm_config,
                    guidance_context=guidance_context,
                )
            x_t = jnp.reshape(x_t, (batch_size, pose_dim))
            if return_trajectory:
                steps.append(x_t)
            t_prev_list.append(t)
            model_prev_list.append(
                _data_prediction_from_continuous_model(
                    params,
                    x_t,
                    t,
                    cond=cond,
                    model_config=model_config,
                    schedule=schedule,
                    dpm_config=dpm_config,
                )
            )

        for step_index in range(int(dpm_config.order), int(dpm_config.steps) + 1):
            t = jnp.full((batch_size,), time_steps[step_index], dtype=jnp.float32)
            if bool(dpm_config.lower_order_final) and int(dpm_config.steps) < 10:
                step_order = min(int(dpm_config.order), int(dpm_config.steps) + 1 - step_index)
            else:
                step_order = int(dpm_config.order)
            if step_order == 1:
                x_t = _dpm_solver_first_update(
                    params,
                    x_t,
                    t_prev_list[-1],
                    t,
                    cond=cond,
                    model_config=model_config,
                    schedule=schedule,
                    dpm_config=dpm_config,
                    rng_key=loop_keys[min(step_index - 1, len(loop_keys) - 1)],
                    guidance_context=guidance_context,
                )
            elif step_order == 2:
                x_t = _multistep_second_update(
                    x_t,
                    model_prev_list,
                    t_prev_list,
                    t,
                    schedule=schedule,
                    rng_key=loop_keys[min(step_index - 1, len(loop_keys) - 1)],
                    params=params,
                    cond=cond,
                    model_config=model_config,
                    dpm_config=dpm_config,
                    guidance_context=guidance_context,
                )
            else:
                x_t = _multistep_third_update(
                    x_t,
                    model_prev_list,
                    t_prev_list,
                    t,
                    schedule=schedule,
                    rng_key=loop_keys[min(step_index - 1, len(loop_keys) - 1)],
                    params=params,
                    cond=cond,
                    model_config=model_config,
                    dpm_config=dpm_config,
                    guidance_context=guidance_context,
                )
            x_t = jnp.reshape(x_t, (batch_size, pose_dim))
            if return_trajectory:
                steps.append(x_t)
            if step_index < int(dpm_config.steps):
                t_prev_list = t_prev_list[1:] + [t]
                model_prev_list = model_prev_list[1:] + [
                    _data_prediction_from_continuous_model(
                        params,
                        x_t,
                        t,
                        cond=cond,
                        model_config=model_config,
                        schedule=schedule,
                        dpm_config=dpm_config,
                    )
                ]
    else:
        raise ValueError(f"Unsupported DPM method: {dpm_config.method!r}")
    trajectory = None if steps is None else jnp.stack(steps, axis=1)
    return x_t, trajectory


def p_sample_loop(
    params,
    object_points: jax.Array,
    object_normals: jax.Array,
    *,
    pose_dim: int,
    model_config: ModelConfig,
    schedule: DiffusionSchedule,
    rng_key: jax.Array,
    return_trajectory: bool = False,
    dataset: LoadedDgaDataset | None = None,
    hand_point_spec: DgaHandPointSpec | None = None,
    guidance_config: GuidanceConfig | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    batch_size = int(object_points.shape[0])
    cond = condition_model(params, object_points, object_normals, model_config)
    x_t = jax.random.normal(rng_key, (batch_size, pose_dim), dtype=jnp.float32)
    steps = [x_t] if return_trajectory else None
    loop_keys = jax.random.split(rng_key, len(schedule.betas))
    for index, timestep in enumerate(range(len(schedule.betas) - 1, -1, -1)):
        if guidance_config is not None and guidance_config.enabled and int(timestep) % int(guidance_config.opt_interval) == 0:
            if dataset is None or hand_point_spec is None:
                raise ValueError("dataset and hand_point_spec are required for guided DDPM sampling.")
            x_t = _guided_ddpm_step(
                params,
                x_t,
                timestep,
                cond=cond,
                model_config=model_config,
                schedule=schedule,
                dataset=dataset,
                hand_point_spec=hand_point_spec,
                object_points=object_points,
                object_normals=object_normals,
                guidance_config=guidance_config,
                rng_key=loop_keys[index],
            )
        else:
            x_t, _ = p_sample(
                params,
                x_t,
                timestep,
                cond=cond,
                model_config=model_config,
                schedule=schedule,
                rng_key=loop_keys[index],
            )
        if return_trajectory:
            steps.append(x_t)
    trajectory = None if steps is None else jnp.stack(steps, axis=1)
    return x_t, trajectory


def sample(
    params,
    batch: DgaBatch,
    *,
    dataset: LoadedDgaDataset,
    model_config: ModelConfig,
    diffusion_config: DiffusionConfig,
    rng_key: jax.Array,
    k: int = 1,
    return_trajectory: bool = False,
    dpm_config: DpmSolverConfig | None = None,
    guidance_config: GuidanceConfig | None = None,
) -> SamplingOutput:
    schedule = make_diffusion_schedule(diffusion_config)
    object_points = jnp.asarray(batch.object_points, dtype=jnp.float32)
    object_normals = jnp.asarray(batch.object_normals, dtype=jnp.float32)
    hand_side = str(batch.hand_side[0])
    hand_point_spec = load_dga_hand_point_spec(hand_side)
    sample_keys = jax.random.split(rng_key, int(k))
    sampled = []
    trajectories = []
    for sample_key in sample_keys:
        if dpm_config is not None and dpm_config.use_dpmsolver:
            final_x0, trajectory = dpm_solver_sample_loop(
                params,
                object_points,
                object_normals,
                pose_dim=dataset.normalizer.pose_dim,
                model_config=model_config,
                schedule=schedule,
                rng_key=sample_key,
                dpm_config=dpm_config,
                return_trajectory=return_trajectory,
                guidance_context=(
                    dataset,
                    hand_point_spec,
                    object_points,
                    object_normals,
                    guidance_config,
                ) if guidance_config is not None and guidance_config.enabled else None,
            )
        else:
            final_x0, trajectory = p_sample_loop(
                params,
                object_points,
                object_normals,
                pose_dim=dataset.normalizer.pose_dim,
                model_config=model_config,
                schedule=schedule,
                rng_key=sample_key,
                return_trajectory=return_trajectory,
                dataset=dataset,
                hand_point_spec=hand_point_spec,
                guidance_config=guidance_config,
            )
        sampled.append(final_x0)
        if trajectory is not None:
            trajectories.append(trajectory)
    samples_np = np.asarray(jax.device_get(jnp.stack(sampled, axis=1)), dtype=np.float32)
    samples_denorm = dataset.normalizer.denormalize_numpy(samples_np)
    samples_full = np.asarray(_dga_full_pose(jnp.asarray(samples_denorm, dtype=jnp.float32)), dtype=np.float32)

    trajectory_np = None
    if trajectories:
        trajectory_np = np.asarray(jax.device_get(jnp.stack(trajectories, axis=1)), dtype=np.float32)
        trajectory_np = dataset.normalizer.denormalize_numpy(trajectory_np)
    return SamplingOutput(
        samples=samples_denorm,
        samples_full=samples_full,
        trajectory=trajectory_np,
    )


def first_batch(dataset_path: str | Path, *, batch_size: int) -> tuple[LoadedDgaDataset, DgaBatch]:
    dataset = load_saved_dga_dataset(dataset_path)
    batch = next(iterate_dga_batches(dataset, batch_size=batch_size, shuffle=False, seed=0))
    return dataset, batch


def load_latest_checkpoint_state(checkpoint_dir: str | Path):
    state = load_checkpoint(checkpoint_dir, save_separately=True)
    if state is None:
        raise ValueError(f"No checkpoint found in {Path(checkpoint_dir).expanduser().resolve()}")
    return state
