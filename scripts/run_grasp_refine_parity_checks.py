#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_refine.diffusion import make_diffusion_schedule, predict_x0, q_sample
from grasp_refine.sampling import (
    DpmSolverConfig,
    _dpm_time_steps,
    _get_orders_and_timesteps_for_singlestep,
    _inverse_lambda,
    _marginal_alpha,
    _marginal_lambda,
    _marginal_std,
    _posterior_mean_variance,
)
from grasp_refine.training_types import DiffusionConfig


def _np_predict_x0(schedule, x_t: np.ndarray, t: np.ndarray, pred_noise: np.ndarray) -> np.ndarray:
    return schedule["sqrt_recip_alphas_cumprod"][t][:, None] * x_t - schedule["sqrt_recipm1_alphas_cumprod"][t][:, None] * pred_noise


def _np_q_sample(schedule, x0: np.ndarray, t: np.ndarray, noise: np.ndarray) -> np.ndarray:
    return schedule["sqrt_alphas_cumprod"][t][:, None] * x0 + schedule["sqrt_one_minus_alphas_cumprod"][t][:, None] * noise


def _np_schedule(config: DiffusionConfig) -> dict[str, np.ndarray]:
    betas = np.linspace(config.beta_start, config.beta_end, config.steps, dtype=np.float32)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.concatenate([np.ones((1,), dtype=np.float32), alphas_cumprod[:-1]], axis=0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / np.maximum(1.0 - alphas_cumprod, 1.0e-8)
    posterior_log_variance_clipped = np.log(np.maximum(posterior_variance, 1.0e-20))
    posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / np.maximum(1.0 - alphas_cumprod, 1.0e-8)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / np.maximum(1.0 - alphas_cumprod, 1.0e-8)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": np.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": np.sqrt(1.0 - alphas_cumprod),
        "sqrt_recip_alphas_cumprod": np.sqrt(1.0 / alphas_cumprod),
        "sqrt_recipm1_alphas_cumprod": np.sqrt(1.0 / alphas_cumprod - 1.0),
        "posterior_variance": posterior_variance,
        "posterior_log_variance_clipped": posterior_log_variance_clipped,
        "posterior_mean_coef1": posterior_mean_coef1,
        "posterior_mean_coef2": posterior_mean_coef2,
    }


def _max_abs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _np_interp_safe(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    if xp[0] > xp[-1]:
        xp = xp[::-1]
        fp = fp[::-1]
    return np.interp(x, xp, fp).astype(np.float32)


def _np_posterior_mean_variance(
    schedule: dict[str, np.ndarray],
    x_t: np.ndarray,
    t: np.ndarray,
    pred_noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_x0 = _np_predict_x0(schedule, x_t, t, pred_noise)
    model_mean = (
        schedule["posterior_mean_coef1"][t][:, None] * pred_x0
        + schedule["posterior_mean_coef2"][t][:, None] * x_t
    )
    model_variance = schedule["posterior_variance"][t][:, None]
    model_log_variance = schedule["posterior_log_variance_clipped"][t][:, None]
    return model_mean, model_variance, model_log_variance, pred_x0


def main() -> None:
    config = DiffusionConfig(steps=16, beta_start=1.0e-4, beta_end=1.0e-2)
    schedule = make_diffusion_schedule(config)
    np_schedule = _np_schedule(config)

    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((2, 15), dtype=np.float32)
    noise = rng.standard_normal((2, 15), dtype=np.float32)
    pred_noise = rng.standard_normal((2, 15), dtype=np.float32)
    t = np.asarray([3, 11], dtype=np.int32)

    q_jax = q_sample(schedule, jnp.asarray(x0), jnp.asarray(t), jnp.asarray(noise))
    q_np = _np_q_sample(np_schedule, x0, t, noise)
    x0_jax = predict_x0(schedule, jnp.asarray(q_jax), jnp.asarray(t), jnp.asarray(pred_noise))
    x0_np = _np_predict_x0(np_schedule, q_np, t, pred_noise)
    posterior_mean_jax, posterior_var_jax, posterior_logvar_jax, posterior_x0_jax = _posterior_mean_variance(
        schedule,
        jnp.asarray(q_jax),
        jnp.asarray(t),
        jnp.asarray(pred_noise),
    )
    posterior_mean_np, posterior_var_np, posterior_logvar_np, posterior_x0_np = _np_posterior_mean_variance(
        np_schedule,
        q_np,
        t,
        pred_noise,
    )

    t_cont = jnp.asarray([1.0, 0.7], dtype=jnp.float32)
    lambda_jax = _marginal_lambda(schedule, t_cont)
    alpha_jax = _marginal_alpha(schedule, t_cont)
    std_jax = _marginal_std(schedule, t_cont)
    inverse_t = _inverse_lambda(schedule, lambda_jax)

    total_n = int(schedule.betas.shape[0])
    t_grid = np.linspace(0.0, 1.0, total_n + 1, dtype=np.float32)[1:]
    log_alpha_grid = 0.5 * np.log(np.asarray(schedule.alphas_cumprod))
    interp_log_alpha = _np_interp_safe(np.asarray(t_cont), t_grid, log_alpha_grid)
    alpha_torch = np.exp(interp_log_alpha)
    std_torch = np.sqrt(np.maximum(1.0 - np.exp(2.0 * interp_log_alpha), 1.0e-20))
    lambda_torch = interp_log_alpha - 0.5 * np.log(np.maximum(1.0 - np.exp(2.0 * interp_log_alpha), 1.0e-20))
    lambda_grid = log_alpha_grid - 0.5 * np.log(np.maximum(1.0 - np.exp(2.0 * log_alpha_grid), 1.0e-20))
    inverse_t_torch = _np_interp_safe(np.asarray(lambda_jax), lambda_grid, t_grid)

    dpm_cfg = DpmSolverConfig(steps=7, order=3, t_start=1.0, t_end=0.01, method="singlestep")
    time_steps_jax = _dpm_time_steps(dpm_cfg)
    outer_jax, orders_jax = _get_orders_and_timesteps_for_singlestep(dpm_cfg)
    full_np = np.linspace(dpm_cfg.t_start, dpm_cfg.t_end, dpm_cfg.steps + 1, dtype=np.float32)
    expected_orders = [3, 3, 1]
    cumulative = np.cumsum([0] + expected_orders)
    outer_np = full_np[cumulative]

    results = {
        "q_sample": _max_abs(q_jax, q_np),
        "predict_x0": _max_abs(x0_jax, x0_np),
        "posterior_mean": _max_abs(posterior_mean_jax, posterior_mean_np),
        "posterior_var": _max_abs(posterior_var_jax, posterior_var_np),
        "posterior_logvar": _max_abs(posterior_logvar_jax, posterior_logvar_np),
        "posterior_x0": _max_abs(posterior_x0_jax, posterior_x0_np),
        "marginal_lambda": _max_abs(lambda_jax, lambda_torch),
        "marginal_alpha": _max_abs(alpha_jax, alpha_torch),
        "marginal_std": _max_abs(std_jax, std_torch),
        "inverse_lambda": _max_abs(inverse_t, inverse_t_torch),
        "dpm_time_steps": _max_abs(time_steps_jax, full_np),
        "dpm_outer_steps": _max_abs(outer_jax, outer_np),
    }
    for name, value in results.items():
        print(f"{name:18s}: {value:.8f}")
    print(f"dpm_orders         : {orders_jax}")
    if orders_jax != expected_orders:
        raise SystemExit(f"dpm_orders mismatch: got {orders_jax}, expected {expected_orders}")
    worst = max(results.values())
    print(f"status             : {'PASS' if worst < 1.0e-5 else 'FAIL'}")


if __name__ == "__main__":
    main()
