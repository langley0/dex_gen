from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class AdamState(NamedTuple):
    step: jax.Array
    mean: object
    var: object


def init_adam(params) -> AdamState:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return AdamState(
        step=jnp.asarray(0, dtype=jnp.int32),
        mean=zeros,
        var=zeros,
    )


def adam_update(
    params,
    grads,
    state: AdamState,
    *,
    lr: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1.0e-8,
):
    step = state.step + 1
    mean = jax.tree_util.tree_map(lambda m, g: beta1 * m + (1.0 - beta1) * g, state.mean, grads)
    var = jax.tree_util.tree_map(lambda v, g: beta2 * v + (1.0 - beta2) * (g * g), state.var, grads)

    bias_correction1 = 1.0 - beta1**step.astype(jnp.float32)
    bias_correction2 = 1.0 - beta2**step.astype(jnp.float32)
    mean_hat = jax.tree_util.tree_map(lambda m: m / bias_correction1, mean)
    var_hat = jax.tree_util.tree_map(lambda v: v / bias_correction2, var)
    updates = jax.tree_util.tree_map(lambda m, v: m / (jnp.sqrt(v) + eps), mean_hat, var_hat)
    next_params = jax.tree_util.tree_map(lambda p, u: p - float(lr) * u, params, updates)
    return next_params, AdamState(step=step, mean=mean, var=var)
