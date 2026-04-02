from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from .training_types import ModelConfig


def _init_linear(key: jax.Array, in_dim: int, out_dim: int) -> dict[str, jax.Array]:
    limit = math.sqrt(6.0 / float(in_dim + out_dim))
    w_key, _ = jax.random.split(key)
    weight = jax.random.uniform(w_key, (in_dim, out_dim), minval=-limit, maxval=limit, dtype=jnp.float32)
    bias = jnp.zeros((out_dim,), dtype=jnp.float32)
    return {"w": weight, "b": bias}


def _linear(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    return jnp.einsum("...d,df->...f", x, params["w"]) + params["b"]


def _silu(x: jax.Array) -> jax.Array:
    return x * jax.nn.sigmoid(x)


def _sinusoidal_embedding(timesteps: jax.Array, dim: int) -> jax.Array:
    half_dim = dim // 2
    exponent = -math.log(10_000.0) * jnp.arange(half_dim, dtype=jnp.float32) / max(half_dim, 1)
    freq = jnp.exp(exponent)
    args = timesteps.astype(jnp.float32)[:, None] * freq[None, :]
    emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=1)
    if dim % 2 == 1:
        emb = jnp.concatenate([emb, jnp.zeros_like(emb[:, :1])], axis=1)
    return emb


def init_model_params(rng_key: jax.Array, config: ModelConfig) -> dict[str, object]:
    if config.pose_dim <= 0:
        raise ValueError("pose_dim must be positive.")
    keys = jax.random.split(rng_key, 8)
    pooled_dim = config.context_dim * 2
    input_dim = config.pose_dim + pooled_dim + config.time_embed_dim
    return {
        "scene_encoder": {
            "point_fc1": _init_linear(keys[0], config.point_feature_dim, config.hidden_dim),
            "point_fc2": _init_linear(keys[1], config.hidden_dim, config.context_dim),
        },
        "denoiser": {
            "input": _init_linear(keys[2], input_dim, config.hidden_dim),
            "hidden_0": _init_linear(keys[3], config.hidden_dim, config.hidden_dim),
            "hidden_1": _init_linear(keys[4], config.hidden_dim, config.hidden_dim),
            "hidden_2": _init_linear(keys[5], config.hidden_dim, config.hidden_dim),
            "hidden_3": _init_linear(keys[6], config.hidden_dim, config.hidden_dim),
            "output": _init_linear(keys[7], config.hidden_dim, config.pose_dim),
        },
    }


def encode_scene(
    params: dict[str, object],
    object_points: jax.Array,
    object_normals: jax.Array,
) -> jax.Array:
    features = jnp.concatenate([object_points, object_normals], axis=-1)
    hidden = _silu(_linear(params["point_fc1"], features))
    hidden = _silu(_linear(params["point_fc2"], hidden))
    pooled_mean = jnp.mean(hidden, axis=1)
    pooled_max = jnp.max(hidden, axis=1)
    return jnp.concatenate([pooled_mean, pooled_max], axis=-1)


def apply_model(
    params: dict[str, object],
    noisy_pose: jax.Array,
    timesteps: jax.Array,
    object_points: jax.Array,
    object_normals: jax.Array,
    config: ModelConfig,
    *,
    rng_key: jax.Array | None = None,
    training: bool = True,
    condition_override: jax.Array | None = None,
) -> jax.Array:
    del rng_key, training
    scene_feature = (
        encode_scene(params["scene_encoder"], object_points, object_normals)
        if condition_override is None
        else condition_override
    )
    time_feature = _sinusoidal_embedding(timesteps, config.time_embed_dim)
    hidden = jnp.concatenate([noisy_pose, scene_feature, time_feature], axis=-1)
    hidden = _silu(_linear(params["denoiser"]["input"], hidden))
    hidden = hidden + _silu(_linear(params["denoiser"]["hidden_0"], hidden))
    hidden = hidden + _silu(_linear(params["denoiser"]["hidden_1"], hidden))
    hidden = hidden + _silu(_linear(params["denoiser"]["hidden_2"], hidden))
    hidden = hidden + _silu(_linear(params["denoiser"]["hidden_3"], hidden))
    return _linear(params["denoiser"]["output"], hidden)
