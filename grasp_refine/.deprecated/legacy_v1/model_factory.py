from __future__ import annotations

import jax

from .model import apply_model as apply_model_mlp
from .model import init_model_params as init_model_params_mlp
from .model_dga import apply_model as apply_model_dga
from .model_dga import init_model_params as init_model_params_dga
from .types import ModelConfig


def init_model_params(rng_key: jax.Array, config: ModelConfig) -> dict[str, object]:
    if config.architecture == "mlp":
        return init_model_params_mlp(rng_key, config)
    if config.architecture == "dga_transformer":
        return init_model_params_dga(rng_key, config)
    raise ValueError(f"Unsupported model architecture: {config.architecture!r}")


def apply_model(
    params: dict[str, object],
    noisy_pose: jax.Array,
    timesteps: jax.Array,
    object_points: jax.Array,
    object_normals: jax.Array,
    config: ModelConfig,
    object_index: jax.Array | None = None,
) -> jax.Array:
    if config.architecture == "mlp":
        return apply_model_mlp(
            params,
            noisy_pose,
            timesteps,
            object_points,
            object_normals,
            config,
            object_index=object_index,
        )
    if config.architecture == "dga_transformer":
        return apply_model_dga(
            params,
            noisy_pose,
            timesteps,
            object_points,
            object_normals,
            config,
            object_index=object_index,
        )
    raise ValueError(f"Unsupported model architecture: {config.architecture!r}")
