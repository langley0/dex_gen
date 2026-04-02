from __future__ import annotations

import jax

from .model import encode_scene as encode_scene_mlp
from .model import apply_model as apply_model_mlp
from .model import init_model_params as init_model_params_mlp
from .model_dga import condition as condition_model_dga
from .model_dga import apply_model as apply_model_dga
from .model_dga import init_model_params as init_model_params_dga
from .scene_encoder_pretrained import load_scene_encoder_pretrained_params
from .training_types import ModelConfig


def init_model_params(rng_key: jax.Array, config: ModelConfig) -> dict[str, object]:
    if config.architecture == "mlp":
        return init_model_params_mlp(rng_key, config)
    if config.architecture in ("dga_unet", "dga_transformer"):
        params = init_model_params_dga(rng_key, config)
        if config.scene_encoder_pretrained is not None:
            params = dict(params)
            params["scene_model"] = load_scene_encoder_pretrained_params(config.scene_encoder_pretrained, params["scene_model"])
        return params
    raise ValueError(f"Unsupported model architecture: {config.architecture!r}")


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
    if config.architecture == "mlp":
        return apply_model_mlp(
            params,
            noisy_pose,
            timesteps,
            object_points,
            object_normals,
            config,
            rng_key=rng_key,
            training=training,
            condition_override=condition_override,
        )
    if config.architecture in ("dga_unet", "dga_transformer"):
        return apply_model_dga(
            params,
            noisy_pose,
            timesteps,
            object_points,
            object_normals,
            config,
            rng_key=rng_key,
            training=training,
            condition_override=condition_override,
        )
    raise ValueError(f"Unsupported model architecture: {config.architecture!r}")


def condition_model(
    params: dict[str, object],
    object_points: jax.Array,
    object_normals: jax.Array,
    config: ModelConfig,
) -> jax.Array:
    if config.architecture == "mlp":
        return encode_scene_mlp(params["scene_encoder"], object_points, object_normals)
    if config.architecture in ("dga_unet", "dga_transformer"):
        return condition_model_dga(params, object_points, object_normals, config)
    raise ValueError(f"Unsupported model architecture: {config.architecture!r}")
