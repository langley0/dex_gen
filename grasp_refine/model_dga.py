from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from .scene_encoder_dga import encode_scene, init_scene_encoder_params
from .training_types import ModelConfig


def _init_linear(key: jax.Array, in_dim: int, out_dim: int, *, bias: bool = True) -> dict[str, jax.Array]:
    limit = math.sqrt(6.0 / float(in_dim + out_dim))
    weight = jax.random.uniform(key, (in_dim, out_dim), minval=-limit, maxval=limit, dtype=jnp.float32)
    bias_value = jnp.zeros((out_dim,), dtype=jnp.float32) if bias else jnp.zeros((out_dim,), dtype=jnp.float32)
    return {"w": weight, "b": bias_value}


def _init_conv1d(key: jax.Array, in_channels: int, out_channels: int) -> dict[str, jax.Array]:
    return _init_linear(key, in_channels, out_channels)


def _init_batch_norm(channels: int) -> dict[str, jax.Array]:
    return {
        "scale": jnp.ones((channels,), dtype=jnp.float32),
        "bias": jnp.zeros((channels,), dtype=jnp.float32),
    }


def _init_layer_norm(dim: int) -> dict[str, jax.Array]:
    return {
        "scale": jnp.ones((dim,), dtype=jnp.float32),
        "bias": jnp.zeros((dim,), dtype=jnp.float32),
    }


def _init_group_norm(channels: int) -> dict[str, jax.Array]:
    return {
        "scale": jnp.ones((channels,), dtype=jnp.float32),
        "bias": jnp.zeros((channels,), dtype=jnp.float32),
    }


def _linear(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    return jnp.einsum("...d,df->...f", x, params["w"]) + params["b"]


def _conv1d(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    x_bt = jnp.swapaxes(x, 1, 2)
    y_bt = jnp.einsum("bti,io->bto", x_bt, params["w"]) + params["b"]
    return jnp.swapaxes(y_bt, 1, 2)


def _silu(x: jax.Array) -> jax.Array:
    return x * jax.nn.sigmoid(x)


def _relu(x: jax.Array) -> jax.Array:
    return jax.nn.relu(x)


def _batch_norm(params: dict[str, jax.Array], x: jax.Array, eps: float = 1.0e-5) -> jax.Array:
    mean = jnp.mean(x, axis=(0, 2), keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=(0, 2), keepdims=True)
    normalized = (x - mean) * jax.lax.rsqrt(var + eps)
    return normalized * params["scale"][None, :, None] + params["bias"][None, :, None]


def _group_norm(params: dict[str, jax.Array], x: jax.Array, groups: int = 32, eps: float = 1.0e-6) -> jax.Array:
    batch, channels, length = x.shape
    groups = min(groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    x_grouped = x.reshape(batch, groups, channels // groups, length)
    mean = jnp.mean(x_grouped, axis=(2, 3), keepdims=True)
    var = jnp.mean(jnp.square(x_grouped - mean), axis=(2, 3), keepdims=True)
    normalized = (x_grouped - mean) * jax.lax.rsqrt(var + eps)
    normalized = normalized.reshape(batch, channels, length)
    return normalized * params["scale"][None, :, None] + params["bias"][None, :, None]


def _layer_norm(params: dict[str, jax.Array], x: jax.Array, eps: float = 1.0e-5) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    normalized = (x - mean) * jax.lax.rsqrt(var + eps)
    return normalized * params["scale"] + params["bias"]


def _timestep_embedding(timesteps: jax.Array, dim: int, max_period: int = 10000) -> jax.Array:
    half = dim // 2
    freqs = jnp.exp(-math.log(float(max_period)) * jnp.arange(half, dtype=jnp.float32) / max(half, 1))
    args = timesteps[:, None].astype(jnp.float32) * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2 == 1:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding


def _split_heads(x: jax.Array, num_heads: int) -> jax.Array:
    batch, tokens, channels = x.shape
    head_dim = channels // num_heads
    return x.reshape(batch, tokens, num_heads, head_dim).transpose(0, 2, 1, 3)


def _merge_heads(x: jax.Array) -> jax.Array:
    batch, num_heads, tokens, head_dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, tokens, num_heads * head_dim)


def _dropout(x: jax.Array, rate: jax.Array | float, *, rng_key: jax.Array | None, training: bool) -> jax.Array:
    if not training:
        return x
    if rng_key is None:
        raise ValueError("rng_key is required when dropout is enabled during training.")
    rate_value = jnp.asarray(rate, dtype=x.dtype)
    keep_prob = 1.0 - rate_value
    mask = jax.random.bernoulli(rng_key, keep_prob, x.shape)
    return x * mask.astype(x.dtype) / keep_prob


def _cross_attention(
    params: dict[str, object],
    x: jax.Array,
    *,
    context: jax.Array | None,
    num_heads: int,
    rng_key: jax.Array | None,
    training: bool,
) -> jax.Array:
    query = _linear(params["to_q"], x)
    context_value = x if context is None else context
    key = _linear(params["to_k"], context_value)
    value = _linear(params["to_v"], context_value)
    q = _split_heads(query, num_heads)
    k = _split_heads(key, num_heads)
    v = _split_heads(value, num_heads)
    head_dim = q.shape[-1]
    logits = jnp.einsum("bhid,bhjd->bhij", q, k) * (head_dim ** -0.5)
    attn = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum("bhij,bhjd->bhid", attn, v)
    out = _merge_heads(out)
    out = _linear(params["to_out"], out)
    return _dropout(out, params["dropout_rate"], rng_key=rng_key, training=training)


def _feed_forward(
    params: dict[str, object],
    x: jax.Array,
    *,
    rng_key: jax.Array | None,
    training: bool,
) -> jax.Array:
    projected = _linear(params["proj_in"], x)
    value, gate = jnp.split(projected, 2, axis=-1)
    hidden = value * jax.nn.gelu(gate)
    hidden = _dropout(hidden, params["dropout_rate"], rng_key=rng_key, training=training)
    return _linear(params["proj_out"], hidden)


def _transformer_block(
    params: dict[str, object],
    x: jax.Array,
    context: jax.Array,
    config: ModelConfig,
    *,
    rng_key: jax.Array | None,
    training: bool,
) -> jax.Array:
    attn1_key, attn2_key, ff_key = (jax.random.split(rng_key, 3) if rng_key is not None else (None, None, None))
    x = _cross_attention(
        params["attn1"],
        _layer_norm(params["norm1"], x),
        context=None,
        num_heads=config.num_heads,
        rng_key=attn1_key,
        training=training,
    ) + x
    x = _cross_attention(
        params["attn2"],
        _layer_norm(params["norm2"], x),
        context=context,
        num_heads=config.num_heads,
        rng_key=attn2_key,
        training=training,
    ) + x
    x = _feed_forward(params["ff"], _layer_norm(params["norm3"], x), rng_key=ff_key, training=training) + x
    return x


def _res_block(
    params: dict[str, object],
    x: jax.Array,
    emb: jax.Array,
    *,
    rng_key: jax.Array | None,
    training: bool,
) -> jax.Array:
    h = _group_norm(params["in_norm"], x)
    h = _silu(h)
    h = _conv1d(params["in_conv"], h)
    h = h + _linear(params["emb_proj"], _silu(emb))[:, :, None]
    h = _group_norm(params["out_norm"], h)
    h = _silu(h)
    h = _dropout(h, params["dropout_rate"], rng_key=rng_key, training=training)
    h = _conv1d(params["out_conv"], h)
    skip = x if params["skip"] is None else _conv1d(params["skip"], x)
    return skip + h


def _spatial_transformer(
    params: dict[str, object],
    x: jax.Array,
    context: jax.Array,
    config: ModelConfig,
    *,
    rng_key: jax.Array | None,
    training: bool,
) -> jax.Array:
    x_in = x
    x = _group_norm(params["norm"], x)
    x = _conv1d(params["proj_in"], x)
    x = jnp.swapaxes(x, 1, 2)
    block_keys = (
        jax.random.split(rng_key, len(params["transformer_blocks"]))
        if rng_key is not None
        else (None,) * len(params["transformer_blocks"])
    )
    for block, block_key in zip(params["transformer_blocks"], block_keys, strict=True):
        x = _transformer_block(block, x, context, config, rng_key=block_key, training=training)
    x = jnp.swapaxes(x, 1, 2)
    x = _conv1d(params["proj_out"], x)
    return x + x_in


def _init_res_block(keys: list[jax.Array], channels: int, emb_channels: int) -> dict[str, object]:
    return {
        "in_norm": _init_group_norm(channels),
        "in_conv": _init_conv1d(keys[0], channels, channels),
        "emb_proj": _init_linear(keys[1], emb_channels, channels),
        "out_norm": _init_group_norm(channels),
        "out_conv": _init_conv1d(keys[2], channels, channels),
        "skip": None,
        "dropout_rate": jnp.asarray(0.0, dtype=jnp.float32),
    }


def _init_transformer_block(keys: list[jax.Array], channels: int, config: ModelConfig) -> dict[str, object]:
    inner_dim = config.num_heads * config.transformer_dim_head
    ff_dim = channels * config.transformer_mult_ff
    return {
        "attn1": {
            "to_q": _init_linear(keys[0], inner_dim, inner_dim, bias=False),
            "to_k": _init_linear(keys[1], inner_dim, inner_dim, bias=False),
            "to_v": _init_linear(keys[2], inner_dim, inner_dim, bias=False),
            "to_out": _init_linear(keys[3], inner_dim, inner_dim),
            "dropout_rate": jnp.asarray(config.transformer_dropout, dtype=jnp.float32),
        },
        "attn2": {
            "to_q": _init_linear(keys[4], inner_dim, inner_dim, bias=False),
            "to_k": _init_linear(keys[5], config.context_dim, inner_dim, bias=False),
            "to_v": _init_linear(keys[6], config.context_dim, inner_dim, bias=False),
            "to_out": _init_linear(keys[7], inner_dim, inner_dim),
            "dropout_rate": jnp.asarray(config.transformer_dropout, dtype=jnp.float32),
        },
        "ff": {
            "proj_in": _init_linear(keys[8], inner_dim, ff_dim * 2),
            "proj_out": _init_linear(keys[9], ff_dim, inner_dim),
            "dropout_rate": jnp.asarray(config.transformer_dropout, dtype=jnp.float32),
        },
        "norm1": _init_layer_norm(inner_dim),
        "norm2": _init_layer_norm(inner_dim),
        "norm3": _init_layer_norm(inner_dim),
    }


def _init_spatial_transformer(keys: list[jax.Array], channels: int, config: ModelConfig) -> dict[str, object]:
    inner_dim = config.num_heads * config.transformer_dim_head
    cursor = 0
    blocks = []
    for _ in range(config.transformer_depth):
        blocks.append(_init_transformer_block(keys[cursor : cursor + 10], channels, config))
        cursor += 10
    return {
        "norm": _init_group_norm(channels),
        "proj_in": _init_conv1d(keys[cursor], channels, inner_dim),
        "transformer_blocks": tuple(blocks),
        "proj_out": _init_conv1d(keys[cursor + 1], inner_dim, channels),
    }


def init_model_params(rng_key: jax.Array, config: ModelConfig) -> dict[str, object]:
    if config.pose_dim <= 0:
        raise ValueError("pose_dim must be positive.")
    if config.hidden_dim != config.num_heads * config.transformer_dim_head:
        raise ValueError("hidden_dim must equal num_heads * transformer_dim_head for dga_unet.")

    total_keys = 12 + config.denoiser_blocks * (3 + 10 * config.transformer_depth + 2)
    keys = list(jax.random.split(rng_key, total_keys))
    cursor = 0

    scene_model = init_scene_encoder_params(
        keys[cursor],
        point_feature_dim=config.point_feature_dim,
        context_dim=config.context_dim,
    )
    cursor += 1

    time_embed = {
        "fc1": _init_linear(keys[cursor], config.hidden_dim, config.time_embed_dim),
        "fc2": _init_linear(keys[cursor + 1], config.time_embed_dim, config.time_embed_dim),
    }
    cursor += 2

    in_layers = _init_conv1d(keys[cursor], config.pose_dim, config.hidden_dim)
    cursor += 1

    layers = []
    for _ in range(config.denoiser_blocks):
        res_block = _init_res_block(keys[cursor : cursor + 3], config.hidden_dim, config.time_embed_dim)
        res_block["dropout_rate"] = jnp.asarray(config.resblock_dropout, dtype=jnp.float32)
        cursor += 3
        spatial = _init_spatial_transformer(
            keys[cursor : cursor + (10 * config.transformer_depth + 2)],
            config.hidden_dim,
            config,
        )
        cursor += 10 * config.transformer_depth + 2
        layers.append((res_block, spatial))

    out_layers = {
        "norm": _init_group_norm(config.hidden_dim),
        "conv": _init_conv1d(keys[cursor], config.hidden_dim, config.pose_dim),
    }

    return {
        "scene_model": scene_model,
        "time_embed": time_embed,
        "in_layers": in_layers,
        "layers": tuple(layers),
        "out_layers": out_layers,
    }


def condition(
    params: dict[str, object],
    object_points: jax.Array,
    object_normals: jax.Array,
    config: ModelConfig,
) -> jax.Array:
    del config
    return encode_scene(params["scene_model"], object_points, object_normals)


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
    cond = condition(params, object_points, object_normals, config) if condition_override is None else condition_override
    in_shape = noisy_pose.ndim
    x_t = noisy_pose[:, None, :] if in_shape == 2 else noisy_pose

    t_emb = _timestep_embedding(timesteps, config.hidden_dim)
    t_emb = _silu(_linear(params["time_embed"]["fc1"], t_emb))
    t_emb = _linear(params["time_embed"]["fc2"], t_emb)

    h = jnp.swapaxes(x_t, 1, 2)
    h = _conv1d(params["in_layers"], h)

    if config.use_position_embedding:
        length = h.shape[-1]
        pos_q = jnp.arange(length, dtype=jnp.int32)
        pos_emb = _timestep_embedding(pos_q, h.shape[1]).T[None, :, :]
        h = h + pos_emb

    layer_keys = (
        jax.random.split(rng_key, len(params["layers"]) * 2)
        if rng_key is not None
        else (None,) * (len(params["layers"]) * 2)
    )
    for index, (res_block, spatial) in enumerate(params["layers"]):
        h = _res_block(
            res_block,
            h,
            t_emb,
            rng_key=layer_keys[2 * index],
            training=training,
        )
        h = _spatial_transformer(
            spatial,
            h,
            cond,
            config,
            rng_key=layer_keys[2 * index + 1],
            training=training,
        )

    h = _group_norm(params["out_layers"]["norm"], h)
    h = _silu(h)
    h = _conv1d(params["out_layers"]["conv"], h)
    h = jnp.swapaxes(h, 1, 2)
    return h[:, 0, :] if in_shape == 2 else h
