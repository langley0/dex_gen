from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import lax

from .types import ModelConfig


def _init_linear(key: jax.Array, in_dim: int, out_dim: int) -> dict[str, jax.Array]:
    limit = math.sqrt(6.0 / float(in_dim + out_dim))
    weight = jax.random.uniform(key, (in_dim, out_dim), minval=-limit, maxval=limit, dtype=jnp.float32)
    bias = jnp.zeros((out_dim,), dtype=jnp.float32)
    return {"w": weight, "b": bias}


def _init_layer_norm(dim: int) -> dict[str, jax.Array]:
    return {
        "scale": jnp.ones((dim,), dtype=jnp.float32),
        "bias": jnp.zeros((dim,), dtype=jnp.float32),
    }


def _linear(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    return jnp.einsum("...d,df->...f", x, params["w"]) + params["b"]


def _silu(x: jax.Array) -> jax.Array:
    return x * jax.nn.sigmoid(x)


def _layer_norm(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    normalized = (x - mean) * jax.lax.rsqrt(var + 1.0e-5)
    return normalized * params["scale"] + params["bias"]


def _sinusoidal_embedding(timesteps: jax.Array, dim: int) -> jax.Array:
    half_dim = dim // 2
    exponent = -math.log(10_000.0) * jnp.arange(half_dim, dtype=jnp.float32) / max(half_dim, 1)
    freq = jnp.exp(exponent)
    args = timesteps.astype(jnp.float32)[:, None] * freq[None, :]
    emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=1)
    if dim % 2 == 1:
        emb = jnp.concatenate([emb, jnp.zeros_like(emb[:, :1])], axis=1)
    return emb


def _split_heads(x: jax.Array, num_heads: int) -> jax.Array:
    batch, tokens, channels = x.shape
    head_dim = channels // num_heads
    return x.reshape(batch, tokens, num_heads, head_dim).transpose(0, 2, 1, 3)


def _merge_heads(x: jax.Array) -> jax.Array:
    batch, num_heads, tokens, head_dim = x.shape
    return x.transpose(0, 2, 1, 3).reshape(batch, tokens, num_heads * head_dim)


def _attention(q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
    head_dim = q.shape[-1]
    logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(float(head_dim))
    weights = jax.nn.softmax(logits, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", weights, v)


def _init_transformer_block(keys: list[jax.Array], hidden_dim: int) -> dict[str, object]:
    ff_dim = hidden_dim * 2
    return {
        "attn_norm": _init_layer_norm(hidden_dim),
        "attn_q": _init_linear(keys[0], hidden_dim, hidden_dim),
        "attn_k": _init_linear(keys[1], hidden_dim, hidden_dim),
        "attn_v": _init_linear(keys[2], hidden_dim, hidden_dim),
        "attn_out": _init_linear(keys[3], hidden_dim, hidden_dim),
        "ff_norm": _init_layer_norm(hidden_dim),
        "ff_1": _init_linear(keys[4], hidden_dim, ff_dim),
        "ff_2": _init_linear(keys[5], ff_dim, hidden_dim),
    }


def _init_denoiser_block(keys: list[jax.Array], hidden_dim: int, context_dim: int) -> dict[str, object]:
    ff_dim = hidden_dim * 2
    return {
        "time_norm": _init_layer_norm(hidden_dim),
        "time_proj": _init_linear(keys[0], hidden_dim, hidden_dim),
        "res_1": _init_linear(keys[1], hidden_dim, ff_dim),
        "res_2": _init_linear(keys[2], ff_dim, hidden_dim),
        "cross_norm": _init_layer_norm(hidden_dim),
        "cross_q": _init_linear(keys[3], hidden_dim, hidden_dim),
        "cross_k": _init_linear(keys[4], context_dim, hidden_dim),
        "cross_v": _init_linear(keys[5], context_dim, hidden_dim),
        "cross_out": _init_linear(keys[6], hidden_dim, hidden_dim),
        "ff_norm": _init_layer_norm(hidden_dim),
        "ff_1": _init_linear(keys[7], hidden_dim, ff_dim),
        "ff_2": _init_linear(keys[8], ff_dim, hidden_dim),
    }


def _self_attention_block(params: dict[str, object], x: jax.Array, num_heads: int) -> jax.Array:
    residual = x
    norm_x = _layer_norm(params["attn_norm"], x)
    q = _split_heads(_linear(params["attn_q"], norm_x), num_heads)
    k = _split_heads(_linear(params["attn_k"], norm_x), num_heads)
    v = _split_heads(_linear(params["attn_v"], norm_x), num_heads)
    attn_out = _merge_heads(_attention(q, k, v))
    x = residual + _linear(params["attn_out"], attn_out)
    ff_in = _layer_norm(params["ff_norm"], x)
    ff_out = _linear(params["ff_2"], _silu(_linear(params["ff_1"], ff_in)))
    return x + ff_out


def _cross_attention_block(
    params: dict[str, object],
    x: jax.Array,
    context: jax.Array,
    num_heads: int,
) -> jax.Array:
    residual = x
    norm_x = _layer_norm(params["cross_norm"], x)
    q = _split_heads(_linear(params["cross_q"], norm_x), num_heads)
    k = _split_heads(_linear(params["cross_k"], context), num_heads)
    v = _split_heads(_linear(params["cross_v"], context), num_heads)
    attn_out = _merge_heads(_attention(q, k, v))
    return residual + _linear(params["cross_out"], attn_out)


def _feed_forward_block(params: dict[str, object], x: jax.Array) -> jax.Array:
    ff_in = _layer_norm(params["ff_norm"], x)
    ff_out = _linear(params["ff_2"], _silu(_linear(params["ff_1"], ff_in)))
    return x + ff_out


def init_model_params(rng_key: jax.Array, config: ModelConfig) -> dict[str, object]:
    if config.hidden_dim % config.num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads for dga_transformer.")
    if config.context_tokens <= 0:
        raise ValueError("context_tokens must be positive for dga_transformer.")
    total_keys = 7 + config.scene_encoder_layers * 6 + config.denoiser_blocks * 9
    keys = list(jax.random.split(rng_key, total_keys))
    cursor = 0

    scene_encoder = {
        "point_in": _init_linear(keys[cursor], config.point_feature_dim, config.hidden_dim),
        "point_out": _init_linear(keys[cursor + 1], config.hidden_dim, config.context_dim),
        "token_queries": jax.random.normal(
            keys[cursor + 2],
            (config.context_tokens, config.context_dim),
            dtype=jnp.float32,
        ) / math.sqrt(float(config.context_dim)),
        "token_norm": _init_layer_norm(config.context_dim),
    }
    cursor += 3
    scene_blocks = []
    for _ in range(config.scene_encoder_layers):
        block_keys = keys[cursor : cursor + 6]
        scene_blocks.append(_init_transformer_block(block_keys, config.context_dim))
        cursor += 6

    denoiser = {
        "input_proj": _init_linear(keys[cursor], config.pose_dim, config.hidden_dim),
        "time_mlp_1": _init_linear(keys[cursor + 1], config.time_embed_dim, config.hidden_dim),
        "time_mlp_2": _init_linear(keys[cursor + 2], config.hidden_dim, config.hidden_dim),
        "output_norm": _init_layer_norm(config.hidden_dim),
        "output_proj": _init_linear(keys[cursor + 3], config.hidden_dim, config.pose_dim),
    }
    cursor += 4
    denoiser_blocks = []
    for _ in range(config.denoiser_blocks):
        block_keys = keys[cursor : cursor + 9]
        denoiser_blocks.append(_init_denoiser_block(block_keys, config.hidden_dim, config.context_dim))
        cursor += 9

    return {
        "scene_encoder": scene_encoder,
        "scene_blocks": tuple(scene_blocks),
        "denoiser": denoiser,
        "denoiser_blocks": tuple(denoiser_blocks),
    }


def encode_scene(
    params: dict[str, object],
    object_points: jax.Array,
    object_normals: jax.Array,
    config: ModelConfig,
) -> jax.Array:
    features = jnp.concatenate([object_points, object_normals], axis=-1)
    point_hidden = _silu(_linear(params["scene_encoder"]["point_in"], features))
    point_hidden = _silu(_linear(params["scene_encoder"]["point_out"], point_hidden))
    logits = jnp.einsum("td,bnd->btn", params["scene_encoder"]["token_queries"], point_hidden)
    logits = logits / math.sqrt(float(config.context_dim))
    weights = jax.nn.softmax(logits, axis=-1)
    tokens = jnp.einsum("btn,bnd->btd", weights, point_hidden)
    tokens = _layer_norm(params["scene_encoder"]["token_norm"], tokens)
    for block in params["scene_blocks"]:
        tokens = _self_attention_block(block, tokens, config.num_heads)
    return tokens


def _apply_denoiser(
    params: dict[str, object],
    noisy_pose: jax.Array,
    timesteps: jax.Array,
    scene_tokens: jax.Array,
    config: ModelConfig,
) -> jax.Array:
    time_feature = _sinusoidal_embedding(timesteps, config.time_embed_dim)
    time_feature = _silu(_linear(params["denoiser"]["time_mlp_1"], time_feature))
    time_feature = _silu(_linear(params["denoiser"]["time_mlp_2"], time_feature))
    hidden = _linear(params["denoiser"]["input_proj"], noisy_pose)[:, None, :]
    for block in params["denoiser_blocks"]:
        residual = hidden
        time_residual = _layer_norm(block["time_norm"], hidden) + _linear(block["time_proj"], time_feature)[:, None, :]
        time_residual = _linear(block["res_2"], _silu(_linear(block["res_1"], time_residual)))
        hidden = residual + time_residual
        hidden = _cross_attention_block(block, hidden, scene_tokens, config.num_heads)
        hidden = _feed_forward_block(block, hidden)
    hidden = _layer_norm(params["denoiser"]["output_norm"], hidden)
    return _linear(params["denoiser"]["output_proj"], hidden[:, 0, :])


def apply_model(
    params: dict[str, object],
    noisy_pose: jax.Array,
    timesteps: jax.Array,
    object_points: jax.Array,
    object_normals: jax.Array,
    config: ModelConfig,
    object_index: jax.Array | None = None,
) -> jax.Array:
    if object_index is None:
        scene_tokens = encode_scene(params, object_points, object_normals, config)
    else:
        def _encode_shared(_: None) -> jax.Array:
            shared = encode_scene(params, object_points[:1], object_normals[:1], config)
            return jnp.broadcast_to(shared, (object_points.shape[0],) + shared.shape[1:])

        def _encode_per_sample(_: None) -> jax.Array:
            return encode_scene(params, object_points, object_normals, config)

        scene_tokens = lax.cond(
            jnp.all(object_index == object_index[:1]),
            _encode_shared,
            _encode_per_sample,
            operand=None,
        )
    return _apply_denoiser(params, noisy_pose, timesteps, scene_tokens, config)
