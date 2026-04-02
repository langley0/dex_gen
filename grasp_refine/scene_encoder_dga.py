from __future__ import annotations

import math

import jax
import jax.numpy as jnp


def _init_linear(key: jax.Array, in_dim: int, out_dim: int, *, bias: bool = True) -> dict[str, jax.Array]:
    limit = math.sqrt(6.0 / float(in_dim + out_dim))
    weight = jax.random.uniform(key, (in_dim, out_dim), minval=-limit, maxval=limit, dtype=jnp.float32)
    bias_value = jnp.zeros((out_dim,), dtype=jnp.float32) if bias else jnp.zeros((out_dim,), dtype=jnp.float32)
    return {"w": weight, "b": bias_value}


def _init_batch_norm(channels: int) -> dict[str, jax.Array]:
    return {
        "scale": jnp.ones((channels,), dtype=jnp.float32),
        "bias": jnp.zeros((channels,), dtype=jnp.float32),
    }


def _linear(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    return jnp.einsum("...d,df->...f", x, params["w"]) + params["b"]


def _batch_norm_last_dim(params: dict[str, jax.Array], x: jax.Array, eps: float = 1.0e-5) -> jax.Array:
    reduce_axes = tuple(range(x.ndim - 1))
    mean = jnp.mean(x, axis=reduce_axes, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=reduce_axes, keepdims=True)
    normalized = (x - mean) * jax.lax.rsqrt(var + eps)
    return normalized * params["scale"] + params["bias"]


def _relu(x: jax.Array) -> jax.Array:
    return jax.nn.relu(x)


def _gather_points(points: jax.Array, indices: jax.Array) -> jax.Array:
    batch_indices = jnp.arange(points.shape[0], dtype=jnp.int32)[:, None]
    return points[batch_indices, indices]


def _gather_neighbors(values: jax.Array, indices: jax.Array) -> jax.Array:
    batch_indices = jnp.arange(values.shape[0], dtype=jnp.int32)[:, None, None]
    return values[batch_indices, indices]


def _pairwise_sqdist(query: jax.Array, key: jax.Array) -> jax.Array:
    diff = query[:, :, None, :] - key[:, None, :, :]
    return jnp.sum(jnp.square(diff), axis=-1)


def _knn_indices(query: jax.Array, key: jax.Array, k: int) -> jax.Array:
    num_key = int(key.shape[1])
    effective_k = max(1, min(int(k), num_key))
    distances = _pairwise_sqdist(query, key)
    _, indices = jax.lax.top_k(-distances, effective_k)
    return indices.astype(jnp.int32)


def _farthest_point_sample(points: jax.Array, sample_count: int) -> jax.Array:
    batch_size, num_points, _ = points.shape
    sample_count = max(1, min(int(sample_count), int(num_points)))
    initial_farthest = jnp.zeros((batch_size,), dtype=jnp.int32)
    initial_distance = jnp.full((batch_size, num_points), jnp.inf, dtype=points.dtype)
    initial_centroids = jnp.zeros((batch_size, sample_count), dtype=jnp.int32)

    def body(sample_index: int, state: tuple[jax.Array, jax.Array, jax.Array]):
        farthest, min_distance, centroids = state
        centroids = centroids.at[:, sample_index].set(farthest)
        centroid = _gather_points(points, farthest[:, None])[:, 0, :]
        distance = jnp.sum(jnp.square(points - centroid[:, None, :]), axis=-1)
        min_distance = jnp.minimum(min_distance, distance)
        farthest = jnp.argmax(min_distance, axis=1).astype(jnp.int32)
        return farthest, min_distance, centroids

    _, _, centroids = jax.lax.fori_loop(
        0,
        sample_count,
        body,
        (initial_farthest, initial_distance, initial_centroids),
    )
    return centroids


def _init_transition_down(
    key: jax.Array,
    *,
    in_planes: int,
    out_planes: int,
    stride: int,
) -> dict[str, object]:
    linear_in_dim = int(in_planes) if int(stride) == 1 else 3 + int(in_planes)
    return {
        "linear": _init_linear(key, linear_in_dim, out_planes, bias=False),
        "bn": _init_batch_norm(out_planes),
    }


def _transition_down(
    params: dict[str, object],
    points: jax.Array,
    features: jax.Array,
    *,
    stride: int,
    nsample: int,
) -> tuple[jax.Array, jax.Array]:
    if stride == 1:
        projected = _linear(params["linear"], features)
        projected = _relu(_batch_norm_last_dim(params["bn"], projected))
        return points, projected

    sample_count = max(1, int(points.shape[1]) // stride)
    sample_indices = _farthest_point_sample(points, sample_count)
    sampled_points = _gather_points(points, sample_indices)
    neighbor_indices = _knn_indices(sampled_points, points, nsample)
    grouped_points = _gather_neighbors(points, neighbor_indices)
    grouped_features = _gather_neighbors(features, neighbor_indices)
    relative_points = grouped_points - sampled_points[:, :, None, :]
    grouped = jnp.concatenate([relative_points, grouped_features], axis=-1)
    projected = _linear(params["linear"], grouped)
    projected = _relu(_batch_norm_last_dim(params["bn"], projected))
    pooled = jnp.max(projected, axis=2)
    return sampled_points, pooled


def _init_point_transformer_layer(
    keys: list[jax.Array],
    *,
    in_planes: int,
    out_planes: int,
    share_planes: int,
) -> dict[str, object]:
    reduced_dim = out_planes // int(share_planes)
    return {
        "linear_q": _init_linear(keys[0], in_planes, out_planes),
        "linear_k": _init_linear(keys[1], in_planes, out_planes),
        "linear_v": _init_linear(keys[2], in_planes, out_planes),
        "linear_p1": _init_linear(keys[3], 3, 3),
        "linear_p2": _init_linear(keys[4], 3, out_planes),
        "linear_w1": _init_linear(keys[5], out_planes, reduced_dim),
        "linear_w2": _init_linear(keys[6], reduced_dim, reduced_dim),
        "bn_p": _init_batch_norm(3),
        "bn_w1": _init_batch_norm(out_planes),
        "bn_w2": _init_batch_norm(reduced_dim),
    }


def _point_transformer_layer(
    params: dict[str, object],
    points: jax.Array,
    features: jax.Array,
    *,
    nsample: int,
) -> jax.Array:
    query = _linear(params["linear_q"], features)
    key = _linear(params["linear_k"], features)
    value = _linear(params["linear_v"], features)
    neighbor_indices = _knn_indices(points, points, nsample)
    grouped_key = _gather_neighbors(key, neighbor_indices)
    grouped_value = _gather_neighbors(value, neighbor_indices)
    grouped_points = _gather_neighbors(points, neighbor_indices)
    relative_points = grouped_points - points[:, :, None, :]

    positional = _linear(params["linear_p1"], relative_points)
    positional = _relu(_batch_norm_last_dim(params["bn_p"], positional))
    positional = _linear(params["linear_p2"], positional)

    weights = grouped_key - query[:, :, None, :] + positional
    weights = _relu(_batch_norm_last_dim(params["bn_w1"], weights))
    weights = _linear(params["linear_w1"], weights)
    weights = _relu(_batch_norm_last_dim(params["bn_w2"], weights))
    weights = _linear(params["linear_w2"], weights)
    weights = jax.nn.softmax(weights, axis=2)

    share_planes = 8
    out_planes = int(grouped_value.shape[-1])
    grouped = grouped_value + positional
    grouped = grouped.reshape(grouped.shape[0], grouped.shape[1], grouped.shape[2], share_planes, out_planes // share_planes)
    weighted = grouped * weights[:, :, :, None, :]
    return weighted.sum(axis=2).reshape(features.shape[0], features.shape[1], out_planes)


def _init_point_transformer_block(
    keys: list[jax.Array],
    *,
    in_planes: int,
    planes: int,
    share_planes: int,
) -> dict[str, object]:
    return {
        "linear1": _init_linear(keys[0], in_planes, planes, bias=False),
        "bn1": _init_batch_norm(planes),
        "transformer2": _init_point_transformer_layer(
            keys[1:8],
            in_planes=planes,
            out_planes=planes,
            share_planes=share_planes,
        ),
        "bn2": _init_batch_norm(planes),
        "linear3": _init_linear(keys[8], planes, planes, bias=False),
        "bn3": _init_batch_norm(planes),
    }


def _point_transformer_block(
    params: dict[str, object],
    points: jax.Array,
    features: jax.Array,
    *,
    nsample: int,
) -> jax.Array:
    identity = features
    hidden = _relu(_batch_norm_last_dim(params["bn1"], _linear(params["linear1"], features)))
    hidden = _relu(_batch_norm_last_dim(params["bn2"], _point_transformer_layer(params["transformer2"], points, hidden, nsample=nsample)))
    hidden = _batch_norm_last_dim(params["bn3"], _linear(params["linear3"], hidden))
    return _relu(hidden + identity)


def _init_encoder_stage(
    key: jax.Array,
    *,
    in_planes: int,
    planes: int,
    blocks: int,
    share_planes: int,
    stride: int,
) -> tuple[dict[str, object], int]:
    key_count = 1 + max(0, int(blocks) - 1) * 9
    keys = list(jax.random.split(key, key_count))
    transition = _init_transition_down(keys[0], in_planes=in_planes, out_planes=planes, stride=stride)
    block_params = []
    cursor = 1
    for _ in range(1, int(blocks)):
        block_params.append(
            _init_point_transformer_block(
                keys[cursor : cursor + 9],
                in_planes=planes,
                planes=planes,
                share_planes=share_planes,
            )
        )
        cursor += 9
    return {
        "transition": transition,
        "blocks": tuple(block_params),
    }, int(planes)


def init_scene_encoder_params(
    rng_key: jax.Array,
    *,
    point_feature_dim: int,
    context_dim: int,
) -> dict[str, object]:
    planes = [32, 64, 128, 256, int(context_dim)]
    blocks = [2, 3, 4, 6, 3]
    strides = [1, 4, 4, 4, 4]
    nsamples = [8, 16, 16, 16, 16]
    share_planes = 8

    stage_keys = list(jax.random.split(rng_key, len(planes)))
    in_planes = int(point_feature_dim)
    stages = []
    for stage_key, plane, block_count, stride, nsample in zip(stage_keys, planes, blocks, strides, nsamples, strict=True):
        stage, in_planes = _init_encoder_stage(
            stage_key,
            in_planes=in_planes,
            planes=plane,
            blocks=block_count,
            share_planes=share_planes,
            stride=stride,
        )
        stages.append(stage)
    return {
        "stages": tuple(stages),
    }


def encode_scene(
    params: dict[str, object],
    object_points: jax.Array,
    object_normals: jax.Array,
) -> jax.Array:
    strides = (1, 4, 4, 4, 4)
    nsamples = (8, 16, 16, 16, 16)
    features = jnp.concatenate([object_points, object_normals], axis=-1)
    points = object_points
    for stage, stride, nsample in zip(params["stages"], strides, nsamples, strict=True):
        points, features = _transition_down(
            stage["transition"],
            points,
            features,
            stride=stride,
            nsample=nsample,
        )
        for block in stage["blocks"]:
            features = _point_transformer_block(
                block,
                points,
                features,
                nsample=nsample,
            )
    return features
