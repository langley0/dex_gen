#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_refine import load_saved_dga_dataset
from grasp_refine.scene_encoder_dga import (
    DGA_SCENE_ENCODER_BLOCKS,
    DGA_SCENE_ENCODER_NSAMPLES,
    DGA_SCENE_ENCODER_PLANES,
    DGA_SCENE_ENCODER_SHARE_PLANES,
    DGA_SCENE_ENCODER_STRIDES,
    encode_scene,
    init_scene_encoder_params,
)


def _assert_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise SystemExit(f"{name} mismatch: got {actual!r}, expected {expected!r}")


def _stage_output_channels(stage: dict[str, object]) -> int:
    linear = stage["transition"]["linear"]
    return int(linear["w"].shape[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit JAX scene encoder against DGA PointTransformer design.")
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared normalized DGA dataset (.npz).")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-dim", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_saved_dga_dataset(args.dataset)
    batch_size = min(int(args.batch_size), len(dataset))
    object_points = np.stack(
        [np.asarray(dataset[index]["object_points"], dtype=np.float32) for index in range(batch_size)],
        axis=0,
    )
    object_normals = np.stack(
        [np.asarray(dataset[index]["object_normals"], dtype=np.float32) for index in range(batch_size)],
        axis=0,
    )

    rng_key = jax.random.key(np.uint32(int(args.seed) % (2**32)))
    params = init_scene_encoder_params(
        rng_key,
        point_feature_dim=int(object_points.shape[-1] + object_normals.shape[-1]),
        context_dim=int(args.context_dim),
    )

    planes = DGA_SCENE_ENCODER_PLANES[:-1] + (int(args.context_dim),)
    _assert_equal("stage_count", len(params["stages"]), len(planes))
    _assert_equal("blocks", DGA_SCENE_ENCODER_BLOCKS, (2, 3, 4, 6, 3))
    _assert_equal("strides", DGA_SCENE_ENCODER_STRIDES, (1, 4, 4, 4, 4))
    _assert_equal("nsamples", DGA_SCENE_ENCODER_NSAMPLES, (8, 16, 16, 16, 16))
    _assert_equal("share_planes", DGA_SCENE_ENCODER_SHARE_PLANES, 8)

    stage_channels = tuple(_stage_output_channels(stage) for stage in params["stages"])
    _assert_equal("stage_channels", stage_channels, planes)
    stage_block_counts = tuple(1 + len(stage["blocks"]) for stage in params["stages"])
    _assert_equal("stage_block_counts", stage_block_counts, DGA_SCENE_ENCODER_BLOCKS)

    points = jnp.asarray(object_points, dtype=jnp.float32)
    normals = jnp.asarray(object_normals, dtype=jnp.float32)
    context = encode_scene(params, points, normals)
    context_np = np.asarray(jax.device_get(context), dtype=np.float32)
    expected_tokens = int(object_points.shape[1] // np.prod(DGA_SCENE_ENCODER_STRIDES))

    print(f"dataset path          : {dataset.path}")
    print(f"batch size            : {batch_size}")
    print(f"planes                : {planes}")
    print(f"blocks                : {DGA_SCENE_ENCODER_BLOCKS}")
    print(f"strides               : {DGA_SCENE_ENCODER_STRIDES}")
    print(f"nsamples              : {DGA_SCENE_ENCODER_NSAMPLES}")
    print(f"share planes          : {DGA_SCENE_ENCODER_SHARE_PLANES}")
    print(f"context shape         : {context_np.shape}")
    print(f"expected tokens       : {expected_tokens}")

    if context_np.shape[1] != expected_tokens:
        raise SystemExit(f"token_count mismatch: got {context_np.shape[1]}, expected {expected_tokens}")
    if context_np.shape[2] != int(args.context_dim):
        raise SystemExit(f"context_dim mismatch: got {context_np.shape[2]}, expected {args.context_dim}")

    print("status                : PASS")
    print("parity verdict        : core PointTransformer encoder layout matches DGA design")
    print("remaining difference  : JAX implementation replaces PyTorch/pointops kernels")


if __name__ == "__main__":
    main()
