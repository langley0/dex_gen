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
from grasp_refine.model_dga import condition, init_model_params
from grasp_refine.training_types import ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the DGA PointTransformer scene encoder.")
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

    config = ModelConfig(
        architecture="dga_unet",
        pose_dim=int(dataset.arrays.pose.shape[1]),
        hidden_dim=512,
        context_dim=int(args.context_dim),
        num_heads=8,
        transformer_dim_head=64,
    )
    rng_key = jax.random.key(np.uint32(int(args.seed) % (2**32)))
    params = init_model_params(rng_key, config)
    context = condition(
        params,
        jnp.asarray(object_points, dtype=jnp.float32),
        jnp.asarray(object_normals, dtype=jnp.float32),
        config,
    )
    context_np = np.asarray(jax.device_get(context), dtype=np.float32)
    expected_tokens = int(object_points.shape[1] // 256)

    print(f"dataset path          : {dataset.path}")
    print(f"batch size            : {batch_size}")
    print(f"object points shape   : {object_points.shape}")
    print(f"context shape         : {context_np.shape}")
    print(f"expected token count  : {expected_tokens}")
    print(f"context dim           : {context_np.shape[-1]}")

    if context_np.shape[1] != expected_tokens:
        raise SystemExit(f"Unexpected token count: got {context_np.shape[1]}, expected {expected_tokens}")
    if context_np.shape[2] != int(args.context_dim):
        raise SystemExit(f"Unexpected context dim: got {context_np.shape[2]}, expected {args.context_dim}")


if __name__ == "__main__":
    main()
