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

from grasp_refine.diffusion import diffusion_loss, make_diffusion_schedule
from grasp_refine.hand_points import load_dga_hand_point_spec
from grasp_refine.loader import iterate_dga_batches, load_saved_dga_dataset
from grasp_refine.model_factory import init_model_params
from grasp_refine.optim import adam_update, init_adam
from grasp_refine.training_types import DiffusionConfig, LossConfig, ModelConfig, OptimizerConfig, PreparedDatasetConfig, TrainingConfig


def _global_norm(tree) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    squared = [jnp.sum(jnp.square(leaf)) for leaf in leaves]
    return jnp.sqrt(jnp.sum(jnp.stack(squared)))


def _clip_grads(grads, max_norm: float):
    if max_norm <= 0.0:
        return grads
    norm = _global_norm(grads)
    scale = jnp.minimum(1.0, float(max_norm) / (norm + 1.0e-8))
    return jax.tree_util.tree_map(lambda value: value * scale, grads)


def _batch_dict(batch) -> dict[str, jax.Array]:
    return {
        "pose": jnp.asarray(batch.pose, dtype=jnp.float32),
        "pose_raw": jnp.asarray(batch.pose_raw, dtype=jnp.float32),
        "object_points": jnp.asarray(batch.object_points, dtype=jnp.float32),
        "object_normals": jnp.asarray(batch.object_normals, dtype=jnp.float32),
    }


def _shard_batch(batch, num_devices: int) -> dict[str, jax.Array]:
    batch_size = int(batch.pose.shape[0])
    if batch_size % int(num_devices) != 0:
        raise SystemExit(f"batch_size must be divisible by num_devices, got {batch_size} and {num_devices}")
    per_device = batch_size // int(num_devices)

    def shard(array: np.ndarray, dtype: jnp.dtype) -> jax.Array:
        host = np.asarray(array, dtype=np.dtype(dtype))
        return jnp.asarray(host.reshape((int(num_devices), per_device) + host.shape[1:]), dtype=dtype)

    return {
        "pose": shard(batch.pose, jnp.float32),
        "pose_raw": shard(batch.pose_raw, jnp.float32),
        "object_points": shard(batch.object_points, jnp.float32),
        "object_normals": shard(batch.object_normals, jnp.float32),
    }


def _tree_add(lhs, rhs):
    return jax.tree_util.tree_map(lambda a, b: a + b, lhs, rhs)


def _tree_div(tree, denom: float):
    return jax.tree_util.tree_map(lambda value: value / float(denom), tree)


def _tree_max_abs(lhs, rhs) -> float:
    leaves_l = jax.tree_util.tree_leaves(lhs)
    leaves_r = jax.tree_util.tree_leaves(rhs)
    diffs = [float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(leaves_l, leaves_r, strict=True)]
    return max(diffs) if diffs else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare manual shard-averaged update with JAX pmap data-parallel update.")
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared normalized DGA dataset (.npz).")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3.0e-5)
    parser.add_argument("--diffusion-steps", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    device_count = int(jax.local_device_count())
    if device_count < 2:
        raise SystemExit("This smoke test requires at least 2 local JAX devices.")

    args = parse_args()
    dataset = load_saved_dga_dataset(args.dataset)
    batch = next(iterate_dga_batches(dataset, batch_size=int(args.batch_size), shuffle=False, seed=int(args.seed), drop_last=True))
    hand_point_spec = load_dga_hand_point_spec(str(batch.hand_side[0]))

    config = TrainingConfig(
        dataset=PreparedDatasetConfig(path=args.dataset.expanduser().resolve(), train_fraction=1.0, split_mode="sample"),
        model=ModelConfig(
            architecture="dga_unet",
            pose_dim=int(dataset.arrays.pose.shape[1]),
            hidden_dim=64,
            context_dim=512,
            context_tokens=8,
            denoiser_blocks=1,
            transformer_depth=1,
            num_heads=8,
            transformer_dim_head=8,
        ),
        diffusion=DiffusionConfig(steps=int(args.diffusion_steps)),
        loss=LossConfig(),
        optimizer=OptimizerConfig(lr=float(args.lr), batch_size=int(args.batch_size), epochs=1, grad_clip_norm=float(args.grad_clip_norm)),
        distributed=True,
        device="cpu",
        seed=int(args.seed),
    )

    rng_key = jax.random.key(np.uint32(int(args.seed) % (2**32)))
    params = init_model_params(rng_key, config.model)
    optimizer_state = init_adam(params)
    schedule = make_diffusion_schedule(config.diffusion)

    full_batch = _batch_dict(batch)
    sharded_batch = _shard_batch(batch, device_count)
    shard_keys = jax.random.split(rng_key, device_count)

    def loss_fn(local_params, local_batch, local_rng):
        return diffusion_loss(
            local_params,
            local_batch,
            local_rng,
            model_config=config.model,
            diffusion_config=config.diffusion,
            loss_config=config.loss,
            normalizer=dataset.normalizer,
            hand_point_spec=hand_point_spec,
            schedule=schedule,
            training=True,
        )

    shard_grads = []
    shard_metrics = []
    per_device = int(batch.pose.shape[0]) // device_count
    for device_index in range(device_count):
        local_batch = {key: value[device_index] for key, value in sharded_batch.items()}
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, local_batch, shard_keys[device_index])
        shard_grads.append(grads)
        shard_metrics.append(metrics)
    mean_grads = shard_grads[0]
    mean_metrics = shard_metrics[0]
    for grads in shard_grads[1:]:
        mean_grads = _tree_add(mean_grads, grads)
    for metrics in shard_metrics[1:]:
        mean_metrics = _tree_add(mean_metrics, metrics)
    mean_grads = _tree_div(mean_grads, device_count)
    mean_metrics = _tree_div(mean_metrics, device_count)
    mean_grads = _clip_grads(mean_grads, config.optimizer.grad_clip_norm)
    params_manual, opt_manual = adam_update(params, mean_grads, optimizer_state, lr=config.optimizer.lr)

    def train_step(params_rep, opt_rep, batch_rep, rng_rep):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_rep, batch_rep, rng_rep)
        grads = jax.lax.pmean(grads, axis_name="devices")
        metrics = jax.lax.pmean(metrics, axis_name="devices")
        grads = _clip_grads(grads, config.optimizer.grad_clip_norm)
        next_params, next_opt = adam_update(params_rep, grads, opt_rep, lr=config.optimizer.lr)
        return next_params, next_opt, metrics

    train_step_pmapped = jax.pmap(train_step, axis_name="devices")
    params_rep = jax.device_put_replicated(params, jax.devices()[:device_count])
    opt_rep = jax.device_put_replicated(optimizer_state, jax.devices()[:device_count])
    params_dist, opt_dist, metrics_dist = train_step_pmapped(params_rep, opt_rep, sharded_batch, shard_keys)
    params_dist_host = jax.tree_util.tree_map(lambda value: np.asarray(jax.device_get(value))[0], params_dist)
    opt_dist_host = jax.tree_util.tree_map(lambda value: np.asarray(jax.device_get(value))[0], opt_dist)
    metrics_dist_host = jax.tree_util.tree_map(lambda value: float(np.asarray(jax.device_get(value))[0]), metrics_dist)
    metrics_manual_host = {key: float(np.asarray(value)) for key, value in mean_metrics.items()}

    metric_diff = max(abs(metrics_manual_host[key] - metrics_dist_host[key]) for key in metrics_manual_host)
    param_diff = _tree_max_abs(params_manual, params_dist_host)
    optimizer_diff = _tree_max_abs(opt_manual, opt_dist_host)
    worst = max(metric_diff, param_diff, optimizer_diff)

    print(f"local devices        : {device_count}")
    print(f"global batch         : {int(batch.pose.shape[0])}")
    print(f"per-device batch     : {per_device}")
    print(f"metric diff          : {metric_diff:.8f}")
    print(f"param diff           : {param_diff:.8f}")
    print(f"optimizer diff       : {optimizer_diff:.8f}")
    print(f"exact status         : {'PASS' if worst < 1.0e-5 else 'FAIL'}")
    print(f"practical status     : {'PASS' if worst < 2.0e-3 else 'FAIL'}")


if __name__ == "__main__":
    main()
