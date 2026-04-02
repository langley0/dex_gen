from __future__ import annotations

import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from .batch import GraspDatasetArrays, materialize_grasp_dataset, place_grasp_dataset_on_device
from .checkpoint import load_training_checkpoint, save_training_checkpoint
from .dataset import build_pose_normalizer, load_dataset_records, split_dataset
from .diffusion import diffusion_loss, make_diffusion_schedule
from .inspire_hand import InspireHandSpec, load_inspire_hand_spec
from .model_factory import init_model_params
from .normalization import PoseNormalizer
from .optim import AdamWState, adamw_update, init_adamw
from .types import TrainingConfig, TrainingHistory, TrainingResult


JaxDevice: TypeAlias = Any


@dataclass
class TrainingBundle:
    hand_spec: InspireHandSpec
    normalizer: PoseNormalizer
    train_dataset: object
    val_dataset: object | None
    schedule: object
    params: object
    optimizer_state: AdamWState
    device: JaxDevice


def _select_device(device_name: str) -> JaxDevice:
    try:
        return jax.devices(device_name)[0]
    except Exception:
        return jax.devices()[0]


def _set_seed(seed: int) -> jax.Array:
    np.random.seed(seed)
    return jax.random.key(np.uint32(int(seed) % (2**32)))


def build_training_bundle(config: TrainingConfig) -> TrainingBundle:
    rng_key = _set_seed(config.seed)
    device = _select_device(config.device)
    hand_spec = load_inspire_hand_spec(config.hand_side)
    records = load_dataset_records(config.dataset)
    normalizer = build_pose_normalizer(records, hand_spec, padding=config.dataset.normalizer_padding)
    train_dataset, val_dataset = split_dataset(records, dataset_config=config.dataset, normalizer=normalizer)
    train_dataset = materialize_grasp_dataset(train_dataset)
    if val_dataset is not None:
        val_dataset = materialize_grasp_dataset(val_dataset)

    model_config = config.model
    if model_config.pose_dim != hand_spec.pose_dim:
        model_config = type(config.model)(**{**config.model.__dict__, "pose_dim": hand_spec.pose_dim})
        config = type(config)(**{**config.__dict__, "model": model_config})

    params = init_model_params(rng_key, config.model)
    optimizer_state = init_adamw(params)
    schedule = make_diffusion_schedule(config.diffusion)
    params = jax.device_put(params, device)
    schedule = jax.device_put(schedule, device)
    optimizer_state = jax.device_put(optimizer_state, device)
    train_dataset = place_grasp_dataset_on_device(train_dataset, device)
    if val_dataset is not None:
        val_dataset = place_grasp_dataset_on_device(val_dataset, device)

    return TrainingBundle(
        hand_spec=hand_spec,
        normalizer=normalizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        schedule=schedule,
        params=params,
        optimizer_state=optimizer_state,
        device=device,
    )


def _mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted(items[0].keys())
    return {key: float(np.mean([entry[key] for entry in items])) for key in keys}


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


def _dataset_to_tree(dataset: GraspDatasetArrays) -> dict[str, jax.Array]:
    return {
        "pose": dataset.pose,
        "pose_raw": dataset.pose_raw,
        "object_points": dataset.object_points,
        "object_normals": dataset.object_normals,
        "object_index": dataset.object_index,
        "contact_indices": dataset.contact_indices,
        "energy": dataset.energy,
    }


def _prepare_epoch_batches(
    dataset_tree: dict[str, jax.Array],
    *,
    batch_size: int,
    shuffle: bool,
    rng_key: jax.Array,
) -> dict[str, jax.Array]:
    num_samples = int(dataset_tree["pose"].shape[0])
    num_batches = max(1, math.ceil(num_samples / batch_size))
    padded_size = num_batches * batch_size
    pad_count = padded_size - num_samples

    if shuffle:
        indices = jax.random.permutation(rng_key, num_samples).astype(jnp.int32)
        pad_indices = indices[:pad_count]
    else:
        indices = jnp.arange(num_samples, dtype=jnp.int32)
        pad_indices = jnp.repeat(indices[-1:], pad_count) if pad_count else indices[:0]

    gather_indices = jnp.concatenate([indices, pad_indices], axis=0)
    sample_weight = jnp.concatenate(
        [
            jnp.ones((num_samples,), dtype=jnp.float32),
            jnp.zeros((pad_count,), dtype=jnp.float32),
        ],
        axis=0,
    )
    batched = jax.tree_util.tree_map(
        lambda value: value[gather_indices].reshape((num_batches, batch_size) + value.shape[1:]),
        dataset_tree,
    )
    batched["sample_weight"] = sample_weight.reshape((num_batches, batch_size))
    batched["batch_weight"] = jnp.sum(batched["sample_weight"], axis=1)
    return batched


def _weighted_metrics_mean(metrics: dict[str, jax.Array], batch_weight: jax.Array) -> dict[str, jax.Array]:
    denom = jnp.maximum(jnp.sum(batch_weight), jnp.asarray(1.0, dtype=jnp.float32))
    return {
        key: jnp.sum(value * batch_weight) / denom
        for key, value in metrics.items()
    }


def _build_epoch_runners(config: TrainingConfig, bundle: TrainingBundle):
    model_config = config.model
    if model_config.pose_dim != bundle.hand_spec.pose_dim:
        model_config = type(config.model)(**{**config.model.__dict__, "pose_dim": bundle.hand_spec.pose_dim})

    def loss_fn(params, batch, rng_key):
        return diffusion_loss(
            params,
            batch,
            rng_key,
            model_config=model_config,
            diffusion_config=config.diffusion,
            loss_config=config.loss,
            normalizer=bundle.normalizer,
            hand_spec=bundle.hand_spec,
            schedule=bundle.schedule,
        )

    def train_step(params, optimizer_state, batch, rng_key):
        (loss_value, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, rng_key)
        grads = _clip_grads(grads, config.optimizer.grad_clip_norm)
        next_params, next_optimizer_state = adamw_update(
            params,
            grads,
            optimizer_state,
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )
        return next_params, next_optimizer_state, metrics

    def eval_step(params, batch, rng_key):
        _, metrics = loss_fn(params, batch, rng_key)
        return metrics

    def train_epoch(params, optimizer_state, dataset_tree, rng_key):
        order_key, step_key = jax.random.split(rng_key)
        epoch_batches = _prepare_epoch_batches(
            dataset_tree,
            batch_size=config.optimizer.batch_size,
            shuffle=True,
            rng_key=order_key,
        )
        step_keys = jax.random.split(step_key, epoch_batches["batch_weight"].shape[0])

        def body(carry, xs):
            current_params, current_optimizer_state = carry
            batch, current_step_key = xs
            next_params, next_optimizer_state, metrics = train_step(
                current_params,
                current_optimizer_state,
                batch,
                current_step_key,
            )
            return (next_params, next_optimizer_state), metrics

        (next_params, next_optimizer_state), metrics = jax.lax.scan(
            body,
            (params, optimizer_state),
            (epoch_batches, step_keys),
        )
        mean_metrics = _weighted_metrics_mean(metrics, epoch_batches["batch_weight"])
        return next_params, next_optimizer_state, mean_metrics

    def eval_epoch(params, dataset_tree, rng_key):
        epoch_batches = _prepare_epoch_batches(
            dataset_tree,
            batch_size=config.optimizer.batch_size,
            shuffle=False,
            rng_key=rng_key,
        )
        step_keys = jax.random.split(rng_key, epoch_batches["batch_weight"].shape[0])

        def body(current_params, xs):
            batch, current_step_key = xs
            metrics = eval_step(current_params, batch, current_step_key)
            return current_params, metrics

        _, metrics = jax.lax.scan(
            body,
            params,
            (epoch_batches, step_keys),
        )
        return _weighted_metrics_mean(metrics, epoch_batches["batch_weight"])

    return jax.jit(train_epoch), jax.jit(eval_epoch)


def _run_epoch(
    dataset,
    *,
    shuffle: bool,
    seed: int,
    train_epoch,
    eval_epoch,
    params,
    optimizer_state,
) -> tuple[object, object, dict[str, float]]:
    dataset_tree = _dataset_to_tree(dataset)
    rng_key = jax.random.key(np.uint32(int(seed) % (2**32)))
    if optimizer_state is not None:
        params, optimizer_state, metrics = train_epoch(params, optimizer_state, dataset_tree, rng_key)
    else:
        metrics = eval_epoch(params, dataset_tree, rng_key)
    metrics_host = jax.device_get(metrics)
    return params, optimizer_state, {key: float(np.asarray(value)) for key, value in metrics_host.items()}


def _cleanup_old_checkpoints(output_dir: Path, keep_last: int) -> None:
    checkpoints = sorted(output_dir.glob("epoch_*.pkl"))
    if keep_last <= 0 or len(checkpoints) <= keep_last:
        return
    for checkpoint in checkpoints[:-keep_last]:
        checkpoint.unlink(missing_ok=True)


def train_grasp_diffusion(config: TrainingConfig) -> TrainingResult:
    bundle = build_training_bundle(config)

    output_dir = config.checkpoint.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = output_dir / "latest.pkl"
    best_checkpoint = output_dir / "best.pkl"

    start_epoch = 0
    step = 0
    best_val_loss: float | None = None
    history: list[TrainingHistory] = []
    params = bundle.params
    optimizer_state = bundle.optimizer_state

    if config.checkpoint.resume_from is not None and config.checkpoint.resume_from.exists():
        payload = load_training_checkpoint(config.checkpoint.resume_from)
        params = jax.device_put(payload["params"], bundle.device)
        optimizer_state = jax.device_put(payload["optimizer_state"], bundle.device)
        bundle.normalizer = payload["normalizer"]
        start_epoch = int(payload.get("epoch", 0)) + 1
        step = int(payload.get("step", 0))
        best_val_loss = payload.get("best_val_loss")

    train_epoch, eval_epoch = _build_epoch_runners(config, bundle)
    final_epoch = max(start_epoch, config.optimizer.epochs - 1)

    for epoch in range(start_epoch, config.optimizer.epochs):
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        params, optimizer_state, train_metrics = _run_epoch(
            bundle.train_dataset,
            shuffle=True,
            seed=config.seed + epoch,
            train_epoch=train_epoch,
            eval_epoch=eval_epoch,
            params=params,
            optimizer_state=optimizer_state,
        )
        train_duration = time.perf_counter() - train_start
        val_metrics = None
        val_duration = 0.0
        if bundle.val_dataset is not None:
            val_start = time.perf_counter()
            _, _, val_metrics = _run_epoch(
                bundle.val_dataset,
                shuffle=False,
                seed=config.seed + epoch,
                train_epoch=train_epoch,
                eval_epoch=eval_epoch,
                params=params,
                optimizer_state=None,
            )
            val_duration = time.perf_counter() - val_start

        history_entry = TrainingHistory(
            epoch=epoch,
            train_loss=float(train_metrics.get("loss", math.nan)),
            train_noise_loss=float(train_metrics.get("noise_loss", math.nan)),
            train_joint_limit_loss=float(train_metrics.get("joint_limit_loss", 0.0)),
            train_root_distance_loss=float(train_metrics.get("root_distance_loss", 0.0)),
            val_loss=None if val_metrics is None else float(val_metrics.get("loss", math.nan)),
        )
        history.append(history_entry)
        step += math.ceil(len(bundle.train_dataset) / config.optimizer.batch_size)

        current_val = history_entry.val_loss if history_entry.val_loss is not None else history_entry.train_loss
        checkpoint_duration = 0.0
        if best_val_loss is None or current_val < best_val_loss:
            best_val_loss = current_val

        if epoch == final_epoch:
            checkpoint_start = time.perf_counter()
            save_training_checkpoint(
                latest_checkpoint,
                params=params,
                optimizer_state=optimizer_state,
                normalizer=bundle.normalizer,
                config=config,
                epoch=epoch,
                step=step,
                best_val_loss=best_val_loss,
            )
            shutil.copy2(latest_checkpoint, best_checkpoint)
            checkpoint_duration += time.perf_counter() - checkpoint_start

        epoch_duration = time.perf_counter() - epoch_start
        should_log_epoch = ((epoch + 1) % 100 == 0) or (epoch == final_epoch)
        if should_log_epoch:
            metrics_line = (
                f"[epoch {epoch:04d}] "
                f"train={train_duration:.3f}s "
                f"val={val_duration:.3f}s "
                f"ckpt={checkpoint_duration:.3f}s "
                f"total={epoch_duration:.3f}s "
                f"train_loss={history_entry.train_loss:.6f}"
            )
            if history_entry.val_loss is not None:
                metrics_line += f" val_loss={history_entry.val_loss:.6f}"
            print(metrics_line, flush=True)

    return TrainingResult(
        output_dir=output_dir,
        best_checkpoint=best_checkpoint if best_checkpoint.exists() else None,
        latest_checkpoint=latest_checkpoint if latest_checkpoint.exists() else None,
        history=tuple(history),
    )
