from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .checkpoint import load_checkpoint, save_checkpoint
from .diffusion import diffusion_loss, make_diffusion_schedule
from .hand_points import load_dga_hand_point_spec
from .loader import DgaBatch, iterate_dga_batches, load_saved_dga_dataset, split_dga_dataset
from .model_factory import init_model_params
from .optim import AdamState, adam_update, init_adam
from .scene_encoder_pretrained import merge_param_tree
from .training_types import EpochMetrics, ModelConfig, TrainingConfig, TrainingResult


JaxDevice = Any


def _select_device(device_name: str) -> JaxDevice:
    try:
        return jax.devices(device_name)[0]
    except Exception:
        return jax.devices()[0]


def _resolve_model_config(config: TrainingConfig, pose_dim: int) -> ModelConfig:
    if config.model.pose_dim in (0, pose_dim):
        return replace(config.model, pose_dim=int(pose_dim))
    raise ValueError(f"Configured pose_dim {config.model.pose_dim} does not match dataset pose_dim {pose_dim}.")


def _batch_to_jax(batch: DgaBatch, device: JaxDevice) -> dict[str, jax.Array]:
    return {
        "pose": jax.device_put(jnp.asarray(batch.pose, dtype=jnp.float32), device),
        "pose_raw": jax.device_put(jnp.asarray(batch.pose_raw, dtype=jnp.float32), device),
        "object_points": jax.device_put(jnp.asarray(batch.object_points, dtype=jnp.float32), device),
        "object_normals": jax.device_put(jnp.asarray(batch.object_normals, dtype=jnp.float32), device),
    }


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


def _freeze_scene_encoder_grads(grads, architecture: str):
    if not isinstance(grads, dict):
        return grads
    frozen = dict(grads)
    if architecture in ("dga_unet", "dga_transformer") and "scene_model" in frozen:
        frozen["scene_model"] = jax.tree_util.tree_map(jnp.zeros_like, frozen["scene_model"])
    if architecture == "mlp" and "scene_encoder" in frozen:
        frozen["scene_encoder"] = jax.tree_util.tree_map(jnp.zeros_like, frozen["scene_encoder"])
    return frozen


def _mean_metric_rows(rows: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    if not rows:
        return {}
    totals: dict[str, float] = {}
    total_weight = 0
    for weight, metrics in rows:
        total_weight += int(weight)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value) * float(weight)
    denom = max(total_weight, 1)
    return {key: value / float(denom) for key, value in totals.items()}


def _build_step_functions(config: TrainingConfig, model_config: ModelConfig, normalizer, hand_point_spec, schedule):
    def loss_fn(params, batch, rng_key):
        return diffusion_loss(
            params,
            batch,
            rng_key,
            model_config=model_config,
            diffusion_config=config.diffusion,
            loss_config=config.loss,
            normalizer=normalizer,
            hand_point_spec=hand_point_spec,
            schedule=schedule,
            training=True,
        )

    def eval_loss_fn(params, batch, rng_key):
        return diffusion_loss(
            params,
            batch,
            rng_key,
            model_config=model_config,
            diffusion_config=config.diffusion,
            loss_config=config.loss,
            normalizer=normalizer,
            hand_point_spec=hand_point_spec,
            schedule=schedule,
            training=False,
        )

    def train_step(params, optimizer_state: AdamState, batch, rng_key):
        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, rng_key)
        if model_config.freeze_scene_encoder:
            grads = _freeze_scene_encoder_grads(grads, model_config.architecture)
        grads = _clip_grads(grads, config.optimizer.grad_clip_norm)
        next_params, next_optimizer_state = adam_update(params, grads, optimizer_state, lr=config.optimizer.lr)
        return next_params, next_optimizer_state, metrics

    def eval_step(params, batch, rng_key):
        _, metrics = eval_loss_fn(params, batch, rng_key)
        return metrics

    return jax.jit(train_step), jax.jit(eval_step)


def train_grasp_diffusion(config: TrainingConfig) -> TrainingResult:
    dataset = load_saved_dga_dataset(config.dataset.path)
    train_subset, val_subset = split_dga_dataset(
        dataset,
        train_fraction=float(config.dataset.train_fraction),
        seed=int(config.seed),
        split_mode=config.dataset.split_mode,
        train_object_keys=config.dataset.train_object_keys,
        val_object_keys=config.dataset.val_object_keys,
    )

    hand_side = str(dataset.arrays.hand_side[0])
    hand_point_spec = load_dga_hand_point_spec(hand_side)
    model_config = _resolve_model_config(config, dataset.normalizer.pose_dim)
    device = _select_device(config.device)

    rng_key = jax.random.key(np.uint32(int(config.seed) % (2**32)))
    params = jax.device_put(init_model_params(rng_key, model_config), device)
    optimizer_state = jax.device_put(init_adam(params), device)
    schedule = jax.device_put(make_diffusion_schedule(config.diffusion), device)
    train_step, eval_step = _build_step_functions(config, model_config, dataset.normalizer, hand_point_spec, schedule)

    checkpoint_state = None
    metrics_stream = None
    if config.metrics_path is not None:
        config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_stream = config.metrics_path.open("a", encoding="utf-8")
    if config.checkpoint_dir is not None:
        checkpoint_state = load_checkpoint(
            config.checkpoint_dir,
            save_separately=config.save_model_separately,
        )
        if checkpoint_state is not None:
            merged_params = merge_param_tree(jax.device_get(params), checkpoint_state.params)
            params = jax.device_put(merged_params, device)
            optimizer_state = jax.device_put(checkpoint_state.optimizer_state, device)

    start_epoch = 0 if checkpoint_state is None else int(checkpoint_state.epoch) + 1
    step = 0 if checkpoint_state is None else int(checkpoint_state.step)
    history: list[EpochMetrics] = []
    for epoch in range(start_epoch, config.optimizer.epochs):
        train_rows: list[tuple[int, dict[str, float]]] = []
        for batch_index, batch in enumerate(
            iterate_dga_batches(
                train_subset,
                batch_size=config.optimizer.batch_size,
                shuffle=True,
                seed=config.seed + epoch,
            )
        ):
            batch_jax = _batch_to_jax(batch, device)
            step_key = jax.random.key(np.uint32((config.seed + epoch * 10_003 + batch_index) % (2**32)))
            params, optimizer_state, metrics = train_step(params, optimizer_state, batch_jax, step_key)
            metrics_host = jax.device_get(metrics)
            train_rows.append((int(batch.pose.shape[0]), {key: float(np.asarray(value)) for key, value in metrics_host.items()}))
            step += 1
            if config.log_step > 0 and step % config.log_step == 0:
                loss_value = float(np.asarray(metrics_host["loss"]))
                print(f"[TRAIN] epoch={epoch + 1} step={step} loss={loss_value:.6f}")
                if metrics_stream is not None:
                    metrics_stream.write(
                        json.dumps(
                            {
                                "phase": "train",
                                "epoch": int(epoch + 1),
                                "step": int(step),
                                **{key: float(np.asarray(value)) for key, value in metrics_host.items()},
                            }
                        )
                        + "\n"
                    )
                    metrics_stream.flush()

        val_rows: list[tuple[int, dict[str, float]]] = []
        if config.run_validation and val_subset is not None and len(val_subset) > 0:
            for batch_index, batch in enumerate(
                iterate_dga_batches(
                    val_subset,
                    batch_size=config.optimizer.batch_size,
                    shuffle=False,
                    seed=config.seed + epoch,
                )
            ):
                batch_jax = _batch_to_jax(batch, device)
                step_key = jax.random.key(np.uint32((config.seed + 1_000_003 + epoch * 10_003 + batch_index) % (2**32)))
                metrics = eval_step(params, batch_jax, step_key)
                metrics_host = jax.device_get(metrics)
                val_rows.append((int(batch.pose.shape[0]), {key: float(np.asarray(value)) for key, value in metrics_host.items()}))

        train_metrics = _mean_metric_rows(train_rows)
        val_metrics = _mean_metric_rows(val_rows) if val_rows else None
        history.append(
            EpochMetrics(
                epoch=epoch,
                step=step,
                train_loss=float(train_metrics.get("loss", float("nan"))),
                train_noise_loss=float(train_metrics.get("noise_loss", float("nan"))),
                train_erf_loss=float(train_metrics.get("erf_loss", float("nan"))),
                train_spf_loss=float(train_metrics.get("spf_loss", float("nan"))),
                train_srf_loss=float(train_metrics.get("srf_loss", float("nan"))),
                val_loss=None if val_metrics is None else float(val_metrics.get("loss", float("nan"))),
            )
        )
        if config.checkpoint_dir is not None and config.save_model_interval > 0 and (epoch + 1) % config.save_model_interval == 0:
            save_checkpoint(
                config.checkpoint_dir,
                epoch=epoch,
                step=step,
                params=params,
                optimizer_state=optimizer_state,
                save_separately=config.save_model_separately,
                save_scene_model=config.save_scene_model,
            )
        if metrics_stream is not None:
            metrics_stream.write(
                json.dumps(
                    {
                        "phase": "epoch",
                        "epoch": int(epoch + 1),
                        "step": int(step),
                        "train_loss": float(train_metrics.get("loss", float("nan"))),
                        "train_noise_loss": float(train_metrics.get("noise_loss", float("nan"))),
                        "train_erf_loss": float(train_metrics.get("erf_loss", float("nan"))),
                        "train_spf_loss": float(train_metrics.get("spf_loss", float("nan"))),
                        "train_srf_loss": float(train_metrics.get("srf_loss", float("nan"))),
                        "val_loss": None if val_metrics is None else float(val_metrics.get("loss", float("nan"))),
                    }
                )
                + "\n"
            )
            metrics_stream.flush()

    if metrics_stream is not None:
        metrics_stream.close()

    return TrainingResult(
        resolved_model_config=model_config,
        history=tuple(history),
        final_step=step,
    )
