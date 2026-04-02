from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import (
    CheckpointConfig,
    DatasetConfig,
    DiffusionConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)


def default_training_config() -> TrainingConfig:
    return TrainingConfig(
        dataset=DatasetConfig(),
        model=ModelConfig(),
        diffusion=DiffusionConfig(),
        loss=LossConfig(),
        optimizer=OptimizerConfig(),
        checkpoint=CheckpointConfig(),
    )


def training_config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    return {
        "dataset": {
            "artifact_paths": [str(path) for path in config.dataset.artifact_paths],
            "artifact_glob": config.dataset.artifact_glob,
            "state_name": config.dataset.state_name,
            "train_fraction": config.dataset.train_fraction,
            "seed": config.dataset.seed,
            "object_num_points": config.dataset.object_num_points,
            "object_point_seed": config.dataset.object_point_seed,
            "normalizer_padding": config.dataset.normalizer_padding,
            "drop_invalid_samples": config.dataset.drop_invalid_samples,
            "num_workers": config.dataset.num_workers,
            "pin_memory": config.dataset.pin_memory,
        },
        "model": {
            "architecture": config.model.architecture,
            "pose_dim": config.model.pose_dim,
            "point_feature_dim": config.model.point_feature_dim,
            "hidden_dim": config.model.hidden_dim,
            "context_dim": config.model.context_dim,
            "context_tokens": config.model.context_tokens,
            "scene_encoder_layers": config.model.scene_encoder_layers,
            "denoiser_blocks": config.model.denoiser_blocks,
            "num_heads": config.model.num_heads,
            "dropout": config.model.dropout,
            "time_embed_dim": config.model.time_embed_dim,
        },
        "diffusion": {
            "steps": config.diffusion.steps,
            "beta_start": config.diffusion.beta_start,
            "beta_end": config.diffusion.beta_end,
            "rand_t_type": config.diffusion.rand_t_type,
            "loss_type": config.diffusion.loss_type,
        },
        "loss": {
            "noise_weight": config.loss.noise_weight,
            "joint_limit_weight": config.loss.joint_limit_weight,
            "root_distance_weight": config.loss.root_distance_weight,
            "root_distance_threshold": config.loss.root_distance_threshold,
        },
        "optimizer": {
            "lr": config.optimizer.lr,
            "weight_decay": config.optimizer.weight_decay,
            "batch_size": config.optimizer.batch_size,
            "epochs": config.optimizer.epochs,
            "grad_clip_norm": config.optimizer.grad_clip_norm,
        },
        "checkpoint": {
            "output_dir": str(config.checkpoint.output_dir),
            "save_every": config.checkpoint.save_every,
            "keep_last": config.checkpoint.keep_last,
            "resume_from": None if config.checkpoint.resume_from is None else str(config.checkpoint.resume_from),
        },
        "hand_side": config.hand_side,
        "device": config.device,
        "seed": config.seed,
    }


def training_config_from_dict(data: dict[str, Any]) -> TrainingConfig:
    defaults = default_training_config()

    dataset_data = dict(data.get("dataset", {}))
    model_data = dict(data.get("model", {}))
    diffusion_data = dict(data.get("diffusion", {}))
    loss_data = dict(data.get("loss", {}))
    optimizer_data = dict(data.get("optimizer", {}))
    checkpoint_data = dict(data.get("checkpoint", {}))

    dataset = DatasetConfig(
        artifact_paths=tuple(Path(path).expanduser() for path in dataset_data.get("artifact_paths", defaults.dataset.artifact_paths)),
        artifact_glob=dataset_data.get("artifact_glob", defaults.dataset.artifact_glob),
        state_name=dataset_data.get("state_name", defaults.dataset.state_name),
        train_fraction=float(dataset_data.get("train_fraction", defaults.dataset.train_fraction)),
        seed=int(dataset_data.get("seed", defaults.dataset.seed)),
        object_num_points=int(dataset_data.get("object_num_points", defaults.dataset.object_num_points)),
        object_point_seed=int(dataset_data.get("object_point_seed", defaults.dataset.object_point_seed)),
        normalizer_padding=float(dataset_data.get("normalizer_padding", defaults.dataset.normalizer_padding)),
        drop_invalid_samples=bool(dataset_data.get("drop_invalid_samples", defaults.dataset.drop_invalid_samples)),
        num_workers=int(dataset_data.get("num_workers", defaults.dataset.num_workers)),
        pin_memory=bool(dataset_data.get("pin_memory", defaults.dataset.pin_memory)),
    )
    model = ModelConfig(
        architecture=str(model_data.get("architecture", defaults.model.architecture)),
        pose_dim=int(model_data.get("pose_dim", defaults.model.pose_dim)),
        point_feature_dim=int(model_data.get("point_feature_dim", defaults.model.point_feature_dim)),
        hidden_dim=int(model_data.get("hidden_dim", defaults.model.hidden_dim)),
        context_dim=int(model_data.get("context_dim", defaults.model.context_dim)),
        context_tokens=int(model_data.get("context_tokens", defaults.model.context_tokens)),
        scene_encoder_layers=int(model_data.get("scene_encoder_layers", defaults.model.scene_encoder_layers)),
        denoiser_blocks=int(model_data.get("denoiser_blocks", defaults.model.denoiser_blocks)),
        num_heads=int(model_data.get("num_heads", defaults.model.num_heads)),
        dropout=float(model_data.get("dropout", defaults.model.dropout)),
        time_embed_dim=int(model_data.get("time_embed_dim", defaults.model.time_embed_dim)),
    )
    diffusion = DiffusionConfig(
        steps=int(diffusion_data.get("steps", defaults.diffusion.steps)),
        beta_start=float(diffusion_data.get("beta_start", defaults.diffusion.beta_start)),
        beta_end=float(diffusion_data.get("beta_end", defaults.diffusion.beta_end)),
        rand_t_type=diffusion_data.get("rand_t_type", defaults.diffusion.rand_t_type),
        loss_type=diffusion_data.get("loss_type", defaults.diffusion.loss_type),
    )
    loss = LossConfig(
        noise_weight=float(loss_data.get("noise_weight", defaults.loss.noise_weight)),
        joint_limit_weight=float(loss_data.get("joint_limit_weight", defaults.loss.joint_limit_weight)),
        root_distance_weight=float(loss_data.get("root_distance_weight", defaults.loss.root_distance_weight)),
        root_distance_threshold=float(loss_data.get("root_distance_threshold", defaults.loss.root_distance_threshold)),
    )
    optimizer = OptimizerConfig(
        lr=float(optimizer_data.get("lr", defaults.optimizer.lr)),
        weight_decay=float(optimizer_data.get("weight_decay", defaults.optimizer.weight_decay)),
        batch_size=int(optimizer_data.get("batch_size", defaults.optimizer.batch_size)),
        epochs=int(optimizer_data.get("epochs", defaults.optimizer.epochs)),
        grad_clip_norm=float(optimizer_data.get("grad_clip_norm", defaults.optimizer.grad_clip_norm)),
    )
    resume_from = checkpoint_data.get("resume_from", defaults.checkpoint.resume_from)
    checkpoint = CheckpointConfig(
        output_dir=Path(checkpoint_data.get("output_dir", defaults.checkpoint.output_dir)).expanduser(),
        save_every=int(checkpoint_data.get("save_every", defaults.checkpoint.save_every)),
        keep_last=int(checkpoint_data.get("keep_last", defaults.checkpoint.keep_last)),
        resume_from=None if resume_from is None else Path(resume_from).expanduser(),
    )

    return TrainingConfig(
        dataset=dataset,
        model=model,
        diffusion=diffusion,
        loss=loss,
        optimizer=optimizer,
        checkpoint=checkpoint,
        hand_side=str(data.get("hand_side", defaults.hand_side)),
        device=str(data.get("device", defaults.device)),
        seed=int(data.get("seed", defaults.seed)),
    )


def load_training_config_json(path: str | Path) -> TrainingConfig:
    config_path = Path(path).expanduser().resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Training config at {config_path} must be a JSON object.")
    return training_config_from_dict(payload)


def save_training_config_json(path: str | Path, config: TrainingConfig) -> Path:
    config_path = Path(path).expanduser().resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(training_config_to_dict(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config_path
