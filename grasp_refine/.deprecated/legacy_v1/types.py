from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


ArtifactStateName = Literal["best", "current"]


@dataclass(frozen=True)
class DatasetConfig:
    artifact_paths: tuple[Path, ...] = ()
    artifact_glob: str | None = None
    state_name: ArtifactStateName = "best"
    train_fraction: float = 1.0
    seed: int = 0
    object_num_points: int = 2048
    object_point_seed: int = 13
    normalizer_padding: float = 0.02
    drop_invalid_samples: bool = True
    num_workers: int = 0
    pin_memory: bool = True


@dataclass(frozen=True)
class ModelConfig:
    architecture: Literal["mlp", "dga_transformer"] = "mlp"
    pose_dim: int = 21
    point_feature_dim: int = 6
    hidden_dim: int = 256
    context_dim: int = 256
    context_tokens: int = 8
    scene_encoder_layers: int = 2
    denoiser_blocks: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    time_embed_dim: int = 256


@dataclass(frozen=True)
class DiffusionConfig:
    steps: int = 100
    beta_start: float = 1.0e-4
    beta_end: float = 1.0e-2
    rand_t_type: Literal["all", "half"] = "half"
    loss_type: Literal["l1", "l2"] = "l1"


@dataclass(frozen=True)
class LossConfig:
    noise_weight: float = 1.0
    joint_limit_weight: float = 0.1
    root_distance_weight: float = 0.05
    root_distance_threshold: float = 0.02


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-4
    batch_size: int = 64
    epochs: int = 3000
    grad_clip_norm: float = 1.0


@dataclass(frozen=True)
class CheckpointConfig:
    output_dir: Path = Path("outputs/grasp_refine_train")
    save_every: int = 100
    keep_last: int = 5
    resume_from: Path | None = None


@dataclass(frozen=True)
class TrainingConfig:
    dataset: DatasetConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    hand_side: Literal["right", "left"] = "right"
    device: str = "cuda"
    seed: int = 0


@dataclass(frozen=True)
class TrainingHistory:
    epoch: int
    train_loss: float
    train_noise_loss: float
    train_joint_limit_loss: float
    train_root_distance_loss: float
    val_loss: float | None = None


@dataclass(frozen=True)
class TrainingResult:
    output_dir: Path
    best_checkpoint: Path | None
    latest_checkpoint: Path | None
    history: tuple[TrainingHistory, ...]
