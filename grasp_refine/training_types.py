from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class PreparedDatasetConfig:
    path: Path
    train_fraction: float = 0.9
    split_mode: Literal["sample", "object", "object_random", "object_fixed"] = "object"
    train_object_keys: tuple[str, ...] = ()
    val_object_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelConfig:
    architecture: Literal["mlp", "dga_unet", "dga_transformer"] = "dga_unet"
    pose_dim: int = 0
    point_feature_dim: int = 6
    hidden_dim: int = 512
    context_dim: int = 512
    context_tokens: int = 8
    scene_encoder_layers: int = 1
    denoiser_blocks: int = 4
    transformer_depth: int = 1
    num_heads: int = 8
    time_embed_dim: int = 1024
    resblock_dropout: float = 0.0
    transformer_dropout: float = 0.1
    transformer_dim_head: int = 64
    transformer_mult_ff: int = 2
    use_position_embedding: bool = False
    scene_encoder_pretrained: Path | None = None
    freeze_scene_encoder: bool = False


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
    erf_weight: float = 1.0
    spf_weight: float = 1.0
    srf_weight: float = 1.0
    spf_threshold: float = 0.02
    srf_threshold: float = 0.02


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 1.0e-4
    weight_decay: float = 0.0
    batch_size: int = 32
    epochs: int = 1
    grad_clip_norm: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    dataset: PreparedDatasetConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    log_step: int = 100
    run_validation: bool = False
    checkpoint_dir: Path | None = None
    save_model_interval: int = 1
    save_model_separately: bool = True
    save_scene_model: bool = True
    metrics_path: Path | None = None
    device: str = "cpu"
    seed: int = 2022


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    step: int
    train_loss: float
    train_noise_loss: float
    train_erf_loss: float
    train_spf_loss: float
    train_srf_loss: float
    val_loss: float | None = None


@dataclass(frozen=True)
class TrainingResult:
    resolved_model_config: ModelConfig
    history: tuple[EpochMetrics, ...]
    final_step: int
