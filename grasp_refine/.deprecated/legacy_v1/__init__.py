from __future__ import annotations

import os


def _configure_xla_runtime() -> None:
    # JAX/XLA's Triton GEMM fusion path can emit noisy compiler errors for the
    # scene encoder matmuls used in grasp_refine. Disable that path by default
    # unless the caller explicitly opts back in.
    raw_value = os.environ.get("GRASP_REFINE_DISABLE_TRITON_GEMM", "1").strip().lower()
    if raw_value in {"0", "false", "no", "off"}:
        return
    flag = "--xla_gpu_enable_triton_gemm=false"
    xla_flags = os.environ.get("XLA_FLAGS", "").strip()
    tokens = xla_flags.split() if xla_flags else []
    if flag not in tokens:
        os.environ["XLA_FLAGS"] = " ".join([*tokens, flag]).strip()


_configure_xla_runtime()


from .batch import GraspBatch, collate_grasp_batch, iterate_grasp_batches
from .artifact_subset import ArtifactSubsetResult, subset_grasp_artifact_topk
from .checkpoint import load_training_checkpoint, save_training_checkpoint
from .config_io import (
    default_training_config,
    load_training_config_json,
    save_training_config_json,
    training_config_from_dict,
    training_config_to_dict,
)
from .dataset import GraspArtifactDataset, build_pose_normalizer, load_dataset_records, split_dataset
from .diffusion import DiffusionSchedule, make_diffusion_schedule
from .inspire_hand import InspireHandSpec, load_inspire_hand_spec
from .io import GraspArtifactPayload, GraspRecord, load_grasp_artifact, load_grasp_records, resolve_artifact_paths
from .inference import ObjectConditionedSample, sample_checkpoint_artifacts, save_dga_style_samples
from .losses import joint_limit_loss, root_distance_loss
from .model_factory import apply_model, init_model_params
from .normalization import PoseNormalizer
from .optim import AdamWState, adamw_update, init_adamw
from .pipeline import (
    PipelinePaths,
    PipelineResult,
    build_optimizer_command,
    build_refine_train_command,
    build_view_commands,
    default_pipeline_paths,
    format_command,
    run_grasp_gen_to_refine_pipeline,
)
from .recommendation import (
    DatasetSummary,
    TrainingRecommendation,
    apply_recommendation,
    inspect_dataset_summary,
    recommend_training_setup,
)
from .sampling import SampleBatch, sample_grasp_poses
from .trainer import TrainingBundle, build_training_bundle, train_grasp_diffusion
from .types import (
    ArtifactStateName,
    CheckpointConfig,
    DatasetConfig,
    DiffusionConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    TrainingHistory,
    TrainingResult,
)

__all__ = [
    "ArtifactStateName",
    "ArtifactSubsetResult",
    "AdamWState",
    "CheckpointConfig",
    "DatasetConfig",
    "DatasetSummary",
    "DiffusionConfig",
    "DiffusionSchedule",
    "GraspArtifactDataset",
    "GraspArtifactPayload",
    "GraspBatch",
    "GraspRecord",
    "InspireHandSpec",
    "LossConfig",
    "ModelConfig",
    "ObjectConditionedSample",
    "OptimizerConfig",
    "PipelinePaths",
    "PipelineResult",
    "PoseNormalizer",
    "SampleBatch",
    "TrainingBundle",
    "TrainingConfig",
    "TrainingHistory",
    "TrainingRecommendation",
    "TrainingResult",
    "adamw_update",
    "apply_recommendation",
    "apply_model",
    "build_optimizer_command",
    "build_pose_normalizer",
    "build_training_bundle",
    "build_refine_train_command",
    "build_view_commands",
    "collate_grasp_batch",
    "default_training_config",
    "default_pipeline_paths",
    "format_command",
    "init_adamw",
    "init_model_params",
    "inspect_dataset_summary",
    "iterate_grasp_batches",
    "joint_limit_loss",
    "load_dataset_records",
    "load_grasp_artifact",
    "load_grasp_records",
    "load_inspire_hand_spec",
    "load_training_config_json",
    "load_training_checkpoint",
    "make_diffusion_schedule",
    "recommend_training_setup",
    "resolve_artifact_paths",
    "root_distance_loss",
    "sample_checkpoint_artifacts",
    "save_training_config_json",
    "save_training_checkpoint",
    "save_dga_style_samples",
    "sample_grasp_poses",
    "split_dataset",
    "subset_grasp_artifact_topk",
    "train_grasp_diffusion",
    "training_config_from_dict",
    "training_config_to_dict",
    "run_grasp_gen_to_refine_pipeline",
]
