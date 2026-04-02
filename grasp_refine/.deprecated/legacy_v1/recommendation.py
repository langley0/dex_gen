from __future__ import annotations

from dataclasses import dataclass, replace

from .io import load_grasp_artifact, resolve_artifact_paths
from .types import DatasetConfig, TrainingConfig


@dataclass(frozen=True)
class DatasetSummary:
    artifact_count: int
    sample_count: int
    object_kinds: tuple[str, ...]


@dataclass(frozen=True)
class TrainingRecommendation:
    batch_size: int
    epochs: int
    rationale: str


def inspect_dataset_summary(dataset_config: DatasetConfig) -> DatasetSummary:
    paths = resolve_artifact_paths(dataset_config)
    sample_count = 0
    object_kinds: set[str] = set()
    for path in paths:
        payload = load_grasp_artifact(path, state_name=dataset_config.state_name)
        sample_count += len(payload.records)
        object_kinds.update(record.object_kind for record in payload.records)
    return DatasetSummary(
        artifact_count=len(paths),
        sample_count=sample_count,
        object_kinds=tuple(sorted(object_kinds)),
    )


def recommend_training_setup(summary: DatasetSummary, *, device: str) -> TrainingRecommendation:
    device_name = device.lower()
    uses_gpu = "gpu" in device_name or "cuda" in device_name

    if summary.sample_count <= 1:
        batch_size = 1
        epochs = 300
        rationale = "single-sample dataset, so full-batch fitting with a high epoch count is only useful as a debug baseline"
    elif summary.sample_count <= 16:
        batch_size = 4 if uses_gpu else 2
        epochs = 250
        rationale = "very small dataset, so smaller batches and longer training are the safest default"
    elif summary.sample_count <= 128:
        batch_size = 16 if uses_gpu else 8
        epochs = 200
        rationale = "small dataset, so moderate batches with more repeated passes usually work well"
    elif summary.sample_count <= 1_024:
        batch_size = 32 if uses_gpu else 16
        epochs = 120
        rationale = "mid-sized dataset, so batch size can grow while total epochs come down"
    elif summary.sample_count <= 10_000:
        batch_size = 64 if uses_gpu else 32
        epochs = 80
        rationale = "larger dataset, so throughput matters more than repeated exposure"
    else:
        batch_size = 128 if uses_gpu else 64
        epochs = 50
        rationale = "large dataset, so favor throughput and a lower epoch count"

    batch_size = max(1, min(batch_size, summary.sample_count))
    return TrainingRecommendation(batch_size=batch_size, epochs=epochs, rationale=rationale)


def apply_recommendation(config: TrainingConfig, recommendation: TrainingRecommendation) -> TrainingConfig:
    optimizer = replace(
        config.optimizer,
        batch_size=int(recommendation.batch_size),
        epochs=int(recommendation.epochs),
    )
    save_every = max(1, min(config.checkpoint.save_every, max(1, recommendation.epochs // 10)))
    checkpoint = replace(config.checkpoint, save_every=save_every)
    return replace(config, optimizer=optimizer, checkpoint=checkpoint)
