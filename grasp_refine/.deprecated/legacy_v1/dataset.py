from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .inspire_hand import InspireHandSpec
from .io import GraspRecord, load_grasp_records
from .normalization import PoseNormalizer
from .object_mesh import load_object_mesh, sample_mesh_points
from .types import DatasetConfig


@dataclass(frozen=True)
class _ObjectCloud:
    points: np.ndarray
    normals: np.ndarray


class GraspArtifactDataset:
    def __init__(
        self,
        records: Sequence[GraspRecord],
        *,
        dataset_config: DatasetConfig,
        normalizer: PoseNormalizer,
    ) -> None:
        self.records = list(records)
        self.dataset_config = dataset_config
        self.normalizer = normalizer
        self._object_cloud_cache: dict[str, _ObjectCloud] = {}
        object_kinds = sorted({record.object_kind for record in self.records})
        self._object_index = {kind: index for index, kind in enumerate(object_kinds)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        record = self.records[int(index)]
        cloud = self._object_cloud(record)
        return {
            "pose": self.normalizer.normalize_numpy(record.pose),
            "pose_raw": record.pose.astype(np.float32),
            "object_points": cloud.points.astype(np.float32),
            "object_normals": cloud.normals.astype(np.float32),
            "object_index": np.asarray(self._object_index[record.object_kind], dtype=np.int32),
            "contact_indices": record.contact_indices.astype(np.int32),
            "energy": np.asarray(record.total_energy, dtype=np.float32),
        }

    def _object_cloud(self, record: GraspRecord) -> _ObjectCloud:
        cache_key = record.object_kind
        cached = self._object_cloud_cache.get(cache_key)
        if cached is not None:
            return cached
        mesh = load_object_mesh(record.object_metadata)
        stable_hash = sum((index + 1) * ord(char) for index, char in enumerate(record.object_kind))
        seed = int(self.dataset_config.object_point_seed) + stable_hash % 100_000
        points, normals = sample_mesh_points(mesh, self.dataset_config.object_num_points, seed=seed)
        cached = _ObjectCloud(points=points.astype(np.float32), normals=normals.astype(np.float32))
        self._object_cloud_cache[cache_key] = cached
        return cached


def load_dataset_records(config: DatasetConfig) -> list[GraspRecord]:
    return load_grasp_records(config)


def build_pose_normalizer(records: Sequence[GraspRecord], hand_spec: InspireHandSpec, *, padding: float) -> PoseNormalizer:
    poses = np.stack([record.pose for record in records], axis=0).astype(np.float32)
    translation_lower = poses[:, :3].min(axis=0) - float(padding)
    translation_upper = poses[:, :3].max(axis=0) + float(padding)
    return PoseNormalizer(
        translation_lower=translation_lower.astype(np.float32),
        translation_upper=translation_upper.astype(np.float32),
        joint_lower=hand_spec.joint_lower.astype(np.float32),
        joint_upper=hand_spec.joint_upper.astype(np.float32),
    )


def split_dataset(
    records: Sequence[GraspRecord],
    *,
    dataset_config: DatasetConfig,
    normalizer: PoseNormalizer,
) -> tuple[GraspArtifactDataset, GraspArtifactDataset | None]:
    total_size = len(records)
    if total_size <= 0:
        raise ValueError("records must contain at least one sample.")
    if float(dataset_config.train_fraction) >= 1.0:
        train_dataset = GraspArtifactDataset(records, dataset_config=dataset_config, normalizer=normalizer)
        return train_dataset, None
    train_size = max(1, int(round(total_size * dataset_config.train_fraction)))
    train_size = min(train_size, total_size)
    indices = np.arange(total_size, dtype=np.int32)
    rng = np.random.default_rng(int(dataset_config.seed))
    rng.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_records = [records[int(index)] for index in train_indices]
    val_records = [records[int(index)] for index in val_indices]
    train_dataset = GraspArtifactDataset(train_records, dataset_config=dataset_config, normalizer=normalizer)
    val_dataset = None
    if val_records:
        val_dataset = GraspArtifactDataset(val_records, dataset_config=dataset_config, normalizer=normalizer)
    return train_dataset, val_dataset
