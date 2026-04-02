from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .dataset import DgaDataRecord, build_dga_data_records
from .hand_spec import DgaHandSpec, load_dga_hand_spec
from .normalization import DgaPoseNormalizer, build_pose_normalizer, normalize_records
from .types import DatasetConfig


@dataclass(frozen=True)
class DgaDatasetArrays:
    pose: np.ndarray
    pose_raw: np.ndarray
    pose_full: np.ndarray
    object_points: np.ndarray
    object_normals: np.ndarray
    contact_indices: np.ndarray
    total_energy: np.ndarray
    sample_index: np.ndarray
    source_path: np.ndarray
    hand_side: np.ndarray
    object_kind: np.ndarray
    object_name: np.ndarray
    object_key: np.ndarray


@dataclass(frozen=True)
class MaterializedDgaDataset:
    config: DatasetConfig
    hand_spec: DgaHandSpec
    normalizer: DgaPoseNormalizer
    raw_records: tuple[DgaDataRecord, ...]
    normalized_records: tuple[DgaDataRecord, ...]
    arrays: DgaDatasetArrays


def infer_hand_side(records: Sequence[DgaDataRecord]) -> str:
    sides = sorted({str(record.hand_side) for record in records})
    if not sides:
        raise ValueError("records must contain at least one sample.")
    if len(sides) != 1:
        raise ValueError(f"Expected a single hand side, got {sides}.")
    return sides[0]


def stack_records(raw_records: Sequence[DgaDataRecord], normalized_records: Sequence[DgaDataRecord]) -> DgaDatasetArrays:
    if len(raw_records) != len(normalized_records):
        raise ValueError(f"raw/normalized record count mismatch: {len(raw_records)} vs {len(normalized_records)}")
    if not raw_records:
        raise ValueError("records must contain at least one sample.")

    return DgaDatasetArrays(
        pose=np.stack([np.asarray(record.pose, dtype=np.float32) for record in normalized_records], axis=0).astype(np.float32),
        pose_raw=np.stack([np.asarray(record.pose, dtype=np.float32) for record in raw_records], axis=0).astype(np.float32),
        pose_full=np.stack([np.asarray(record.pose_full, dtype=np.float32) for record in raw_records], axis=0).astype(np.float32),
        object_points=np.stack([np.asarray(record.object_points, dtype=np.float32) for record in raw_records], axis=0).astype(np.float32),
        object_normals=np.stack([np.asarray(record.object_normals, dtype=np.float32) for record in raw_records], axis=0).astype(np.float32),
        contact_indices=np.stack([np.asarray(record.contact_indices, dtype=np.int32) for record in raw_records], axis=0).astype(np.int32),
        total_energy=np.asarray([float(record.total_energy) for record in raw_records], dtype=np.float32),
        sample_index=np.asarray([int(record.sample_index) for record in raw_records], dtype=np.int32),
        source_path=np.asarray([str(record.source_path) for record in raw_records], dtype=np.str_),
        hand_side=np.asarray([str(record.hand_side) for record in raw_records], dtype=np.str_),
        object_kind=np.asarray([str(record.object_kind) for record in raw_records], dtype=np.str_),
        object_name=np.asarray([str(record.object_name) for record in raw_records], dtype=np.str_),
        object_key=np.asarray([str(record.object_key) for record in raw_records], dtype=np.str_),
    )


def build_materialized_dga_dataset(
    config: DatasetConfig,
    *,
    normalizer_padding: float = 0.02,
) -> MaterializedDgaDataset:
    raw_records = tuple(build_dga_data_records(config))
    hand_side = infer_hand_side(raw_records)
    hand_spec = load_dga_hand_spec(hand_side)
    normalizer = build_pose_normalizer(raw_records, hand_spec, padding=normalizer_padding)
    normalized_records = tuple(normalize_records(raw_records, normalizer))
    arrays = stack_records(raw_records, normalized_records)
    return MaterializedDgaDataset(
        config=config,
        hand_spec=hand_spec,
        normalizer=normalizer,
        raw_records=raw_records,
        normalized_records=normalized_records,
        arrays=arrays,
    )


def dataset_metadata(bundle: MaterializedDgaDataset) -> dict[str, Any]:
    return {
        "dataset_config": {
            "artifact_paths": [str(path) for path in bundle.config.artifact_paths],
            "artifact_glob": bundle.config.artifact_glob,
            "state_name": bundle.config.state_name,
            "object_num_points": int(bundle.config.object_num_points),
            "object_point_seed": int(bundle.config.object_point_seed),
            "coordinate_mode": bundle.config.coordinate_mode,
            "drop_invalid_samples": bool(bundle.config.drop_invalid_samples),
        },
        "hand_spec": {
            "side": bundle.hand_spec.side,
            "joint_dim": int(bundle.hand_spec.joint_dim),
        },
        "normalizer": {
            key: value.astype(np.float32).tolist()
            for key, value in bundle.normalizer.state_dict().items()
        },
    }


def save_materialized_dga_dataset(path: str | Path, bundle: MaterializedDgaDataset) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pose": bundle.arrays.pose.astype(np.float32),
        "pose_raw": bundle.arrays.pose_raw.astype(np.float32),
        "pose_full": bundle.arrays.pose_full.astype(np.float32),
        "object_points": bundle.arrays.object_points.astype(np.float32),
        "object_normals": bundle.arrays.object_normals.astype(np.float32),
        "contact_indices": bundle.arrays.contact_indices.astype(np.int32),
        "total_energy": bundle.arrays.total_energy.astype(np.float32),
        "sample_index": bundle.arrays.sample_index.astype(np.int32),
        "source_path": bundle.arrays.source_path.astype(np.str_),
        "hand_side": bundle.arrays.hand_side.astype(np.str_),
        "object_kind": bundle.arrays.object_kind.astype(np.str_),
        "object_name": bundle.arrays.object_name.astype(np.str_),
        "object_key": bundle.arrays.object_key.astype(np.str_),
        "translation_lower": bundle.normalizer.translation_lower.astype(np.float32),
        "translation_upper": bundle.normalizer.translation_upper.astype(np.float32),
        "joint_lower": bundle.normalizer.joint_lower.astype(np.float32),
        "joint_upper": bundle.normalizer.joint_upper.astype(np.float32),
        "metadata_json": np.asarray(json.dumps(dataset_metadata(bundle), sort_keys=True), dtype=np.str_),
    }
    np.savez_compressed(output_path, **payload)
    return output_path


def load_saved_normalizer(path: str | Path) -> DgaPoseNormalizer:
    dataset_path = Path(path).expanduser().resolve()
    with np.load(dataset_path, allow_pickle=False) as payload:
        state = {
            "translation_lower": np.asarray(payload["translation_lower"], dtype=np.float32),
            "translation_upper": np.asarray(payload["translation_upper"], dtype=np.float32),
            "joint_lower": np.asarray(payload["joint_lower"], dtype=np.float32),
            "joint_upper": np.asarray(payload["joint_upper"], dtype=np.float32),
        }
    return DgaPoseNormalizer.from_state_dict(state)
