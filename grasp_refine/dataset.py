from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import ortho6d_to_matrix, quat_to_matrix
from .io import SourceGraspRecord, load_source_records
from .object_mesh import MeshData, load_object_mesh, sample_mesh_points
from .types import CoordinateMode, DatasetConfig


@dataclass(frozen=True)
class DgaDataRecord:
    source_path: str
    sample_index: int
    hand_side: str
    object_kind: str
    object_name: str
    object_key: str
    pose: np.ndarray
    pose_full: np.ndarray
    object_points: np.ndarray
    object_normals: np.ndarray
    contact_indices: np.ndarray
    total_energy: float


def _stable_kind_seed(object_kind: str, base_seed: int) -> int:
    stable_hash = sum((index + 1) * ord(char) for index, char in enumerate(object_kind))
    return int(base_seed) + stable_hash % 100_000


def _dga_pose_from_full_pose(full_pose: np.ndarray) -> np.ndarray:
    pose_array = np.asarray(full_pose, dtype=np.float32).reshape(-1)
    if pose_array.shape[0] < 10:
        raise ValueError(f"Expected full pose with root pose and joints, got shape {pose_array.shape}.")
    return np.concatenate([pose_array[:3], pose_array[9:]], axis=0).astype(np.float32)


def _canonical_object_cloud(record: SourceGraspRecord, config: DatasetConfig, cache: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    cached = cache.get(record.object_kind)
    if cached is not None:
        return cached
    mesh = load_object_mesh(record.object_metadata)
    seed = _stable_kind_seed(record.object_kind, config.object_point_seed)
    points, normals = sample_mesh_points(mesh, config.object_num_points, seed=seed)
    cached = (points.astype(np.float32), normals.astype(np.float32))
    cache[record.object_kind] = cached
    return cached


def _world_object_cloud(record: SourceGraspRecord, canonical_points: np.ndarray, canonical_normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    object_pos = np.asarray(record.object_metadata.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
    object_quat = np.asarray(record.object_metadata.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32).reshape(4)
    object_rotation = quat_to_matrix(object_quat)
    world_points = canonical_points @ object_rotation.T + object_pos[None, :]
    world_normals = canonical_normals @ object_rotation.T
    return world_points.astype(np.float32), world_normals.astype(np.float32)


def _transform_object_cloud(
    record: SourceGraspRecord,
    canonical_points: np.ndarray,
    canonical_normals: np.ndarray,
    *,
    coordinate_mode: CoordinateMode,
) -> tuple[np.ndarray, np.ndarray]:
    world_points, world_normals = _world_object_cloud(record, canonical_points, canonical_normals)
    if coordinate_mode == "world_object_rotated":
        return world_points, world_normals
    if coordinate_mode != "hand_aligned_object":
        raise ValueError(f"Unsupported coordinate mode: {coordinate_mode!r}")

    hand_root_pos = np.asarray(record.full_pose[:3], dtype=np.float32).reshape(3)
    hand_root_rot = ortho6d_to_matrix(np.asarray(record.full_pose[3:9], dtype=np.float32))

    # Rotate the object cloud into a frame where the hand root rotation becomes
    # identity while preserving the same hand-object geometry.
    aligned_points = (world_points - hand_root_pos[None, :]) @ hand_root_rot + hand_root_pos[None, :]
    aligned_normals = world_normals @ hand_root_rot
    return aligned_points.astype(np.float32), aligned_normals.astype(np.float32)


def build_dga_data_records(config: DatasetConfig) -> list[DgaDataRecord]:
    source_records = load_source_records(config)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    converted: list[DgaDataRecord] = []
    for record in source_records:
        canonical_points, canonical_normals = _canonical_object_cloud(record, config, cache)
        object_points, object_normals = _transform_object_cloud(
            record,
            canonical_points,
            canonical_normals,
            coordinate_mode=config.coordinate_mode,
        )
        converted.append(
            DgaDataRecord(
                source_path=str(record.source_path),
                sample_index=record.sample_index,
                hand_side=record.hand_side,
                object_kind=record.object_kind,
                object_name=record.object_name,
                object_key=record.object_key,
                pose=_dga_pose_from_full_pose(record.full_pose),
                pose_full=np.asarray(record.full_pose, dtype=np.float32),
                object_points=object_points,
                object_normals=object_normals,
                contact_indices=np.asarray(record.contact_indices, dtype=np.int32),
                total_energy=float(record.total_energy),
            )
        )
    return converted
