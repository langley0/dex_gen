from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from .materialize import DgaDatasetArrays
from .normalization import DgaPoseNormalizer
from .object_identity import build_saved_object_key


SplitMode = Literal["sample", "object", "object_random", "object_fixed"]


@dataclass(frozen=True)
class LoadedDgaDataset:
    path: Path
    arrays: DgaDatasetArrays
    normalizer: DgaPoseNormalizer
    metadata: dict[str, Any]

    def __len__(self) -> int:
        return int(self.arrays.pose.shape[0])

    def __getitem__(self, index: int) -> dict[str, np.ndarray | np.generic | str]:
        idx = int(index)
        return {
            "pose": np.asarray(self.arrays.pose[idx], dtype=np.float32),
            "pose_raw": np.asarray(self.arrays.pose_raw[idx], dtype=np.float32),
            "pose_full": np.asarray(self.arrays.pose_full[idx], dtype=np.float32),
            "object_points": np.asarray(self.arrays.object_points[idx], dtype=np.float32),
            "object_normals": np.asarray(self.arrays.object_normals[idx], dtype=np.float32),
            "contact_indices": np.asarray(self.arrays.contact_indices[idx], dtype=np.int32),
            "total_energy": np.asarray(self.arrays.total_energy[idx], dtype=np.float32),
            "sample_index": np.asarray(self.arrays.sample_index[idx], dtype=np.int32),
            "source_path": str(self.arrays.source_path[idx]),
            "hand_side": str(self.arrays.hand_side[idx]),
            "object_kind": str(self.arrays.object_kind[idx]),
            "object_name": str(self.arrays.object_name[idx]),
            "object_key": str(self.arrays.object_key[idx]),
        }


@dataclass(frozen=True)
class DgaDatasetSubset:
    dataset: LoadedDgaDataset
    indices: np.ndarray

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, np.ndarray | np.generic | str]:
        source_index = int(self.indices[int(index)])
        return self.dataset[source_index]


@dataclass(frozen=True)
class DgaBatch:
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


def _load_array(payload: Any, key: str, dtype: np.dtype[Any]) -> np.ndarray:
    return np.asarray(payload[key], dtype=dtype)


def load_saved_dga_dataset(path: str | Path) -> LoadedDgaDataset:
    dataset_path = Path(path).expanduser().resolve()
    with np.load(dataset_path, allow_pickle=False) as payload:
        object_kind = _load_array(payload, "object_kind", np.str_)
        object_name = _load_array(payload, "object_name", np.str_)
        if "object_key" in payload:
            object_key = _load_array(payload, "object_key", np.str_)
        else:
            object_key = np.asarray(
                [
                    build_saved_object_key(
                        object_kind=str(kind),
                        object_name=str(name),
                    )
                    for kind, name in zip(object_kind, object_name, strict=False)
                ],
                dtype=np.str_,
            )
        arrays = DgaDatasetArrays(
            pose=_load_array(payload, "pose", np.float32),
            pose_raw=_load_array(payload, "pose_raw", np.float32),
            pose_full=_load_array(payload, "pose_full", np.float32),
            object_points=_load_array(payload, "object_points", np.float32),
            object_normals=_load_array(payload, "object_normals", np.float32),
            contact_indices=_load_array(payload, "contact_indices", np.int32),
            total_energy=_load_array(payload, "total_energy", np.float32),
            sample_index=_load_array(payload, "sample_index", np.int32),
            source_path=_load_array(payload, "source_path", np.str_),
            hand_side=_load_array(payload, "hand_side", np.str_),
            object_kind=object_kind,
            object_name=object_name,
            object_key=object_key,
        )
        normalizer = DgaPoseNormalizer.from_state_dict(
            {
                "translation_lower": _load_array(payload, "translation_lower", np.float32),
                "translation_upper": _load_array(payload, "translation_upper", np.float32),
                "joint_lower": _load_array(payload, "joint_lower", np.float32),
                "joint_upper": _load_array(payload, "joint_upper", np.float32),
            }
        )
        metadata = json.loads(str(payload["metadata_json"].item()))
    return LoadedDgaDataset(
        path=dataset_path,
        arrays=arrays,
        normalizer=normalizer,
        metadata=metadata,
    )


def collate_dga_batch(samples: Sequence[dict[str, np.ndarray | np.generic | str]]) -> DgaBatch:
    if not samples:
        raise ValueError("samples must contain at least one item.")
    return DgaBatch(
        pose=np.stack([np.asarray(sample["pose"], dtype=np.float32) for sample in samples], axis=0),
        pose_raw=np.stack([np.asarray(sample["pose_raw"], dtype=np.float32) for sample in samples], axis=0),
        pose_full=np.stack([np.asarray(sample["pose_full"], dtype=np.float32) for sample in samples], axis=0),
        object_points=np.stack([np.asarray(sample["object_points"], dtype=np.float32) for sample in samples], axis=0),
        object_normals=np.stack([np.asarray(sample["object_normals"], dtype=np.float32) for sample in samples], axis=0),
        contact_indices=np.stack([np.asarray(sample["contact_indices"], dtype=np.int32) for sample in samples], axis=0),
        total_energy=np.stack([np.asarray(sample["total_energy"], dtype=np.float32) for sample in samples], axis=0),
        sample_index=np.stack([np.asarray(sample["sample_index"], dtype=np.int32) for sample in samples], axis=0),
        source_path=np.asarray([str(sample["source_path"]) for sample in samples], dtype=np.str_),
        hand_side=np.asarray([str(sample["hand_side"]) for sample in samples], dtype=np.str_),
        object_kind=np.asarray([str(sample["object_kind"]) for sample in samples], dtype=np.str_),
        object_name=np.asarray([str(sample["object_name"]) for sample in samples], dtype=np.str_),
        object_key=np.asarray([str(sample["object_key"]) for sample in samples], dtype=np.str_),
    )


def split_dga_dataset(
    dataset: LoadedDgaDataset,
    *,
    train_fraction: float,
    seed: int,
    split_mode: SplitMode = "object",
    train_object_keys: Sequence[str] | None = None,
    val_object_keys: Sequence[str] | None = None,
) -> tuple[DgaDatasetSubset, DgaDatasetSubset | None]:
    if len(dataset) <= 0:
        raise ValueError("dataset must contain at least one sample.")
    if not (0.0 < float(train_fraction) <= 1.0):
        raise ValueError("train_fraction must be in (0, 1].")
    if split_mode != "object_fixed" and float(train_fraction) >= 1.0:
        full_indices = np.arange(len(dataset), dtype=np.int32)
        return DgaDatasetSubset(dataset=dataset, indices=full_indices), None

    if split_mode == "sample":
        indices = np.arange(len(dataset), dtype=np.int32)
        rng = np.random.default_rng(int(seed))
        rng.shuffle(indices)
        train_size = max(1, int(round(len(indices) * float(train_fraction))))
        train_size = min(train_size, len(indices))
        train_indices = np.sort(indices[:train_size]).astype(np.int32)
        val_indices = np.sort(indices[train_size:]).astype(np.int32)
    elif split_mode in ("object", "object_random"):
        object_keys = np.asarray(dataset.arrays.object_key, dtype=np.str_)
        unique_object_keys = np.unique(object_keys)
        rng = np.random.default_rng(int(seed))
        shuffled_object_keys = unique_object_keys.copy()
        rng.shuffle(shuffled_object_keys)
        train_key_count = max(1, int(round(len(shuffled_object_keys) * float(train_fraction))))
        train_key_count = min(train_key_count, len(shuffled_object_keys))
        train_key_set = set(str(key) for key in shuffled_object_keys[:train_key_count])
        train_mask = np.asarray([str(key) in train_key_set for key in object_keys], dtype=bool)
        train_indices = np.nonzero(train_mask)[0].astype(np.int32)
        val_indices = np.nonzero(~train_mask)[0].astype(np.int32)
    elif split_mode == "object_fixed":
        object_keys = np.asarray(dataset.arrays.object_key, dtype=np.str_)
        unique_object_keys = {str(key) for key in np.unique(object_keys)}
        if train_object_keys is None or len(train_object_keys) == 0:
            raise ValueError("train_object_keys must be provided for object_fixed split.")
        train_key_set = {str(key) for key in train_object_keys}
        unknown_train_keys = sorted(train_key_set - unique_object_keys)
        if unknown_train_keys:
            raise ValueError(f"Unknown train object keys: {unknown_train_keys}")
        if val_object_keys is None:
            val_key_set = unique_object_keys - train_key_set
        else:
            val_key_set = {str(key) for key in val_object_keys}
            unknown_val_keys = sorted(val_key_set - unique_object_keys)
            if unknown_val_keys:
                raise ValueError(f"Unknown val object keys: {unknown_val_keys}")
        overlap = sorted(train_key_set & val_key_set)
        if overlap:
            raise ValueError(f"train/val object keys must be disjoint, got overlap: {overlap}")
        assigned_key_set = train_key_set | val_key_set
        unassigned = sorted(unique_object_keys - assigned_key_set)
        if unassigned:
            raise ValueError(f"object_fixed split leaves unassigned object keys: {unassigned}")
        train_mask = np.asarray([str(key) in train_key_set for key in object_keys], dtype=bool)
        train_indices = np.nonzero(train_mask)[0].astype(np.int32)
        val_indices = np.nonzero(~train_mask)[0].astype(np.int32)
    else:
        raise ValueError(f"Unsupported split mode: {split_mode!r}")

    train_subset = DgaDatasetSubset(dataset=dataset, indices=train_indices)
    val_subset = None if val_indices.size == 0 else DgaDatasetSubset(dataset=dataset, indices=val_indices)
    return train_subset, val_subset


def iterate_dga_batches(
    dataset: Sequence[dict[str, np.ndarray | np.generic | str]] | LoadedDgaDataset | DgaDatasetSubset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    drop_last: bool = False,
) -> Iterable[DgaBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    indices = np.arange(len(dataset), dtype=np.int32)
    if shuffle:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        if drop_last and batch_indices.shape[0] < batch_size:
            continue
        yield collate_dga_batch([dataset[int(index)] for index in batch_indices])
