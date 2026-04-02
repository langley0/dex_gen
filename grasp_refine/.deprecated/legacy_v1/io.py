from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .types import ArtifactStateName, DatasetConfig


@dataclass(frozen=True)
class GraspRecord:
    source_path: Path
    sample_index: int
    state_name: ArtifactStateName
    hand_side: str
    object_kind: str
    object_name: str
    object_metadata: dict[str, Any]
    pose: np.ndarray
    contact_indices: np.ndarray
    total_energy: float


@dataclass(frozen=True)
class GraspArtifactPayload:
    path: Path
    metadata: dict[str, Any]
    records: tuple[GraspRecord, ...]


def resolve_artifact_paths(config: DatasetConfig) -> list[Path]:
    paths = [Path(path).expanduser().resolve() for path in config.artifact_paths]
    if config.artifact_glob:
        paths.extend(sorted(Path().resolve().glob(config.artifact_glob)))
    unique_paths = sorted({path.resolve() for path in paths if path.exists() and path.suffix == ".npz"})
    if not unique_paths:
        raise ValueError("No grasp artifact .npz files were found for training.")
    return unique_paths


def _state_arrays(
    payload: dict[str, np.ndarray],
    state_name: ArtifactStateName,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if state_name == "best":
        pose = np.asarray(payload["best_hand_pose"], dtype=np.float32)
        contact_indices = np.asarray(payload["best_contact_indices"], dtype=np.int32)
        total = np.asarray(payload["best_energy_total"], dtype=np.float32)
    elif state_name == "current":
        pose = np.asarray(payload["hand_pose"], dtype=np.float32)
        contact_indices = np.asarray(payload["contact_indices"], dtype=np.int32)
        total = np.asarray(payload["energy_total"], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported state name: {state_name!r}")
    return pose, contact_indices, total


def load_grasp_artifact(path: str | Path, *, state_name: ArtifactStateName = "best") -> GraspArtifactPayload:
    artifact_path = Path(path).expanduser().resolve()
    with np.load(artifact_path, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}

    metadata = json.loads(str(payload.pop("metadata_json").item()))
    hand_pose, contact_indices, total_energy = _state_arrays(payload, state_name)
    hand_side = str(metadata.get("hand", {}).get("side", "right"))
    prop_meta = dict(metadata.get("prop", {}))
    object_kind = str(prop_meta.get("kind", "unknown"))
    object_name = str(prop_meta.get("name", object_kind))

    records = []
    for sample_index in range(hand_pose.shape[0]):
        records.append(
            GraspRecord(
                source_path=artifact_path,
                sample_index=sample_index,
                state_name=state_name,
                hand_side=hand_side,
                object_kind=object_kind,
                object_name=object_name,
                object_metadata=prop_meta,
                pose=np.asarray(hand_pose[sample_index], dtype=np.float32),
                contact_indices=np.asarray(contact_indices[sample_index], dtype=np.int32),
                total_energy=float(total_energy[sample_index]),
            )
        )

    return GraspArtifactPayload(path=artifact_path, metadata=metadata, records=tuple(records))


def load_grasp_records(config: DatasetConfig) -> list[GraspRecord]:
    records: list[GraspRecord] = []
    for path in resolve_artifact_paths(config):
        payload = load_grasp_artifact(path, state_name=config.state_name)
        for record in payload.records:
            if config.drop_invalid_samples and not np.all(np.isfinite(record.pose)):
                continue
            records.append(record)
    if not records:
        raise ValueError("No valid grasp samples were loaded from the provided artifacts.")
    return records
