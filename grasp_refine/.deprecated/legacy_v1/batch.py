from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np


# Pylance does not treat jax.Device as a valid static type in annotations.
JaxDevice: TypeAlias = Any


@dataclass(frozen=True)
class GraspBatch:
    pose: np.ndarray | jax.Array
    pose_raw: np.ndarray | jax.Array
    object_points: np.ndarray | jax.Array
    object_normals: np.ndarray | jax.Array
    object_index: np.ndarray | jax.Array
    contact_indices: np.ndarray | jax.Array
    energy: np.ndarray | jax.Array

    def as_jax(self, device: JaxDevice | None = None) -> dict[str, jax.Array]:
        source_values = (
            self.pose,
            self.pose_raw,
            self.object_points,
            self.object_normals,
            self.object_index,
            self.contact_indices,
            self.energy,
        )
        batch = {
            "pose": jnp.asarray(self.pose, dtype=jnp.float32),
            "pose_raw": jnp.asarray(self.pose_raw, dtype=jnp.float32),
            "object_points": jnp.asarray(self.object_points, dtype=jnp.float32),
            "object_normals": jnp.asarray(self.object_normals, dtype=jnp.float32),
            "object_index": jnp.asarray(self.object_index, dtype=jnp.int32),
            "contact_indices": jnp.asarray(self.contact_indices, dtype=jnp.int32),
            "energy": jnp.asarray(self.energy, dtype=jnp.float32),
        }
        if all(isinstance(value, jax.Array) for value in source_values):
            return batch
        if device is not None:
            return {key: jax.device_put(value, device) for key, value in batch.items()}
        return batch


@dataclass(frozen=True)
class GraspDatasetArrays:
    pose: np.ndarray | jax.Array
    pose_raw: np.ndarray | jax.Array
    object_points: np.ndarray | jax.Array
    object_normals: np.ndarray | jax.Array
    object_index: np.ndarray | jax.Array
    contact_indices: np.ndarray | jax.Array
    energy: np.ndarray | jax.Array

    def __len__(self) -> int:
        return int(self.pose.shape[0])


def collate_grasp_batch(samples: Sequence[dict[str, np.ndarray]]) -> GraspBatch:
    return GraspBatch(
        pose=np.stack([sample["pose"] for sample in samples], axis=0).astype(np.float32),
        pose_raw=np.stack([sample["pose_raw"] for sample in samples], axis=0).astype(np.float32),
        object_points=np.stack([sample["object_points"] for sample in samples], axis=0).astype(np.float32),
        object_normals=np.stack([sample["object_normals"] for sample in samples], axis=0).astype(np.float32),
        object_index=np.stack([sample["object_index"] for sample in samples], axis=0).astype(np.int32),
        contact_indices=np.stack([sample["contact_indices"] for sample in samples], axis=0).astype(np.int32),
        energy=np.stack([sample["energy"] for sample in samples], axis=0).astype(np.float32),
    )


def materialize_grasp_dataset(dataset: Sequence[dict[str, np.ndarray]]) -> GraspDatasetArrays:
    samples = [dataset[int(index)] for index in range(len(dataset))]
    if not samples:
        raise ValueError("dataset must contain at least one sample.")
    batch = collate_grasp_batch(samples)
    return GraspDatasetArrays(
        pose=batch.pose,
        pose_raw=batch.pose_raw,
        object_points=batch.object_points,
        object_normals=batch.object_normals,
        object_index=batch.object_index,
        contact_indices=batch.contact_indices,
        energy=batch.energy,
    )


def place_grasp_dataset_on_device(dataset: GraspDatasetArrays, device: JaxDevice) -> GraspDatasetArrays:
    return GraspDatasetArrays(
        pose=jax.device_put(jnp.asarray(dataset.pose, dtype=jnp.float32), device),
        pose_raw=jax.device_put(jnp.asarray(dataset.pose_raw, dtype=jnp.float32), device),
        object_points=jax.device_put(jnp.asarray(dataset.object_points, dtype=jnp.float32), device),
        object_normals=jax.device_put(jnp.asarray(dataset.object_normals, dtype=jnp.float32), device),
        object_index=jax.device_put(jnp.asarray(dataset.object_index, dtype=jnp.int32), device),
        contact_indices=jax.device_put(jnp.asarray(dataset.contact_indices, dtype=jnp.int32), device),
        energy=jax.device_put(jnp.asarray(dataset.energy, dtype=jnp.float32), device),
    )


def iterate_grasp_batches(
    dataset: Sequence[dict[str, np.ndarray]] | GraspDatasetArrays,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterable[GraspBatch]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    indices = np.arange(len(dataset), dtype=np.int32)
    if shuffle:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        if isinstance(dataset, GraspDatasetArrays):
            yield GraspBatch(
                pose=dataset.pose[batch_indices],
                pose_raw=dataset.pose_raw[batch_indices],
                object_points=dataset.object_points[batch_indices],
                object_normals=dataset.object_normals[batch_indices],
                object_index=dataset.object_index[batch_indices],
                contact_indices=dataset.contact_indices[batch_indices],
                energy=dataset.energy[batch_indices],
            )
        else:
            yield collate_grasp_batch([dataset[int(index)] for index in batch_indices])
