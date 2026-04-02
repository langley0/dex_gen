from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .dataset import DgaDataRecord
from .hand_spec import DgaHandSpec


@dataclass(frozen=True)
class DgaPoseNormalizer:
    translation_lower: np.ndarray
    translation_upper: np.ndarray
    joint_lower: np.ndarray
    joint_upper: np.ndarray

    @property
    def pose_dim(self) -> int:
        return 3 + int(self.joint_lower.shape[0])

    def normalize_numpy(self, pose: np.ndarray) -> np.ndarray:
        pose_array = np.asarray(pose, dtype=np.float32)
        if pose_array.shape[-1] != self.pose_dim:
            raise ValueError(f"Expected pose dim {self.pose_dim}, got {pose_array.shape[-1]}.")
        result = pose_array.copy()
        result[..., :3] = self._normalize_block_np(pose_array[..., :3], self.translation_lower, self.translation_upper)
        result[..., 3:] = self._normalize_block_np(pose_array[..., 3:], self.joint_lower, self.joint_upper)
        return result.astype(np.float32)

    def denormalize_numpy(self, pose: np.ndarray) -> np.ndarray:
        pose_array = np.asarray(pose, dtype=np.float32)
        if pose_array.shape[-1] != self.pose_dim:
            raise ValueError(f"Expected pose dim {self.pose_dim}, got {pose_array.shape[-1]}.")
        result = pose_array.copy()
        result[..., :3] = self._denormalize_block_np(pose_array[..., :3], self.translation_lower, self.translation_upper)
        result[..., 3:] = self._denormalize_block_np(pose_array[..., 3:], self.joint_lower, self.joint_upper)
        return result.astype(np.float32)

    def denormalize_jax(self, pose: jax.Array) -> jax.Array:
        if pose.shape[-1] != self.pose_dim:
            raise ValueError(f"Expected pose dim {self.pose_dim}, got {pose.shape[-1]}.")
        translation_lower = jnp.asarray(self.translation_lower, dtype=pose.dtype)
        translation_upper = jnp.asarray(self.translation_upper, dtype=pose.dtype)
        joint_lower = jnp.asarray(self.joint_lower, dtype=pose.dtype)
        joint_upper = jnp.asarray(self.joint_upper, dtype=pose.dtype)
        root = self._denormalize_block_jax(pose[..., :3], translation_lower, translation_upper)
        joints = self._denormalize_block_jax(pose[..., 3:], joint_lower, joint_upper)
        return jnp.concatenate([root, joints], axis=-1)

    def project_pose_numpy(self, pose: np.ndarray) -> np.ndarray:
        pose_array = np.asarray(pose, dtype=np.float32)
        if pose_array.shape[-1] != self.pose_dim:
            raise ValueError(f"Expected pose dim {self.pose_dim}, got {pose_array.shape[-1]}.")
        result = pose_array.copy()
        result[..., :3] = np.clip(result[..., :3], self.translation_lower, self.translation_upper)
        result[..., 3:] = np.clip(result[..., 3:], self.joint_lower, self.joint_upper)
        return result.astype(np.float32)

    def project_pose_jax(self, pose: jax.Array) -> jax.Array:
        if pose.shape[-1] != self.pose_dim:
            raise ValueError(f"Expected pose dim {self.pose_dim}, got {pose.shape[-1]}.")
        translation_lower = jnp.asarray(self.translation_lower, dtype=pose.dtype)
        translation_upper = jnp.asarray(self.translation_upper, dtype=pose.dtype)
        joint_lower = jnp.asarray(self.joint_lower, dtype=pose.dtype)
        joint_upper = jnp.asarray(self.joint_upper, dtype=pose.dtype)
        root = jnp.clip(pose[..., :3], translation_lower, translation_upper)
        joints = jnp.clip(pose[..., 3:], joint_lower, joint_upper)
        return jnp.concatenate([root, joints], axis=-1)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "translation_lower": self.translation_lower.astype(np.float32),
            "translation_upper": self.translation_upper.astype(np.float32),
            "joint_lower": self.joint_lower.astype(np.float32),
            "joint_upper": self.joint_upper.astype(np.float32),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, np.ndarray]) -> "DgaPoseNormalizer":
        return cls(
            translation_lower=np.asarray(state["translation_lower"], dtype=np.float32),
            translation_upper=np.asarray(state["translation_upper"], dtype=np.float32),
            joint_lower=np.asarray(state["joint_lower"], dtype=np.float32),
            joint_upper=np.asarray(state["joint_upper"], dtype=np.float32),
        )

    @staticmethod
    def _normalize_block_np(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        denom = np.maximum(upper - lower, 1.0e-6)
        normalized = (values - lower) / denom
        return normalized * 2.0 - 1.0

    @staticmethod
    def _denormalize_block_np(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        restored = (values + 1.0) * 0.5
        return restored * (upper - lower) + lower

    @staticmethod
    def _denormalize_block_jax(values: jax.Array, lower: jax.Array, upper: jax.Array) -> jax.Array:
        restored = (values + 1.0) * 0.5
        return restored * (upper - lower) + lower


def build_pose_normalizer(
    records: Sequence[DgaDataRecord],
    hand_spec: DgaHandSpec,
    *,
    padding: float = 0.02,
) -> DgaPoseNormalizer:
    if not records:
        raise ValueError("records must contain at least one sample.")
    poses = np.stack([np.asarray(record.pose, dtype=np.float32) for record in records], axis=0)
    if poses.shape[1] != hand_spec.pose_dim:
        raise ValueError(f"Pose dim {poses.shape[1]} does not match hand spec dim {hand_spec.pose_dim}.")
    translation_lower = poses[:, :3].min(axis=0) - float(padding)
    translation_upper = poses[:, :3].max(axis=0) + float(padding)
    return DgaPoseNormalizer(
        translation_lower=translation_lower.astype(np.float32),
        translation_upper=translation_upper.astype(np.float32),
        joint_lower=hand_spec.joint_lower.astype(np.float32),
        joint_upper=hand_spec.joint_upper.astype(np.float32),
    )


def normalize_records(
    records: Sequence[DgaDataRecord],
    normalizer: DgaPoseNormalizer,
) -> list[DgaDataRecord]:
    normalized: list[DgaDataRecord] = []
    for record in records:
        normalized.append(
            DgaDataRecord(
                source_path=record.source_path,
                sample_index=record.sample_index,
                hand_side=record.hand_side,
                object_kind=record.object_kind,
                object_name=record.object_name,
                object_key=record.object_key,
                pose=normalizer.normalize_numpy(record.pose),
                pose_full=np.asarray(record.pose_full, dtype=np.float32),
                object_points=np.asarray(record.object_points, dtype=np.float32),
                object_normals=np.asarray(record.object_normals, dtype=np.float32),
                contact_indices=np.asarray(record.contact_indices, dtype=np.int32),
                total_energy=float(record.total_energy),
            )
        )
    return normalized
