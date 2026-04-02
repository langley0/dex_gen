from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class PoseNormalizer:
    translation_lower: np.ndarray
    translation_upper: np.ndarray
    joint_lower: np.ndarray
    joint_upper: np.ndarray

    @property
    def pose_dim(self) -> int:
        return 3 + 6 + int(self.joint_lower.shape[0])

    def normalize_numpy(self, pose: np.ndarray) -> np.ndarray:
        pose_array = np.asarray(pose, dtype=np.float32)
        result = pose_array.copy()
        result[..., :3] = self._normalize_block_np(pose_array[..., :3], self.translation_lower, self.translation_upper)
        result[..., 3:9] = pose_array[..., 3:9]
        result[..., 9:] = self._normalize_block_np(pose_array[..., 9:], self.joint_lower, self.joint_upper)
        return result.astype(np.float32)

    def denormalize_jax(self, pose: jax.Array) -> jax.Array:
        translation_lower = jnp.asarray(self.translation_lower, dtype=pose.dtype)
        translation_upper = jnp.asarray(self.translation_upper, dtype=pose.dtype)
        joint_lower = jnp.asarray(self.joint_lower, dtype=pose.dtype)
        joint_upper = jnp.asarray(self.joint_upper, dtype=pose.dtype)
        root = self._denormalize_block_jax(pose[..., :3], translation_lower, translation_upper)
        rot = pose[..., 3:9]
        joints = self._denormalize_block_jax(pose[..., 9:], joint_lower, joint_upper)
        return jnp.concatenate([root, rot, joints], axis=-1)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "translation_lower": self.translation_lower.astype(np.float32),
            "translation_upper": self.translation_upper.astype(np.float32),
            "joint_lower": self.joint_lower.astype(np.float32),
            "joint_upper": self.joint_upper.astype(np.float32),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, np.ndarray]) -> "PoseNormalizer":
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
    def _denormalize_block_jax(values: jax.Array, lower: jax.Array, upper: jax.Array) -> jax.Array:
        restored = (values + 1.0) * 0.5
        return restored * (upper - lower) + lower
