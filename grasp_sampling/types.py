from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class MotionSpec(NamedTuple):
    name: str
    kind: str
    axis_name: str
    axis: np.ndarray
    direction: int


def default_motion_specs() -> tuple[MotionSpec, ...]:
    axes = (
        ("x", np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        ("y", np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        ("z", np.array([0.0, 0.0, 1.0], dtype=np.float32)),
    )
    motions: list[MotionSpec] = []
    for axis_name, axis in axes:
        motions.append(MotionSpec(name=f"t{axis_name}+", kind="translation", axis_name=axis_name, axis=axis, direction=1))
        motions.append(
            MotionSpec(name=f"t{axis_name}-", kind="translation", axis_name=axis_name, axis=axis, direction=-1)
        )
    for axis_name, axis in axes:
        motions.append(MotionSpec(name=f"r{axis_name}+", kind="rotation", axis_name=axis_name, axis=axis, direction=1))
        motions.append(MotionSpec(name=f"r{axis_name}-", kind="rotation", axis_name=axis_name, axis=axis, direction=-1))
    return tuple(motions)


@dataclass(frozen=True)
class SamplingEvalState:
    base_hand_pose: np.ndarray
    base_contact_indices: np.ndarray
    object_relative_pos: np.ndarray
    object_relative_quat: np.ndarray
    squeeze_deltas: np.ndarray
    qpos_targets: np.ndarray
    initial_contact_count: np.ndarray
    initial_penetration_count: np.ndarray
    initial_depth_sum: np.ndarray
    initial_max_depth: np.ndarray
    initial_overlap: np.ndarray
    motion_scores: np.ndarray
    motion_max_translation: np.ndarray
    motion_max_rotation_rad: np.ndarray
    motion_final_translation: np.ndarray
    motion_final_rotation_rad: np.ndarray
    motion_contact_min: np.ndarray
    motion_contact_final: np.ndarray
    motion_lost: np.ndarray
    motion_fail: np.ndarray
    motion_early_stop: np.ndarray
    motion_steps: np.ndarray
    overall_scores: np.ndarray
    success: np.ndarray
    chosen_attempt_index: int

    @property
    def attempt_count(self) -> int:
        return int(self.squeeze_deltas.shape[0])

    @property
    def motion_count(self) -> int:
        return int(self.motion_scores.shape[1])
