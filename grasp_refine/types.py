from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RefineEnergyTerms:
    total: float
    distance: float
    equilibrium: float
    penetration: float
    contact: float
    root_reg: float
    joint_reg: float


@dataclass(frozen=True)
class RefineResultState:
    initial_hand_pose: np.ndarray
    refined_hand_pose: np.ndarray
    best_hand_pose: np.ndarray
    contact_indices: np.ndarray
    contact_target_local: np.ndarray
    final_active_mask: np.ndarray
    history_total: np.ndarray
    history_distance: np.ndarray
    history_equilibrium: np.ndarray
    history_penetration: np.ndarray
    history_contact: np.ndarray
    history_root_reg: np.ndarray
    history_joint_reg: np.ndarray
    history_active_joint_count: np.ndarray
    initial_energy: RefineEnergyTerms
    final_energy: RefineEnergyTerms
    best_energy: RefineEnergyTerms
    initial_actual_contact_count: int
    initial_actual_penetration_count: int
    initial_actual_depth_sum: float
    initial_actual_max_depth: float
    final_actual_contact_count: int
    final_actual_penetration_count: int
    final_actual_depth_sum: float
    final_actual_max_depth: float
    best_actual_contact_count: int
    best_actual_penetration_count: int
    best_actual_depth_sum: float
    best_actual_max_depth: float


@dataclass(frozen=True)
class RefineRunArtifact:
    path: Path
    metadata: dict[str, object]
    state: RefineResultState
