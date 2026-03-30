from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .types import RefineEnergyTerms, RefineResultState


CONTACT_LOCAL_EPS = 1.0e-8


@dataclass(frozen=True)
class RefineConfig:
    steps: int = 32
    step_size: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1.0e-8
    penetration_weight: float = 60.0
    contact_weight: float = 30.0
    support_weight: float = 12.0
    distance_weight: float = 0.1
    equilibrium_weight: float = 0.05
    root_reg_weight: float = 1.0
    joint_reg_weight: float = 0.1
    penetration_threshold: float = 5.0e-4
    support_points_per_body: int = 6
    support_distance_sigma: float = 1.5e-2
    support_max_distance: float = 3.0e-2
    actual_object_density: float = 400.0


@dataclass(frozen=True)
class SourceGrasp:
    source_path: Path
    state_name: str
    sample_index: int
    hand_side: str
    prop_meta: dict[str, object]
    hand_pose: np.ndarray
    contact_indices: np.ndarray


@dataclass(frozen=True)
class RefineArtifact:
    metadata: dict[str, object]
    state: RefineResultState


@dataclass(frozen=True)
class SingleRefineCallbacks:
    evaluate_terms_with_grad: Callable[
        [np.ndarray, np.ndarray, np.ndarray, RefineConfig],
        tuple[RefineEnergyTerms, np.ndarray, np.ndarray],
    ]
    active_pose_mask: Callable[[np.ndarray, np.ndarray, float], np.ndarray]
    actual_overlap_counts: Callable[[np.ndarray], tuple[int, int, float, float]]


def refine_source_grasp(
    source: SourceGrasp,
    *,
    contact_target_local: np.ndarray,
    callbacks: SingleRefineCallbacks,
    config: RefineConfig,
) -> RefineArtifact:
    initial_hand_pose = np.asarray(source.hand_pose, dtype=np.float32)
    contact_indices = np.asarray(source.contact_indices, dtype=np.int32)
    contact_target_local = np.asarray(contact_target_local, dtype=np.float32)

    initial_energy, projected_initial_hand_pose, _ = callbacks.evaluate_terms_with_grad(
        initial_hand_pose,
        contact_indices,
        contact_target_local,
        config,
    )
    best_hand_pose = projected_initial_hand_pose.copy()
    best_energy = initial_energy
    current_hand_pose = projected_initial_hand_pose.copy()

    adam_m = np.zeros_like(current_hand_pose, dtype=np.float32)
    adam_v = np.zeros_like(current_hand_pose, dtype=np.float32)

    history_total: list[float] = [initial_energy.total]
    history_distance: list[float] = [initial_energy.distance]
    history_equilibrium: list[float] = [initial_energy.equilibrium]
    history_penetration: list[float] = [initial_energy.penetration]
    history_contact: list[float] = [initial_energy.contact]
    history_root_reg: list[float] = [initial_energy.root_reg]
    history_joint_reg: list[float] = [initial_energy.joint_reg]
    history_active_joint_count: list[int] = [
        int(np.count_nonzero(callbacks.active_pose_mask(current_hand_pose, contact_indices, config.penetration_threshold)[9:]))
    ]

    final_mask = np.ones_like(current_hand_pose, dtype=np.float32)
    for step_index in range(int(config.steps)):
        final_mask = np.asarray(
            callbacks.active_pose_mask(current_hand_pose, contact_indices, config.penetration_threshold),
            dtype=np.float32,
        )
        current_energy, projected, grad_np = callbacks.evaluate_terms_with_grad(
            current_hand_pose,
            contact_indices,
            contact_target_local,
            config,
        )

        t = step_index + 1
        grad_np = np.nan_to_num(np.asarray(grad_np, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0) * final_mask
        adam_m = float(config.beta1) * adam_m + (1.0 - float(config.beta1)) * grad_np
        adam_v = float(config.beta2) * adam_v + (1.0 - float(config.beta2)) * np.square(grad_np)
        m_hat = adam_m / max(1.0 - float(config.beta1) ** t, CONTACT_LOCAL_EPS)
        v_hat = adam_v / max(1.0 - float(config.beta2) ** t, CONTACT_LOCAL_EPS)
        proposal = projected - float(config.step_size) * m_hat / (np.sqrt(v_hat) + float(config.adam_eps))
        proposal = proposal * final_mask + projected * (1.0 - final_mask)

        next_terms, next_projected, _ = callbacks.evaluate_terms_with_grad(
            np.asarray(proposal, dtype=np.float32),
            contact_indices,
            contact_target_local,
            config,
        )
        current_hand_pose = np.asarray(next_projected, dtype=np.float32)

        history_total.append(next_terms.total)
        history_distance.append(next_terms.distance)
        history_equilibrium.append(next_terms.equilibrium)
        history_penetration.append(next_terms.penetration)
        history_contact.append(next_terms.contact)
        history_root_reg.append(next_terms.root_reg)
        history_joint_reg.append(next_terms.joint_reg)
        history_active_joint_count.append(int(np.count_nonzero(final_mask[9:])))

        if next_terms.total < best_energy.total:
            best_energy = next_terms
            best_hand_pose = current_hand_pose.copy()

    final_energy, refined_hand_pose, _ = callbacks.evaluate_terms_with_grad(
        current_hand_pose,
        contact_indices,
        contact_target_local,
        config,
    )
    initial_actual = callbacks.actual_overlap_counts(projected_initial_hand_pose)
    final_actual = callbacks.actual_overlap_counts(refined_hand_pose)
    best_actual = callbacks.actual_overlap_counts(best_hand_pose)

    metadata: dict[str, object] = {
        "source": {
            "result_path": str(source.source_path),
            "state": source.state_name,
            "sample_index": int(source.sample_index),
        },
        "hand": {"side": source.hand_side},
        "prop": dict(source.prop_meta),
        "config": asdict(config),
        "result": {
            "improved": bool(best_energy.total < initial_energy.total),
            "initial_total": float(initial_energy.total),
            "final_total": float(final_energy.total),
            "best_total": float(best_energy.total),
        },
    }
    state = RefineResultState(
        initial_hand_pose=projected_initial_hand_pose,
        refined_hand_pose=refined_hand_pose,
        best_hand_pose=best_hand_pose,
        contact_indices=contact_indices,
        contact_target_local=np.asarray(contact_target_local, dtype=np.float32),
        final_active_mask=np.asarray(final_mask, dtype=np.float32),
        history_total=np.asarray(history_total, dtype=np.float32),
        history_distance=np.asarray(history_distance, dtype=np.float32),
        history_equilibrium=np.asarray(history_equilibrium, dtype=np.float32),
        history_penetration=np.asarray(history_penetration, dtype=np.float32),
        history_contact=np.asarray(history_contact, dtype=np.float32),
        history_root_reg=np.asarray(history_root_reg, dtype=np.float32),
        history_joint_reg=np.asarray(history_joint_reg, dtype=np.float32),
        history_active_joint_count=np.asarray(history_active_joint_count, dtype=np.int32),
        initial_energy=initial_energy,
        final_energy=final_energy,
        best_energy=best_energy,
        initial_actual_contact_count=int(initial_actual[0]),
        initial_actual_penetration_count=int(initial_actual[1]),
        initial_actual_depth_sum=float(initial_actual[2]),
        initial_actual_max_depth=float(initial_actual[3]),
        final_actual_contact_count=int(final_actual[0]),
        final_actual_penetration_count=int(final_actual[1]),
        final_actual_depth_sum=float(final_actual[2]),
        final_actual_max_depth=float(final_actual[3]),
        best_actual_contact_count=int(best_actual[0]),
        best_actual_penetration_count=int(best_actual[1]),
        best_actual_depth_sum=float(best_actual[2]),
        best_actual_max_depth=float(best_actual[3]),
    )
    return RefineArtifact(metadata=metadata, state=state)
