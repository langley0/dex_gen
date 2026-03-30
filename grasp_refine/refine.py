from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .types import RefineEnergyTerms, RefineResultState


CONTACT_LOCAL_EPS = 1.0e-8
GRAD_ACTIVITY_EPS = 1.0e-4


@dataclass(frozen=True)
class RefineConfig:
    steps: int = 32
    guidance_scale: float = 1.0
    grad_scale: float = 0.1
    noise_scale_start: float = 1.0e-2
    noise_scale_end: float = 1.0e-3
    surface_pull_weight: float = 1.0
    external_repulsion_weight: float = 0.3
    self_repulsion_weight: float = 1.0
    surface_pull_threshold: float = 2.0e-2
    self_repulsion_threshold: float = 2.0e-2
    external_threshold: float = 1.0e-3
    grad_clip_norm: float = 1.0
    actual_object_density: float = 400.0
    seed: int = 0


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
    sample_scales: Callable[[], np.ndarray]
    actual_overlap_counts: Callable[[np.ndarray], tuple[int, int, float, float]]


def _scheduled_noise_scale(config: RefineConfig, step_index: int) -> float:
    if int(config.steps) <= 1:
        return float(config.noise_scale_end)
    alpha = float(step_index) / float(max(int(config.steps) - 1, 1))
    return float(config.noise_scale_start) + (float(config.noise_scale_end) - float(config.noise_scale_start)) * alpha


def _active_mask_from_grad(grad: np.ndarray) -> np.ndarray:
    grad = np.asarray(grad, dtype=np.float32)
    return (np.abs(grad) > GRAD_ACTIVITY_EPS).astype(np.float32)


def _guided_sampling_step(
    current_hand_pose: np.ndarray,
    grad_np: np.ndarray,
    *,
    step_scales: np.ndarray,
    noise_scale: float,
    rng: np.random.Generator,
    config: RefineConfig,
) -> np.ndarray:
    current_hand_pose = np.asarray(current_hand_pose, dtype=np.float32)
    grad_np = np.nan_to_num(np.asarray(grad_np, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    step_scales = np.asarray(step_scales, dtype=np.float32)

    scaled_grad = grad_np * step_scales * float(config.grad_scale)
    grad_norm = float(np.linalg.norm(scaled_grad))
    if float(config.grad_clip_norm) > 0.0 and grad_norm > float(config.grad_clip_norm):
        scaled_grad *= float(config.grad_clip_norm) / max(grad_norm, CONTACT_LOCAL_EPS)
        grad_norm = float(np.linalg.norm(scaled_grad))

    radius = float(np.sqrt(float(current_hand_pose.size)) * noise_scale)
    if radius <= CONTACT_LOCAL_EPS:
        return current_hand_pose.copy()

    d_sample_norm = float(noise_scale) * rng.standard_normal(current_hand_pose.shape, dtype=np.float32)
    if grad_norm > CONTACT_LOCAL_EPS:
        d_star_norm = -radius * scaled_grad / grad_norm
    else:
        d_star_norm = np.zeros_like(current_hand_pose, dtype=np.float32)
    mix_direction_norm = d_sample_norm + float(config.guidance_scale) * (d_star_norm - d_sample_norm)
    mix_norm = float(np.linalg.norm(mix_direction_norm))
    if mix_norm <= CONTACT_LOCAL_EPS:
        return current_hand_pose.copy()

    mix_step_actual = step_scales * (radius * mix_direction_norm / mix_norm)
    return np.asarray(current_hand_pose + mix_step_actual, dtype=np.float32)


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
    step_scales = np.asarray(callbacks.sample_scales(), dtype=np.float32)
    rng = np.random.default_rng(int(config.seed))

    initial_energy, projected_initial_hand_pose, initial_grad = callbacks.evaluate_terms_with_grad(
        initial_hand_pose,
        contact_indices,
        contact_target_local,
        config,
    )
    best_hand_pose = projected_initial_hand_pose.copy()
    best_energy = initial_energy
    current_hand_pose = projected_initial_hand_pose.copy()
    current_grad = np.asarray(initial_grad, dtype=np.float32)

    history_total: list[float] = [initial_energy.total]
    history_distance: list[float] = [initial_energy.distance]
    history_equilibrium: list[float] = [initial_energy.equilibrium]
    history_penetration: list[float] = [initial_energy.penetration]
    history_contact: list[float] = [initial_energy.contact]
    history_root_reg: list[float] = [initial_energy.root_reg]
    history_joint_reg: list[float] = [initial_energy.joint_reg]
    history_active_joint_count: list[int] = [int(np.count_nonzero(_active_mask_from_grad(current_grad)[9:]))]

    final_mask = _active_mask_from_grad(current_grad)
    for step_index in range(int(config.steps)):
        current_energy, projected, grad_np = callbacks.evaluate_terms_with_grad(
            current_hand_pose,
            contact_indices,
            contact_target_local,
            config,
        )
        noise_scale = _scheduled_noise_scale(config, step_index)
        proposal = _guided_sampling_step(
            projected,
            grad_np,
            step_scales=step_scales,
            noise_scale=noise_scale,
            rng=rng,
            config=config,
        )
        next_terms, next_projected, next_grad = callbacks.evaluate_terms_with_grad(
            np.asarray(proposal, dtype=np.float32),
            contact_indices,
            contact_target_local,
            config,
        )
        current_hand_pose = np.asarray(next_projected, dtype=np.float32)
        current_grad = np.asarray(next_grad, dtype=np.float32)
        final_mask = _active_mask_from_grad(current_grad)

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
            "method": "dexgrasp_anything_guided_sampling",
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
