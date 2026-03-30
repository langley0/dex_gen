from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .refine import CONTACT_LOCAL_EPS, RefineConfig


@dataclass(frozen=True)
class BatchRefineCallbacks:
    evaluate_terms_with_grad: Callable[
        [np.ndarray, np.ndarray, np.ndarray, RefineConfig],
        tuple[dict[str, np.ndarray], np.ndarray, np.ndarray],
    ]
    active_pose_masks: Callable[[np.ndarray, np.ndarray, float], tuple[np.ndarray, np.ndarray]]
    actual_overlap_batch: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


def refine_result_batch(
    *,
    metadata: dict[str, Any],
    initial_hand_pose: np.ndarray,
    contact_indices: np.ndarray,
    source_total: np.ndarray,
    contact_target_local: np.ndarray,
    callbacks: BatchRefineCallbacks,
    config: RefineConfig | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    config = RefineConfig() if config is None else config
    initial_hand_pose = np.asarray(initial_hand_pose, dtype=np.float32)
    contact_indices = np.asarray(contact_indices, dtype=np.int32)
    source_total = np.asarray(source_total, dtype=np.float32)
    contact_target_local = np.asarray(contact_target_local, dtype=np.float32)

    initial_terms, projected_initial, _ = callbacks.evaluate_terms_with_grad(
        initial_hand_pose,
        contact_indices,
        contact_target_local,
        config,
    )
    current_hand_pose = projected_initial.copy()
    best_hand_pose = projected_initial.copy()
    best_terms = {name: np.asarray(values, dtype=np.float32).copy() for name, values in initial_terms.items()}

    adam_m = np.zeros_like(current_hand_pose, dtype=np.float32)
    adam_v = np.zeros_like(current_hand_pose, dtype=np.float32)

    history_mean_total = [float(np.mean(initial_terms["total"]))]
    history_mean_penetration = [float(np.mean(initial_terms["penetration"]))]
    history_mean_contact = [float(np.mean(initial_terms["contact"]))]

    final_masks = np.ones_like(current_hand_pose, dtype=np.float32)
    final_active_joint_count = np.zeros((current_hand_pose.shape[0],), dtype=np.int32)

    for step_index in range(int(config.steps)):
        final_masks, final_active_joint_count = callbacks.active_pose_masks(
            current_hand_pose,
            contact_indices,
            float(config.penetration_threshold),
        )
        current_terms, projected, grad_np = callbacks.evaluate_terms_with_grad(
            current_hand_pose,
            contact_indices,
            contact_target_local,
            config,
        )
        grad_np = np.nan_to_num(np.asarray(grad_np, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0) * final_masks

        t = step_index + 1
        adam_m = float(config.beta1) * adam_m + (1.0 - float(config.beta1)) * grad_np
        adam_v = float(config.beta2) * adam_v + (1.0 - float(config.beta2)) * np.square(grad_np)
        m_hat = adam_m / max(1.0 - float(config.beta1) ** t, CONTACT_LOCAL_EPS)
        v_hat = adam_v / max(1.0 - float(config.beta2) ** t, CONTACT_LOCAL_EPS)
        proposal = np.asarray(projected, dtype=np.float32) - float(config.step_size) * m_hat / (
            np.sqrt(v_hat) + float(config.adam_eps)
        )
        proposal = proposal * final_masks + np.asarray(projected, dtype=np.float32) * (1.0 - final_masks)

        next_terms, next_projected, _ = callbacks.evaluate_terms_with_grad(
            np.asarray(proposal, dtype=np.float32),
            contact_indices,
            contact_target_local,
            config,
        )
        current_hand_pose = np.asarray(next_projected, dtype=np.float32)

        is_new_best = np.asarray(next_terms["total"], dtype=np.float32) < np.asarray(best_terms["total"], dtype=np.float32)
        best_hand_pose = np.where(is_new_best[:, None], current_hand_pose, best_hand_pose)
        for name in best_terms:
            best_terms[name] = np.where(is_new_best, np.asarray(next_terms[name], dtype=np.float32), best_terms[name])

        history_mean_total.append(float(np.mean(next_terms["total"])))
        history_mean_penetration.append(float(np.mean(next_terms["penetration"])))
        history_mean_contact.append(float(np.mean(next_terms["contact"])))

    final_terms, refined_hand_pose, _ = callbacks.evaluate_terms_with_grad(
        current_hand_pose,
        contact_indices,
        contact_target_local,
        config,
    )
    improved_mask = np.asarray(best_terms["total"], dtype=np.float32) < np.asarray(initial_terms["total"], dtype=np.float32)
    fixed_mask = (np.asarray(initial_terms["penetration"], dtype=np.float32) > float(config.penetration_threshold)) & (
        np.asarray(best_terms["penetration"], dtype=np.float32) <= float(config.penetration_threshold)
    )
    initial_actual_contact, initial_actual_penetration_count, initial_actual_depth_sum, initial_actual_max_depth = callbacks.actual_overlap_batch(
        projected_initial
    )
    best_actual_contact, best_actual_penetration_count, best_actual_depth_sum, best_actual_max_depth = callbacks.actual_overlap_batch(
        best_hand_pose
    )
    actual_fixed_mask = (np.asarray(initial_actual_penetration_count) > 0) & (np.asarray(best_actual_penetration_count) == 0)

    result_metadata = dict(metadata)
    result_metadata["refine"] = asdict(config)
    result_metadata["result"] = {
        "batch_size": int(initial_hand_pose.shape[0]),
        "improved_count": int(np.count_nonzero(improved_mask)),
        "fixed_count": int(np.count_nonzero(fixed_mask)),
        "actual_fixed_count": int(np.count_nonzero(actual_fixed_mask)),
        "best_fixed_sample": int(np.argmin(np.where(fixed_mask, best_terms["total"], np.inf))) if np.any(fixed_mask) else -1,
        "best_actual_fixed_sample": (
            int(np.argmin(np.where(actual_fixed_mask, best_terms["total"], np.inf))) if np.any(actual_fixed_mask) else -1
        ),
    }

    state = {
        "sample_indices": np.arange(initial_hand_pose.shape[0], dtype=np.int32),
        "source_total": source_total,
        "initial_hand_pose": np.asarray(projected_initial, dtype=np.float32),
        "refined_hand_pose": np.asarray(refined_hand_pose, dtype=np.float32),
        "best_hand_pose": np.asarray(best_hand_pose, dtype=np.float32),
        "contact_indices": contact_indices,
        "contact_target_local": contact_target_local,
        "final_active_mask": np.asarray(final_masks, dtype=np.float32),
        "final_active_joint_count": np.asarray(final_active_joint_count, dtype=np.int32),
        "initial_total": np.asarray(initial_terms["total"], dtype=np.float32),
        "initial_distance": np.asarray(initial_terms["distance"], dtype=np.float32),
        "initial_equilibrium": np.asarray(initial_terms["equilibrium"], dtype=np.float32),
        "initial_penetration": np.asarray(initial_terms["penetration"], dtype=np.float32),
        "initial_contact": np.asarray(initial_terms["contact"], dtype=np.float32),
        "initial_root_reg": np.asarray(initial_terms["root_reg"], dtype=np.float32),
        "initial_joint_reg": np.asarray(initial_terms["joint_reg"], dtype=np.float32),
        "final_total": np.asarray(final_terms["total"], dtype=np.float32),
        "final_distance": np.asarray(final_terms["distance"], dtype=np.float32),
        "final_equilibrium": np.asarray(final_terms["equilibrium"], dtype=np.float32),
        "final_penetration": np.asarray(final_terms["penetration"], dtype=np.float32),
        "final_contact": np.asarray(final_terms["contact"], dtype=np.float32),
        "final_root_reg": np.asarray(final_terms["root_reg"], dtype=np.float32),
        "final_joint_reg": np.asarray(final_terms["joint_reg"], dtype=np.float32),
        "best_total": np.asarray(best_terms["total"], dtype=np.float32),
        "best_distance": np.asarray(best_terms["distance"], dtype=np.float32),
        "best_equilibrium": np.asarray(best_terms["equilibrium"], dtype=np.float32),
        "best_penetration": np.asarray(best_terms["penetration"], dtype=np.float32),
        "best_contact": np.asarray(best_terms["contact"], dtype=np.float32),
        "best_root_reg": np.asarray(best_terms["root_reg"], dtype=np.float32),
        "best_joint_reg": np.asarray(best_terms["joint_reg"], dtype=np.float32),
        "improved_mask": np.asarray(improved_mask, dtype=np.bool_),
        "fixed_mask": np.asarray(fixed_mask, dtype=np.bool_),
        "initial_actual_contact_count": np.asarray(initial_actual_contact, dtype=np.int32),
        "initial_actual_penetration_count": np.asarray(initial_actual_penetration_count, dtype=np.int32),
        "initial_actual_depth_sum": np.asarray(initial_actual_depth_sum, dtype=np.float32),
        "initial_actual_max_depth": np.asarray(initial_actual_max_depth, dtype=np.float32),
        "best_actual_contact_count": np.asarray(best_actual_contact, dtype=np.int32),
        "best_actual_penetration_count": np.asarray(best_actual_penetration_count, dtype=np.int32),
        "best_actual_depth_sum": np.asarray(best_actual_depth_sum, dtype=np.float32),
        "best_actual_max_depth": np.asarray(best_actual_max_depth, dtype=np.float32),
        "actual_fixed_mask": np.asarray(actual_fixed_mask, dtype=np.bool_),
        "history_mean_total": np.asarray(history_mean_total, dtype=np.float32),
        "history_mean_penetration": np.asarray(history_mean_penetration, dtype=np.float32),
        "history_mean_contact": np.asarray(history_mean_contact, dtype=np.float32),
    }
    return result_metadata, state


def save_refine_batch(path: str | Path, *, metadata: dict[str, Any], state: dict[str, np.ndarray]) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: np.asarray(value) for name, value in state.items()}
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_)
    np.savez_compressed(output_path, **payload)
    return output_path


def load_refine_batch(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    input_path = Path(path).expanduser().resolve()
    with np.load(input_path, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    metadata = json.loads(str(payload.pop("metadata_json").item()))
    return metadata, payload
