from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .types import RefineEnergyTerms, RefineResultState, RefineRunArtifact


def _energy_to_arrays(prefix: str, energy: RefineEnergyTerms) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_total": np.asarray(energy.total, dtype=np.float32),
        f"{prefix}_distance": np.asarray(energy.distance, dtype=np.float32),
        f"{prefix}_equilibrium": np.asarray(energy.equilibrium, dtype=np.float32),
        f"{prefix}_penetration": np.asarray(energy.penetration, dtype=np.float32),
        f"{prefix}_contact": np.asarray(energy.contact, dtype=np.float32),
        f"{prefix}_root_reg": np.asarray(energy.root_reg, dtype=np.float32),
        f"{prefix}_joint_reg": np.asarray(energy.joint_reg, dtype=np.float32),
    }


def _arrays_to_energy(prefix: str, payload: dict[str, np.ndarray]) -> RefineEnergyTerms:
    return RefineEnergyTerms(
        total=float(payload[f"{prefix}_total"]),
        distance=float(payload[f"{prefix}_distance"]),
        equilibrium=float(payload[f"{prefix}_equilibrium"]),
        penetration=float(payload[f"{prefix}_penetration"]),
        contact=float(payload[f"{prefix}_contact"]),
        root_reg=float(payload[f"{prefix}_root_reg"]),
        joint_reg=float(payload[f"{prefix}_joint_reg"]),
    )


def save_refine_run(path: str | Path, *, metadata: dict[str, Any], state: RefineResultState) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "initial_hand_pose": np.asarray(state.initial_hand_pose, dtype=np.float32),
        "refined_hand_pose": np.asarray(state.refined_hand_pose, dtype=np.float32),
        "best_hand_pose": np.asarray(state.best_hand_pose, dtype=np.float32),
        "contact_indices": np.asarray(state.contact_indices, dtype=np.int32),
        "contact_target_local": np.asarray(state.contact_target_local, dtype=np.float32),
        "final_active_mask": np.asarray(state.final_active_mask, dtype=np.float32),
        "history_total": np.asarray(state.history_total, dtype=np.float32),
        "history_distance": np.asarray(state.history_distance, dtype=np.float32),
        "history_equilibrium": np.asarray(state.history_equilibrium, dtype=np.float32),
        "history_penetration": np.asarray(state.history_penetration, dtype=np.float32),
        "history_contact": np.asarray(state.history_contact, dtype=np.float32),
        "history_root_reg": np.asarray(state.history_root_reg, dtype=np.float32),
        "history_joint_reg": np.asarray(state.history_joint_reg, dtype=np.float32),
        "history_active_joint_count": np.asarray(state.history_active_joint_count, dtype=np.int32),
        "initial_actual_contact_count": np.asarray(state.initial_actual_contact_count, dtype=np.int32),
        "initial_actual_penetration_count": np.asarray(state.initial_actual_penetration_count, dtype=np.int32),
        "initial_actual_depth_sum": np.asarray(state.initial_actual_depth_sum, dtype=np.float32),
        "initial_actual_max_depth": np.asarray(state.initial_actual_max_depth, dtype=np.float32),
        "final_actual_contact_count": np.asarray(state.final_actual_contact_count, dtype=np.int32),
        "final_actual_penetration_count": np.asarray(state.final_actual_penetration_count, dtype=np.int32),
        "final_actual_depth_sum": np.asarray(state.final_actual_depth_sum, dtype=np.float32),
        "final_actual_max_depth": np.asarray(state.final_actual_max_depth, dtype=np.float32),
        "best_actual_contact_count": np.asarray(state.best_actual_contact_count, dtype=np.int32),
        "best_actual_penetration_count": np.asarray(state.best_actual_penetration_count, dtype=np.int32),
        "best_actual_depth_sum": np.asarray(state.best_actual_depth_sum, dtype=np.float32),
        "best_actual_max_depth": np.asarray(state.best_actual_max_depth, dtype=np.float32),
        **_energy_to_arrays("initial_energy", state.initial_energy),
        **_energy_to_arrays("final_energy", state.final_energy),
        **_energy_to_arrays("best_energy", state.best_energy),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_),
    }
    np.savez_compressed(output_path, **payload)
    return output_path


def load_refine_run(path: str | Path) -> RefineRunArtifact:
    input_path = Path(path).expanduser().resolve()
    with np.load(input_path, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    metadata = json.loads(str(payload.pop("metadata_json").item()))
    state = RefineResultState(
        initial_hand_pose=np.asarray(payload["initial_hand_pose"], dtype=np.float32),
        refined_hand_pose=np.asarray(payload["refined_hand_pose"], dtype=np.float32),
        best_hand_pose=np.asarray(payload["best_hand_pose"], dtype=np.float32),
        contact_indices=np.asarray(payload["contact_indices"], dtype=np.int32),
        contact_target_local=np.asarray(payload["contact_target_local"], dtype=np.float32),
        final_active_mask=np.asarray(payload["final_active_mask"], dtype=np.float32),
        history_total=np.asarray(payload["history_total"], dtype=np.float32),
        history_distance=np.asarray(payload["history_distance"], dtype=np.float32),
        history_equilibrium=np.asarray(payload["history_equilibrium"], dtype=np.float32),
        history_penetration=np.asarray(payload["history_penetration"], dtype=np.float32),
        history_contact=np.asarray(payload["history_contact"], dtype=np.float32),
        history_root_reg=np.asarray(payload["history_root_reg"], dtype=np.float32),
        history_joint_reg=np.asarray(payload["history_joint_reg"], dtype=np.float32),
        history_active_joint_count=np.asarray(payload["history_active_joint_count"], dtype=np.int32),
        initial_energy=_arrays_to_energy("initial_energy", payload),
        final_energy=_arrays_to_energy("final_energy", payload),
        best_energy=_arrays_to_energy("best_energy", payload),
        initial_actual_contact_count=int(payload["initial_actual_contact_count"]),
        initial_actual_penetration_count=int(payload["initial_actual_penetration_count"]),
        initial_actual_depth_sum=float(payload["initial_actual_depth_sum"]),
        initial_actual_max_depth=float(payload["initial_actual_max_depth"]),
        final_actual_contact_count=int(payload["final_actual_contact_count"]),
        final_actual_penetration_count=int(payload["final_actual_penetration_count"]),
        final_actual_depth_sum=float(payload["final_actual_depth_sum"]),
        final_actual_max_depth=float(payload["final_actual_max_depth"]),
        best_actual_contact_count=int(payload["best_actual_contact_count"]),
        best_actual_penetration_count=int(payload["best_actual_penetration_count"]),
        best_actual_depth_sum=float(payload["best_actual_depth_sum"]),
        best_actual_max_depth=float(payload["best_actual_max_depth"]),
    )
    return RefineRunArtifact(path=input_path, metadata=metadata, state=state)
