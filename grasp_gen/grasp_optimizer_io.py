from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .grasp_optimizer_state import GraspBatchEnergy, GraspBatchState


@dataclass(frozen=True)
class GraspRunArtifact:
    path: Path
    metadata: dict[str, Any]
    state: GraspBatchState


def _energy_arrays(prefix: str, energy: GraspBatchEnergy) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_total": np.asarray(energy.total, dtype=np.float32),
        f"{prefix}_distance": np.asarray(energy.distance, dtype=np.float32),
    }


def _arrays_to_energy(data: dict[str, np.ndarray], prefix: str) -> GraspBatchEnergy:
    distance_key = f"{prefix}_distance"
    if distance_key in data:
        distance = jnp.asarray(data[distance_key], dtype=jnp.float32)
    else:
        distance = jnp.asarray(data[f"{prefix}_total"], dtype=jnp.float32)
    return GraspBatchEnergy(
        total=jnp.asarray(data[f"{prefix}_total"], dtype=jnp.float32),
        distance=distance,
    )


def _state_arrays(state: GraspBatchState) -> dict[str, np.ndarray]:
    return {
        "hand_pose": np.asarray(state.hand_pose, dtype=np.float32),
        "contact_indices": np.asarray(state.contact_indices, dtype=np.int32),
        "best_hand_pose": np.asarray(state.best_hand_pose, dtype=np.float32),
        "best_contact_indices": np.asarray(state.best_contact_indices, dtype=np.int32),
        "ema_grad": np.asarray(state.ema_grad, dtype=np.float32),
        "accepted_steps": np.asarray(state.accepted_steps, dtype=np.int32),
        "rejected_steps": np.asarray(state.rejected_steps, dtype=np.int32),
        "step_index": np.asarray(state.step_index, dtype=np.int32),
        "rng_key_data": np.asarray(jax.random.key_data(state.rng_key), dtype=np.uint32),
        **_energy_arrays("energy", state.energy),
        **_energy_arrays("best_energy", state.best_energy),
    }


def _arrays_to_state(data: dict[str, np.ndarray]) -> GraspBatchState:
    return GraspBatchState(
        hand_pose=jnp.asarray(data["hand_pose"], dtype=jnp.float32),
        contact_indices=jnp.asarray(data["contact_indices"], dtype=jnp.int32),
        energy=_arrays_to_energy(data, "energy"),
        best_hand_pose=jnp.asarray(data["best_hand_pose"], dtype=jnp.float32),
        best_contact_indices=jnp.asarray(data["best_contact_indices"], dtype=jnp.int32),
        best_energy=_arrays_to_energy(data, "best_energy"),
        ema_grad=jnp.asarray(data["ema_grad"], dtype=jnp.float32),
        accepted_steps=jnp.asarray(data["accepted_steps"], dtype=jnp.int32),
        rejected_steps=jnp.asarray(data["rejected_steps"], dtype=jnp.int32),
        step_index=jnp.asarray(data["step_index"], dtype=jnp.int32),
        rng_key=jax.random.wrap_key_data(jnp.asarray(data["rng_key_data"], dtype=jnp.uint32)),
    )


def save_grasp_run(
    path: str | Path,
    *,
    metadata: dict[str, Any],
    state: GraspBatchState,
) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _state_arrays(state)
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_)
    np.savez_compressed(output_path, **payload)
    return output_path


def load_grasp_run(path: str | Path) -> GraspRunArtifact:
    input_path = Path(path).expanduser().resolve()
    with np.load(input_path, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}

    metadata = json.loads(str(payload.pop("metadata_json").item()))
    state = _arrays_to_state(payload)
    return GraspRunArtifact(path=input_path, metadata=metadata, state=state)
