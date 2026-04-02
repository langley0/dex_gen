from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ArtifactSubsetResult:
    input_path: Path
    output_path: Path
    state_name: str
    requested_top_k: int
    selected_count: int
    selected_indices: np.ndarray
    selected_energies: np.ndarray


def _energy_key_for_state(state_name: str) -> str:
    if state_name == "best":
        return "best_energy_total"
    if state_name == "current":
        return "energy_total"
    raise ValueError(f"Unsupported state_name {state_name!r}")


def _result_stats(prefix: str, payload: dict[str, np.ndarray]) -> dict[str, float | int]:
    total = np.asarray(payload[f"{prefix}_total"], dtype=np.float32)
    distance = np.asarray(payload.get(f"{prefix}_distance", total), dtype=np.float32)
    penetration = np.asarray(payload.get(f"{prefix}_penetration", np.zeros_like(total)), dtype=np.float32)
    equilibrium = np.asarray(payload.get(f"{prefix}_equilibrium", np.zeros_like(total)), dtype=np.float32)
    force = np.asarray(payload.get(f"{prefix}_force", np.zeros_like(total)), dtype=np.float32)
    torque = np.asarray(payload.get(f"{prefix}_torque", np.zeros_like(total)), dtype=np.float32)
    return {
        f"{prefix}_min": float(np.min(total)),
        f"{prefix}_mean": float(np.mean(total)),
        f"{prefix}_distance_min": float(np.min(distance)),
        f"{prefix}_distance_mean": float(np.mean(distance)),
        f"{prefix}_penetration_min": float(np.min(penetration)),
        f"{prefix}_penetration_mean": float(np.mean(penetration)),
        f"{prefix}_equilibrium_min": float(np.min(equilibrium)),
        f"{prefix}_equilibrium_mean": float(np.mean(equilibrium)),
        f"{prefix}_force_min": float(np.min(force)),
        f"{prefix}_force_mean": float(np.mean(force)),
        f"{prefix}_torque_min": float(np.min(torque)),
        f"{prefix}_torque_mean": float(np.mean(torque)),
        f"{prefix}_best_sample_index": int(np.argmin(total)),
    }


def subset_grasp_artifact_topk(
    input_path: str | Path,
    output_path: str | Path,
    *,
    top_k: int,
    state_name: str = "best",
) -> ArtifactSubsetResult:
    src = Path(input_path).expanduser().resolve()
    dst = Path(output_path).expanduser().resolve()
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    with np.load(src, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}

    metadata = json.loads(str(payload["metadata_json"].item()))
    energy_key = _energy_key_for_state(state_name)
    energies = np.asarray(payload[energy_key], dtype=np.float32)
    if energies.ndim != 1:
        raise ValueError(f"Expected a 1D energy array at {energy_key}, got shape {energies.shape}.")

    batch_size = int(energies.shape[0])
    selected_count = min(int(top_k), batch_size)
    selected_indices = np.argsort(energies, kind="stable")[:selected_count].astype(np.int32)

    subset_payload: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if key == "metadata_json":
            continue
        array = np.asarray(value)
        if array.ndim >= 1 and array.shape[0] == batch_size:
            subset_payload[key] = np.asarray(array[selected_indices], dtype=array.dtype)
        else:
            subset_payload[key] = array

    metadata = dict(metadata)
    run_meta = dict(metadata.get("run", {}))
    run_meta["batch"] = int(selected_count)
    metadata["run"] = run_meta

    result_meta = dict(metadata.get("result", {}))
    result_meta["best_sample_index"] = 0
    result_meta["selection_source_count"] = batch_size
    result_meta["selection_selected_count"] = int(selected_count)
    result_meta["selection_energy_key"] = energy_key
    result_meta["selection_requested_top_k"] = int(top_k)
    result_meta["selection_indices"] = selected_indices.tolist()
    result_meta["selection_energy_min"] = float(np.min(energies[selected_indices]))
    result_meta["selection_energy_mean"] = float(np.mean(energies[selected_indices]))
    result_meta.update(_result_stats("best_energy", subset_payload))
    result_meta.update(_result_stats("energy", subset_payload))
    metadata["result"] = result_meta
    metadata["selection"] = {
        "source_artifact": str(src),
        "state_name": state_name,
        "top_k": int(top_k),
        "selected_indices": selected_indices.tolist(),
    }

    subset_payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **subset_payload)

    return ArtifactSubsetResult(
        input_path=src,
        output_path=dst,
        state_name=state_name,
        requested_top_k=int(top_k),
        selected_count=int(selected_count),
        selected_indices=selected_indices,
        selected_energies=np.asarray(energies[selected_indices], dtype=np.float32),
    )
