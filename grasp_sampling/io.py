from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .types import SamplingEvalState


@dataclass(frozen=True)
class SamplingRunArtifact:
    path: Path
    metadata: dict[str, Any]
    state: SamplingEvalState


def _state_arrays(state: SamplingEvalState) -> dict[str, np.ndarray]:
    return {
        "base_hand_pose": np.asarray(state.base_hand_pose, dtype=np.float32),
        "base_contact_indices": np.asarray(state.base_contact_indices, dtype=np.int32),
        "object_relative_pos": np.asarray(state.object_relative_pos, dtype=np.float32),
        "object_relative_quat": np.asarray(state.object_relative_quat, dtype=np.float32),
        "squeeze_deltas": np.asarray(state.squeeze_deltas, dtype=np.float32),
        "qpos_targets": np.asarray(state.qpos_targets, dtype=np.float32),
        "initial_contact_count": np.asarray(state.initial_contact_count, dtype=np.int32),
        "initial_penetration_count": np.asarray(state.initial_penetration_count, dtype=np.int32),
        "initial_depth_sum": np.asarray(state.initial_depth_sum, dtype=np.float32),
        "initial_max_depth": np.asarray(state.initial_max_depth, dtype=np.float32),
        "initial_overlap": np.asarray(state.initial_overlap, dtype=np.bool_),
        "motion_scores": np.asarray(state.motion_scores, dtype=np.float32),
        "motion_max_translation": np.asarray(state.motion_max_translation, dtype=np.float32),
        "motion_max_rotation_rad": np.asarray(state.motion_max_rotation_rad, dtype=np.float32),
        "motion_final_translation": np.asarray(state.motion_final_translation, dtype=np.float32),
        "motion_final_rotation_rad": np.asarray(state.motion_final_rotation_rad, dtype=np.float32),
        "motion_contact_min": np.asarray(state.motion_contact_min, dtype=np.int32),
        "motion_contact_final": np.asarray(state.motion_contact_final, dtype=np.int32),
        "motion_lost": np.asarray(state.motion_lost, dtype=np.bool_),
        "motion_fail": np.asarray(state.motion_fail, dtype=np.bool_),
        "motion_early_stop": np.asarray(state.motion_early_stop, dtype=np.bool_),
        "motion_steps": np.asarray(state.motion_steps, dtype=np.int32),
        "overall_scores": np.asarray(state.overall_scores, dtype=np.float32),
        "success": np.asarray(state.success, dtype=np.bool_),
        "chosen_attempt_index": np.asarray(int(state.chosen_attempt_index), dtype=np.int32),
    }


def _arrays_to_state(data: dict[str, np.ndarray]) -> SamplingEvalState:
    return SamplingEvalState(
        base_hand_pose=np.asarray(data["base_hand_pose"], dtype=np.float32),
        base_contact_indices=np.asarray(data["base_contact_indices"], dtype=np.int32),
        object_relative_pos=np.asarray(data["object_relative_pos"], dtype=np.float32),
        object_relative_quat=np.asarray(data["object_relative_quat"], dtype=np.float32),
        squeeze_deltas=np.asarray(data["squeeze_deltas"], dtype=np.float32),
        qpos_targets=np.asarray(data["qpos_targets"], dtype=np.float32),
        initial_contact_count=np.asarray(data["initial_contact_count"], dtype=np.int32),
        initial_penetration_count=np.asarray(data["initial_penetration_count"], dtype=np.int32),
        initial_depth_sum=np.asarray(data["initial_depth_sum"], dtype=np.float32),
        initial_max_depth=np.asarray(data["initial_max_depth"], dtype=np.float32),
        initial_overlap=np.asarray(data["initial_overlap"], dtype=np.bool_),
        motion_scores=np.asarray(data["motion_scores"], dtype=np.float32),
        motion_max_translation=np.asarray(data["motion_max_translation"], dtype=np.float32),
        motion_max_rotation_rad=np.asarray(data["motion_max_rotation_rad"], dtype=np.float32),
        motion_final_translation=np.asarray(data["motion_final_translation"], dtype=np.float32),
        motion_final_rotation_rad=np.asarray(data["motion_final_rotation_rad"], dtype=np.float32),
        motion_contact_min=np.asarray(data["motion_contact_min"], dtype=np.int32),
        motion_contact_final=np.asarray(data["motion_contact_final"], dtype=np.int32),
        motion_lost=np.asarray(data["motion_lost"], dtype=np.bool_),
        motion_fail=np.asarray(data["motion_fail"], dtype=np.bool_),
        motion_early_stop=np.asarray(data["motion_early_stop"], dtype=np.bool_),
        motion_steps=np.asarray(data["motion_steps"], dtype=np.int32),
        overall_scores=np.asarray(data["overall_scores"], dtype=np.float32),
        success=np.asarray(data["success"], dtype=np.bool_),
        chosen_attempt_index=int(np.asarray(data["chosen_attempt_index"]).item()),
    )


def save_sampling_run(path: str | Path, *, metadata: dict[str, Any], state: SamplingEvalState) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = _state_arrays(state)
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_)
    np.savez_compressed(output_path, **payload)
    return output_path


def load_sampling_run(path: str | Path) -> SamplingRunArtifact:
    input_path = Path(path).expanduser().resolve()
    with np.load(input_path, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}

    metadata = json.loads(str(payload.pop("metadata_json").item()))
    state = _arrays_to_state(payload)
    return SamplingRunArtifact(path=input_path, metadata=metadata, state=state)
