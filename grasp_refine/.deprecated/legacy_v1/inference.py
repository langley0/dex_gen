from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import numpy as np

from .checkpoint import load_training_checkpoint
from .config_io import training_config_from_dict
from .diffusion import make_diffusion_schedule
from .inspire_hand import InspireHandSpec, load_inspire_hand_spec
from .io import GraspArtifactPayload, GraspRecord, load_grasp_artifact
from .object_mesh import load_object_mesh, sample_mesh_points
from .sampling import sample_grasp_poses


@dataclass(frozen=True)
class ObjectConditionedSample:
    object_key: str
    artifact_path: Path
    object_name: str
    object_kind: str
    object_metadata: dict[str, object]
    object_points: np.ndarray
    object_normals: np.ndarray
    generated_poses: np.ndarray
    generated_joint_limit_loss: np.ndarray
    generated_root_distance_loss: np.ndarray


def _stable_kind_seed(object_kind: str, base_seed: int) -> int:
    stable_hash = sum((index + 1) * ord(char) for index, char in enumerate(object_kind))
    return int(base_seed) + stable_hash % 100_000


def _object_cloud_from_record(record: GraspRecord, *, num_points: int, object_point_seed: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = load_object_mesh(record.object_metadata)
    seed = _stable_kind_seed(record.object_kind, base_seed=object_point_seed)
    return sample_mesh_points(mesh, int(num_points), seed=seed)


def _scalar_losses(
    poses: np.ndarray,
    *,
    hand_spec: InspireHandSpec,
    object_points: np.ndarray,
    root_distance_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    pose_array = np.asarray(poses, dtype=np.float32)
    if pose_array.ndim == 1:
        pose_array = pose_array[None, :]
    joints = pose_array[:, 9:]
    below = np.maximum(hand_spec.joint_lower[None, :] - joints, 0.0)
    above = np.maximum(joints - hand_spec.joint_upper[None, :], 0.0)
    joint = np.mean(below + above, axis=1)

    root = pose_array[:, :3]
    diff = root[:, None, :] - object_points[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    nearest = np.min(distances, axis=1)
    root_loss = np.maximum(nearest - float(root_distance_threshold), 0.0)
    return joint.astype(np.float32), root_loss.astype(np.float32)


def _unique_object_key(record: GraspRecord, artifact_path: Path, seen: dict[str, int]) -> str:
    base = record.object_name or record.object_kind or artifact_path.stem
    count = seen.get(base, 0)
    seen[base] = count + 1
    if count == 0:
        return str(base)
    return f"{base}__{count + 1:02d}"


def _select_record(payload: GraspArtifactPayload) -> GraspRecord:
    if not payload.records:
        raise ValueError(f"No grasp records were loaded from {payload.path}.")
    return payload.records[0]


def sample_checkpoint_artifacts(
    checkpoint_path: str | Path,
    artifact_paths: tuple[Path, ...] | list[Path],
    *,
    state_name: str = "best",
    seed: int = 0,
    num_generated: int = 32,
    object_num_points: int | None = None,
    device: str | None = None,
) -> list[ObjectConditionedSample]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    payload = load_training_checkpoint(checkpoint_path)
    train_config = training_config_from_dict(payload["config"])
    if device is not None:
        train_config = replace(train_config, device=str(device))

    hand_spec = load_inspire_hand_spec(train_config.hand_side)
    model_config = train_config.model
    if model_config.pose_dim != hand_spec.pose_dim:
        model_config = replace(model_config, pose_dim=hand_spec.pose_dim)

    schedule = make_diffusion_schedule(train_config.diffusion)
    params = payload["params"]
    if device is not None:
        try:
            jax_device = jax.devices(device)[0]
        except Exception:
            jax_device = jax.devices()[0]
        params = jax.device_put(params, jax_device)
        schedule = jax.device_put(schedule, jax_device)
    else:
        jax_device = None

    results: list[ObjectConditionedSample] = []
    seen_keys: dict[str, int] = {}
    artifact_list = [Path(path).expanduser().resolve() for path in artifact_paths]
    for artifact_index, artifact_path in enumerate(artifact_list):
        artifact_payload = load_grasp_artifact(artifact_path, state_name=state_name)
        record = _select_record(artifact_payload)
        num_points = train_config.dataset.object_num_points if object_num_points is None else int(object_num_points)
        object_points, object_normals = _object_cloud_from_record(
            record,
            num_points=num_points,
            object_point_seed=train_config.dataset.object_point_seed,
        )
        object_points_jax = jax.numpy.asarray(object_points)
        object_normals_jax = jax.numpy.asarray(object_normals)
        if jax_device is not None:
            object_points_jax = jax.device_put(object_points_jax, jax_device)
            object_normals_jax = jax.device_put(object_normals_jax, jax_device)

        rng_seed = np.uint32((int(seed) + 7919 * artifact_index) % (2**32))
        sample_batch = sample_grasp_poses(
            params,
            object_points=object_points_jax,
            object_normals=object_normals_jax,
            rng_key=jax.random.key(rng_seed),
            model_config=model_config,
            diffusion_config=train_config.diffusion,
            normalizer=payload["normalizer"],
            schedule=schedule,
            num_samples=int(num_generated),
        )
        generated_poses = np.asarray(sample_batch.pose, dtype=np.float32)
        generated_joint, generated_root = _scalar_losses(
            generated_poses,
            hand_spec=hand_spec,
            object_points=object_points,
            root_distance_threshold=train_config.loss.root_distance_threshold,
        )
        results.append(
            ObjectConditionedSample(
                object_key=_unique_object_key(record, artifact_path, seen_keys),
                artifact_path=artifact_path,
                object_name=record.object_name,
                object_kind=record.object_kind,
                object_metadata=dict(record.object_metadata),
                object_points=np.asarray(object_points, dtype=np.float32),
                object_normals=np.asarray(object_normals, dtype=np.float32),
                generated_poses=generated_poses,
                generated_joint_limit_loss=np.asarray(generated_joint, dtype=np.float32),
                generated_root_distance_loss=np.asarray(generated_root, dtype=np.float32),
            )
        )
    return results


def _summary_for_sample(sample: ObjectConditionedSample) -> dict[str, float | int | str]:
    return {
        "object_key": sample.object_key,
        "object_name": sample.object_name,
        "object_kind": sample.object_kind,
        "generated_count": int(sample.generated_poses.shape[0]),
        "joint_mean": float(np.mean(sample.generated_joint_limit_loss)),
        "joint_min": float(np.min(sample.generated_joint_limit_loss)),
        "joint_max": float(np.max(sample.generated_joint_limit_loss)),
        "joint_success_ratio": float(np.mean(sample.generated_joint_limit_loss <= 0.0)),
        "root_mean": float(np.mean(sample.generated_root_distance_loss)),
        "root_min": float(np.min(sample.generated_root_distance_loss)),
        "root_max": float(np.max(sample.generated_root_distance_loss)),
        "root_success_ratio": float(np.mean(sample.generated_root_distance_loss <= 0.0)),
        "artifact_path": str(sample.artifact_path),
    }


def save_dga_style_samples(
    output_dir: str | Path,
    *,
    checkpoint_path: str | Path,
    samples: list[ObjectConditionedSample],
    architecture: str,
    hand_side: str,
    seed: int,
    num_generated: int,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "method": f"grasp_refine@{architecture}",
        "desc": "Object-conditioned sampling without source-grasp conditioning",
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "seed": int(seed),
        "num_generated": int(num_generated),
        "hand_side": str(hand_side),
        "sample_qpos": {sample.object_key: np.asarray(sample.generated_poses, dtype=np.float32) for sample in samples},
        "joint_limit_loss": {
            sample.object_key: np.asarray(sample.generated_joint_limit_loss, dtype=np.float32) for sample in samples
        },
        "root_distance_loss": {
            sample.object_key: np.asarray(sample.generated_root_distance_loss, dtype=np.float32) for sample in samples
        },
        "object_metadata": {sample.object_key: dict(sample.object_metadata) for sample in samples},
        "artifact_path": {sample.object_key: str(sample.artifact_path) for sample in samples},
        "object_name": {sample.object_key: sample.object_name for sample in samples},
        "object_kind": {sample.object_key: sample.object_kind for sample in samples},
    }
    summary = {
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": str(architecture),
        "hand_side": str(hand_side),
        "seed": int(seed),
        "num_generated": int(num_generated),
        "objects": [_summary_for_sample(sample) for sample in samples],
    }

    pickle_path = output_dir / "res_diffuser.pkl"
    with pickle_path.open("wb") as stream:
        pickle.dump(payload, stream)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return pickle_path, summary_path
