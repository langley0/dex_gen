#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_refine.checkpoint import load_training_checkpoint
from grasp_refine.config_io import training_config_from_dict
from grasp_refine.diffusion import make_diffusion_schedule
from grasp_refine.inspire_hand import load_inspire_hand_spec
from grasp_refine.io import load_grasp_artifact
from grasp_refine.object_mesh import load_object_mesh, sample_mesh_points
from grasp_refine.sampling import sample_grasp_poses


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare generated grasp quality across grasp_refine checkpoints.")
    parser.add_argument("--artifact", type=Path, required=True, help="Source artifact (.npz).")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True, help="Checkpoint to evaluate. Repeatable.")
    parser.add_argument("--state", choices=("best", "current"), default="best")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-generated", type=int, default=32)
    parser.add_argument("--object-num-points", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def _stable_kind_seed(object_kind: str, base_seed: int) -> int:
    stable_hash = sum((index + 1) * ord(char) for index, char in enumerate(object_kind))
    return int(base_seed) + stable_hash % 100_000


def _scalar_losses(
    poses: np.ndarray,
    *,
    hand_spec,
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


def _summarize(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _evaluate_checkpoint(
    checkpoint_path: Path,
    artifact_path: Path,
    *,
    state_name: str,
    sample_index: int,
    seed: int,
    num_generated: int,
    object_num_points: int | None,
    device: str | None,
) -> dict[str, object]:
    payload = load_training_checkpoint(checkpoint_path)
    config = training_config_from_dict(payload["config"])
    if device is not None:
        config = replace(config, device=device)

    record = load_grasp_artifact(artifact_path, state_name=state_name).records[int(sample_index)]
    hand_spec = load_inspire_hand_spec(config.hand_side)
    model_config = config.model
    if model_config.pose_dim != hand_spec.pose_dim:
        model_config = replace(model_config, pose_dim=hand_spec.pose_dim)

    num_points = config.dataset.object_num_points if object_num_points is None else int(object_num_points)
    mesh = load_object_mesh(record.object_metadata)
    point_seed = _stable_kind_seed(record.object_kind, config.dataset.object_point_seed)
    object_points, object_normals = sample_mesh_points(mesh, num_points, seed=point_seed)

    schedule = make_diffusion_schedule(config.diffusion)
    params = payload["params"]
    object_points_jax = jax.numpy.asarray(object_points)
    object_normals_jax = jax.numpy.asarray(object_normals)
    if device is not None:
        try:
            jax_device = jax.devices(device)[0]
        except Exception:
            jax_device = jax.devices()[0]
        params = jax.device_put(params, jax_device)
        schedule = jax.device_put(schedule, jax_device)
        object_points_jax = jax.device_put(object_points_jax, jax_device)
        object_normals_jax = jax.device_put(object_normals_jax, jax_device)

    rng_key = jax.random.key(np.uint32(int(seed) % (2**32)))
    sample_batch = sample_grasp_poses(
        params,
        object_points=object_points_jax,
        object_normals=object_normals_jax,
        rng_key=rng_key,
        model_config=model_config,
        diffusion_config=config.diffusion,
        normalizer=payload["normalizer"],
        schedule=schedule,
        num_samples=int(num_generated),
    )
    generated_poses = np.asarray(sample_batch.pose, dtype=np.float32)

    source_joint, source_root = _scalar_losses(
        record.pose,
        hand_spec=hand_spec,
        object_points=object_points,
        root_distance_threshold=config.loss.root_distance_threshold,
    )
    generated_joint, generated_root = _scalar_losses(
        generated_poses,
        hand_spec=hand_spec,
        object_points=object_points,
        root_distance_threshold=config.loss.root_distance_threshold,
    )

    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "artifact": str(artifact_path.resolve()),
        "architecture": model_config.architecture,
        "object_name": record.object_name,
        "source_joint_loss": float(source_joint[0]),
        "source_root_loss": float(source_root[0]),
        "generated_joint": _summarize(generated_joint),
        "generated_root": _summarize(generated_root),
        "generated_root_below_threshold_ratio": float(np.mean(generated_root <= 0.0)),
        "generated_joint_below_threshold_ratio": float(np.mean(generated_joint <= 0.0)),
        "generated_count": int(generated_poses.shape[0]),
    }


def main() -> None:
    args = parse_args()
    results = []
    for checkpoint in args.checkpoint:
        results.append(
            _evaluate_checkpoint(
                checkpoint.expanduser().resolve(),
                args.artifact.expanduser().resolve(),
                state_name=args.state,
                sample_index=int(args.sample_index),
                seed=int(args.seed),
                num_generated=int(args.num_generated),
                object_num_points=args.object_num_points,
                device=args.device,
            )
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
