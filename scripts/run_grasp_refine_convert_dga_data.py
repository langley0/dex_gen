#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_refine import DatasetConfig, build_dga_data_records, load_source_records, resolve_artifact_paths
from grasp_refine.geometry import ortho6d_to_matrix, quat_to_matrix
from grasp_refine.object_mesh import load_object_mesh, sample_mesh_points


def _stable_kind_seed(object_kind: str, base_seed: int) -> int:
    stable_hash = sum((index + 1) * ord(char) for index, char in enumerate(object_kind))
    return int(base_seed) + stable_hash % 100_000


def _canonical_cloud(record, config: DatasetConfig, cache: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    cached = cache.get(record.object_kind)
    if cached is not None:
        return cached
    mesh = load_object_mesh(record.object_metadata)
    seed = _stable_kind_seed(record.object_kind, config.object_point_seed)
    points, normals = sample_mesh_points(mesh, config.object_num_points, seed=seed)
    cached = (points.astype(np.float32), normals.astype(np.float32))
    cache[record.object_kind] = cached
    return cached


def _world_cloud(record, canonical_points: np.ndarray, canonical_normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    object_pos = np.asarray(record.object_metadata.get("pos", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
    object_quat = np.asarray(record.object_metadata.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32).reshape(4)
    object_rotation = quat_to_matrix(object_quat)
    world_points = canonical_points @ object_rotation.T + object_pos[None, :]
    world_normals = canonical_normals @ object_rotation.T
    return world_points.astype(np.float32), world_normals.astype(np.float32)


def _validate_geometric_equivalence(source_records, converted_records, config: DatasetConfig) -> dict[str, float | int | str]:
    if len(source_records) != len(converted_records):
        raise ValueError(f"Source/converted record count mismatch: {len(source_records)} vs {len(converted_records)}")

    cloud_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    max_pose_abs_error = 0.0
    max_point_abs_error = 0.0
    max_normal_abs_error = 0.0

    for source, converted in zip(source_records, converted_records, strict=True):
        expected_pose = np.concatenate([source.full_pose[:3], source.full_pose[9:]], axis=0).astype(np.float32)
        pose_error = float(np.max(np.abs(expected_pose - converted.pose)))
        max_pose_abs_error = max(max_pose_abs_error, pose_error)

        canonical_points, canonical_normals = _canonical_cloud(source, config, cloud_cache)
        world_points, world_normals = _world_cloud(source, canonical_points, canonical_normals)

        hand_root_pos = np.asarray(source.full_pose[:3], dtype=np.float32).reshape(3)
        hand_root_rot = ortho6d_to_matrix(np.asarray(source.full_pose[3:9], dtype=np.float32))

        if config.coordinate_mode == "hand_aligned_object":
            original_points_in_hand_frame = (world_points - hand_root_pos[None, :]) @ hand_root_rot
            converted_points_in_hand_frame = converted.object_points - hand_root_pos[None, :]
            original_normals_in_hand_frame = world_normals @ hand_root_rot
            converted_normals_in_hand_frame = converted.object_normals
        elif config.coordinate_mode == "world_object_rotated":
            original_points_in_hand_frame = world_points
            converted_points_in_hand_frame = converted.object_points
            original_normals_in_hand_frame = world_normals
            converted_normals_in_hand_frame = converted.object_normals
        else:
            raise ValueError(f"Unsupported coordinate mode: {config.coordinate_mode!r}")

        point_error = float(np.max(np.abs(original_points_in_hand_frame - converted_points_in_hand_frame)))
        normal_error = float(np.max(np.abs(original_normals_in_hand_frame - converted_normals_in_hand_frame)))
        max_point_abs_error = max(max_point_abs_error, point_error)
        max_normal_abs_error = max(max_normal_abs_error, normal_error)

    tolerance = 1.0e-4
    return {
        "record_count": int(len(source_records)),
        "coordinate_mode": str(config.coordinate_mode),
        "max_pose_abs_error": float(max_pose_abs_error),
        "max_point_abs_error": float(max_point_abs_error),
        "max_normal_abs_error": float(max_normal_abs_error),
        "pose_ok": bool(max_pose_abs_error <= tolerance),
        "points_ok": bool(max_point_abs_error <= tolerance),
        "normals_ok": bool(max_normal_abs_error <= tolerance),
        "tolerance": float(tolerance),
    }


def _save_converted_records(path: Path, converted_records, config: DatasetConfig, validation: dict[str, float | int | str]) -> Path:
    output_path = path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "pose": np.stack([record.pose for record in converted_records], axis=0).astype(np.float32),
        "pose_full": np.stack([record.pose_full for record in converted_records], axis=0).astype(np.float32),
        "object_points": np.stack([record.object_points for record in converted_records], axis=0).astype(np.float32),
        "object_normals": np.stack([record.object_normals for record in converted_records], axis=0).astype(np.float32),
        "contact_indices": np.stack([record.contact_indices for record in converted_records], axis=0).astype(np.int32),
        "total_energy": np.asarray([record.total_energy for record in converted_records], dtype=np.float32),
        "sample_index": np.asarray([record.sample_index for record in converted_records], dtype=np.int32),
        "source_path": np.asarray([record.source_path for record in converted_records], dtype=np.str_),
        "hand_side": np.asarray([record.hand_side for record in converted_records], dtype=np.str_),
        "object_kind": np.asarray([record.object_kind for record in converted_records], dtype=np.str_),
        "object_name": np.asarray([record.object_name for record in converted_records], dtype=np.str_),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "dataset_config": {
                        "artifact_paths": [str(path) for path in config.artifact_paths],
                        "artifact_glob": config.artifact_glob,
                        "state_name": config.state_name,
                        "object_num_points": int(config.object_num_points),
                        "object_point_seed": int(config.object_point_seed),
                        "coordinate_mode": config.coordinate_mode,
                        "drop_invalid_samples": bool(config.drop_invalid_samples),
                    },
                    "validation": validation,
                },
                sort_keys=True,
            ),
            dtype=np.str_,
        ),
    }
    np.savez_compressed(output_path, **payload)
    return output_path


def _default_output_path(artifact_paths: tuple[Path, ...], state_name: str, coordinate_mode: str) -> Path:
    stem = artifact_paths[0].stem if len(artifact_paths) == 1 else f"bundle_{len(artifact_paths)}"
    return ROOT / "outputs" / "grasp_refine_dga_data" / f"{stem}_{state_name}_{coordinate_mode}.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert grasp_gen artifacts into DGA-style data records and validate geometric equivalence."
    )
    parser.add_argument("--artifact", type=Path, action="append", default=None, help="Input artifact (.npz). Repeatable.")
    parser.add_argument("--artifact-glob", type=str, default=None, help="Optional glob for multiple input artifacts.")
    parser.add_argument("--state", choices=("best", "current"), default="best")
    parser.add_argument("--object-num-points", type=int, default=2048)
    parser.add_argument("--object-point-seed", type=int, default=13)
    parser.add_argument("--coordinate-mode", choices=("hand_aligned_object", "world_object_rotated"), default="hand_aligned_object")
    parser.add_argument("--output", type=Path, default=None, help="Optional output .npz path.")
    parser.add_argument("--validate-only", action="store_true", help="Run conversion+validation without saving the converted dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DatasetConfig(
        artifact_paths=tuple(path.expanduser().resolve() for path in (args.artifact or [])),
        artifact_glob=args.artifact_glob,
        state_name=args.state,
        object_num_points=int(args.object_num_points),
        object_point_seed=int(args.object_point_seed),
        coordinate_mode=args.coordinate_mode,
    )

    artifact_paths = tuple(resolve_artifact_paths(config))
    source_records = load_source_records(config)
    converted_records = build_dga_data_records(config)
    validation = _validate_geometric_equivalence(source_records, converted_records, config)

    print(f"artifacts           : {len(artifact_paths)}")
    print(f"records             : {len(converted_records)}")
    print(f"state               : {config.state_name}")
    print(f"coordinate mode     : {config.coordinate_mode}")
    print(f"object num points   : {config.object_num_points}")
    print(f"max pose abs error  : {validation['max_pose_abs_error']:.8f}")
    print(f"max point abs error : {validation['max_point_abs_error']:.8f}")
    print(f"max normal abs error: {validation['max_normal_abs_error']:.8f}")
    print(f"pose ok             : {validation['pose_ok']}")
    print(f"points ok           : {validation['points_ok']}")
    print(f"normals ok          : {validation['normals_ok']}")

    if not (validation["pose_ok"] and validation["points_ok"] and validation["normals_ok"]):
        raise SystemExit("Geometric equivalence validation failed.")

    if args.validate_only:
        return

    output_path = _default_output_path(artifact_paths, config.state_name, config.coordinate_mode) if args.output is None else args.output
    saved_path = _save_converted_records(output_path, converted_records, config, validation)
    print(f"output              : {saved_path}")


if __name__ == "__main__":
    main()
