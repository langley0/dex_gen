#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_refine import DatasetConfig
from grasp_refine.materialize import build_materialized_dga_dataset, load_saved_normalizer, save_materialized_dga_dataset


def _default_output_path(config: DatasetConfig) -> Path:
    if config.artifact_paths:
        stem = Path(config.artifact_paths[0]).stem if len(config.artifact_paths) == 1 else f"bundle_{len(config.artifact_paths)}"
    else:
        stem = "artifact_glob"
    return ROOT / "outputs" / "grasp_refine_dga_data" / f"{stem}_{config.state_name}_{config.coordinate_mode}_normalized.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a normalized DGA-style dataset from grasp_gen artifacts.")
    parser.add_argument("--artifact", type=Path, action="append", default=None, help="Input artifact (.npz). Repeatable.")
    parser.add_argument("--artifact-glob", type=str, default=None, help="Optional glob for multiple input artifacts.")
    parser.add_argument("--state", choices=("best", "current"), default="best")
    parser.add_argument("--object-num-points", type=int, default=2048)
    parser.add_argument("--object-point-seed", type=int, default=13)
    parser.add_argument("--coordinate-mode", choices=("hand_aligned_object", "world_object_rotated"), default="hand_aligned_object")
    parser.add_argument("--normalizer-padding", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=None, help="Optional output .npz path.")
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

    bundle = build_materialized_dga_dataset(config, normalizer_padding=float(args.normalizer_padding))
    output_path = _default_output_path(config) if args.output is None else args.output
    saved_path = save_materialized_dga_dataset(output_path, bundle)

    restored_normalizer = load_saved_normalizer(saved_path)
    restored_pose = restored_normalizer.denormalize_numpy(bundle.arrays.pose)
    roundtrip_max_abs_error = float(np.max(np.abs(restored_pose - bundle.arrays.pose_raw)))

    print(f"records                : {bundle.arrays.pose.shape[0]}")
    print(f"pose dim               : {bundle.arrays.pose.shape[1]}")
    print(f"hand side              : {bundle.hand_spec.side}")
    print(f"joint dim              : {bundle.hand_spec.joint_dim}")
    print(f"coordinate mode        : {bundle.config.coordinate_mode}")
    print(f"translation lower      : {bundle.normalizer.translation_lower}")
    print(f"translation upper      : {bundle.normalizer.translation_upper}")
    print(f"normalized pose min    : {bundle.arrays.pose.min():.8f}")
    print(f"normalized pose max    : {bundle.arrays.pose.max():.8f}")
    print(f"roundtrip max abs error: {roundtrip_max_abs_error:.8f}")
    print(f"output                 : {saved_path}")


if __name__ == "__main__":
    main()
