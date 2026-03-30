#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_refine import RefineConfig, refine_result_batch, save_refine_batch
from scripts._grasp_refine_bridge import load_source_artifact, make_batch_callbacks, select_batch_source


def _default_output_path(result_path: Path, state_name: str) -> Path:
    return ROOT / "outputs" / "grasp_refine" / f"{result_path.stem}_{state_name}_batch_refine.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch grasp refinement on all samples from a grasp_gen result.")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--state", choices=("best", "current"), default="best")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--penetration-weight", type=float, default=60.0)
    parser.add_argument("--contact-weight", type=float, default=30.0)
    parser.add_argument("--support-weight", type=float, default=12.0)
    parser.add_argument("--distance-weight", type=float, default=0.1)
    parser.add_argument("--equilibrium-weight", type=float, default=0.05)
    parser.add_argument("--root-reg-weight", type=float, default=1.0)
    parser.add_argument("--joint-reg-weight", type=float, default=0.1)
    parser.add_argument("--penetration-threshold", type=float, default=5.0e-4)
    parser.add_argument("--support-points-per-body", type=int, default=6)
    parser.add_argument("--support-distance-sigma", type=float, default=1.5e-2)
    parser.add_argument("--support-max-distance", type=float, default=3.0e-2)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RefineConfig(
        steps=int(args.steps),
        step_size=float(args.step_size),
        penetration_weight=float(args.penetration_weight),
        contact_weight=float(args.contact_weight),
        support_weight=float(args.support_weight),
        distance_weight=float(args.distance_weight),
        equilibrium_weight=float(args.equilibrium_weight),
        root_reg_weight=float(args.root_reg_weight),
        joint_reg_weight=float(args.joint_reg_weight),
        penetration_threshold=float(args.penetration_threshold),
        support_points_per_body=int(args.support_points_per_body),
        support_distance_sigma=float(args.support_distance_sigma),
        support_max_distance=float(args.support_max_distance),
    )
    source_artifact = load_source_artifact(args.result)
    metadata, initial_hand_pose, contact_indices, source_total = select_batch_source(source_artifact, state_name=args.state)
    contact_target_local, callbacks = make_batch_callbacks(
        metadata=metadata,
        initial_hand_pose=initial_hand_pose,
        contact_indices=contact_indices,
        config=config,
    )
    metadata, state = refine_result_batch(
        metadata=metadata,
        initial_hand_pose=initial_hand_pose,
        contact_indices=contact_indices,
        source_total=source_total,
        contact_target_local=contact_target_local,
        callbacks=callbacks,
        config=config,
    )
    output_path = _default_output_path(args.result.resolve(), args.state) if args.output is None else args.output.resolve()
    save_refine_batch(output_path, metadata=metadata, state=state)

    fixed_mask = np.asarray(state["fixed_mask"], dtype=bool)
    actual_fixed_mask = np.asarray(state["actual_fixed_mask"], dtype=bool)
    improved_mask = np.asarray(state["improved_mask"], dtype=bool)
    best_total = np.asarray(state["best_total"], dtype=np.float32)
    best_fixed_index = int(np.argmin(np.where(fixed_mask, best_total, np.inf))) if np.any(fixed_mask) else -1
    best_actual_fixed_index = int(np.argmin(np.where(actual_fixed_mask, best_total, np.inf))) if np.any(actual_fixed_mask) else -1

    print(f"output path         : {output_path}")
    print(f"source result       : {metadata['source']['result_path']}")
    print(f"state               : {metadata['source']['state']}")
    print(f"batch size          : {len(state['sample_indices'])}")
    print(f"improved count      : {int(np.count_nonzero(improved_mask))}")
    print(f"fixed count         : {int(np.count_nonzero(fixed_mask))}")
    print(f"actual fixed count  : {int(np.count_nonzero(actual_fixed_mask))}")
    print(f"mean initial pen    : {float(np.mean(state['initial_penetration'])):.6f}")
    print(f"mean best pen       : {float(np.mean(state['best_penetration'])):.6f}")
    print(f"mean init actual ds : {float(np.mean(state['initial_actual_depth_sum'])):.6f}")
    print(f"mean best actual ds : {float(np.mean(state['best_actual_depth_sum'])):.6f}")
    print(f"best fixed index    : {best_fixed_index}")
    print(f"best actual fixed   : {best_actual_fixed_index}")
    if best_fixed_index >= 0:
        print(f"best fixed total    : {float(state['best_total'][best_fixed_index]):.6f}")
        print(f"initial pen -> best : {float(state['initial_penetration'][best_fixed_index]):.6f} -> {float(state['best_penetration'][best_fixed_index]):.6f}")
    if best_actual_fixed_index >= 0:
        print(
            "init actual -> best : "
            f"{float(state['initial_actual_depth_sum'][best_actual_fixed_index]):.6f} -> "
            f"{float(state['best_actual_depth_sum'][best_actual_fixed_index]):.6f}"
        )


if __name__ == "__main__":
    main()
