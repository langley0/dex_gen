#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_refine import RefineConfig, refine_source_grasp, save_refine_run
from scripts._grasp_refine_bridge import (
    load_source_artifact,
    make_single_callbacks,
    select_source_grasp,
)


def _default_output_path(result_path: Path, state_name: str, sample_index: int) -> Path:
    sample_token = f"best{sample_index}" if sample_index >= 0 else "bestauto"
    return ROOT / "outputs" / "grasp_refine" / f"{result_path.stem}_{state_name}_{sample_token}_refine.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local grasp refinement on a grasp_gen result.")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--state", choices=("best", "current"), default="best")
    parser.add_argument("--index", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=32)
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
    parser.add_argument("--object-density", type=float, default=400.0)
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
        actual_object_density=float(args.object_density),
    )

    source_artifact = load_source_artifact(args.result)
    source = select_source_grasp(source_artifact, state_name=args.state, index=int(args.index))
    metadata = dict(source_artifact.metadata)
    contact_target_local, callbacks = make_single_callbacks(source, metadata=metadata, config=config)
    artifact = refine_source_grasp(source, contact_target_local=contact_target_local, callbacks=callbacks, config=config)
    source_meta = artifact.metadata["source"]
    output_path = _default_output_path(args.result.resolve(), args.state, int(source_meta["sample_index"])) if args.output is None else args.output.resolve()
    save_refine_run(output_path, metadata=artifact.metadata, state=artifact.state)

    state = artifact.state
    print(f"output path         : {output_path}")
    print(f"source result       : {source_meta['result_path']}")
    print(f"state / sample      : {source_meta['state']} / {source_meta['sample_index']}")
    print(f"initial total       : {state.initial_energy.total:.6f}")
    print(f"final total         : {state.final_energy.total:.6f}")
    print(f"best total          : {state.best_energy.total:.6f}")
    print(f"initial penetration : {state.initial_energy.penetration:.6f}")
    print(f"final penetration   : {state.final_energy.penetration:.6f}")
    print(f"best penetration    : {state.best_energy.penetration:.6f}")
    print(
        "initial actual      : "
        f"contacts={state.initial_actual_contact_count} "
        f"penetrations={state.initial_actual_penetration_count} "
        f"depth_sum={state.initial_actual_depth_sum:.6f}"
    )
    print(
        "best actual         : "
        f"contacts={state.best_actual_contact_count} "
        f"penetrations={state.best_actual_penetration_count} "
        f"depth_sum={state.best_actual_depth_sum:.6f}"
    )
    print(f"active joints(final): {int(state.history_active_joint_count[-1])}")


if __name__ == "__main__":
    main()
