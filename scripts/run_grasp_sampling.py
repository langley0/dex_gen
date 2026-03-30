#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_gen.grasp_optimizer_io import load_grasp_run
from grasp_sampling import PhysicsEvalConfig, evaluate_source_grasp, save_sampling_run, select_source_grasp


def _default_output_path(result_path: Path, state_name: str, sample_index: int) -> Path:
    stem = result_path.stem
    sample_token = f"best{sample_index}" if sample_index >= 0 else "bestauto"
    return ROOT / "outputs" / "grasp_sampling" / f"{stem}_{state_name}_{sample_token}_physics.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run physical grasp tests from a grasp_gen optimizer result.")
    parser.add_argument("--result", type=Path, required=True, help="Input grasp_gen optimizer artifact (.npz).")
    parser.add_argument("--state", choices=("best", "current"), default="best")
    parser.add_argument("--index", type=int, default=-1, help="Batch sample index, or -1 for lowest-energy sample.")
    parser.add_argument("--translation-delta", type=float, default=0.03)
    parser.add_argument("--rotation-delta-deg", type=float, default=25.0)
    parser.add_argument("--settle-time", type=float, default=0.20)
    parser.add_argument("--motion-time", type=float, default=0.60)
    parser.add_argument("--timestep", type=float, default=0.005)
    parser.add_argument("--lost-contact-steps", type=int, default=10)
    parser.add_argument("--max-translation-scale", type=float, default=0.75)
    parser.add_argument("--max-translation-min", type=float, default=0.05)
    parser.add_argument("--max-rotation-deg", type=float, default=75.0)
    parser.add_argument("--squeeze-max-delta", type=float, default=0.10)
    parser.add_argument("--squeeze-steps", type=int, default=5)
    parser.add_argument("--object-density", type=float, default=400.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.translation_delta < 0.0:
        raise SystemExit("--translation-delta must be non-negative.")
    if args.rotation_delta_deg < 0.0:
        raise SystemExit("--rotation-delta-deg must be non-negative.")
    if args.settle_time < 0.0:
        raise SystemExit("--settle-time must be non-negative.")
    if args.motion_time <= 0.0:
        raise SystemExit("--motion-time must be positive.")
    if args.timestep <= 0.0:
        raise SystemExit("--timestep must be positive.")
    if args.lost_contact_steps <= 0:
        raise SystemExit("--lost-contact-steps must be positive.")
    if args.max_translation_scale <= 0.0:
        raise SystemExit("--max-translation-scale must be positive.")
    if args.max_translation_min <= 0.0:
        raise SystemExit("--max-translation-min must be positive.")
    if args.max_rotation_deg <= 0.0:
        raise SystemExit("--max-rotation-deg must be positive.")
    if args.squeeze_max_delta < 0.0:
        raise SystemExit("--squeeze-max-delta must be non-negative.")
    if args.squeeze_steps <= 0:
        raise SystemExit("--squeeze-steps must be positive.")
    if args.object_density <= 0.0:
        raise SystemExit("--object-density must be positive.")

    artifact = load_grasp_run(args.result)
    source = select_source_grasp(artifact, state_name=args.state, index=args.index)
    output_path = _default_output_path(args.result.resolve(), args.state, source.sample_index) if args.output is None else args.output.resolve()

    config = PhysicsEvalConfig(
        translation_delta=args.translation_delta,
        rotation_delta_deg=args.rotation_delta_deg,
        settle_time=args.settle_time,
        motion_time=args.motion_time,
        timestep=args.timestep,
        lost_contact_steps=args.lost_contact_steps,
        max_translation_scale=args.max_translation_scale,
        max_translation_min=args.max_translation_min,
        max_rotation_deg=args.max_rotation_deg,
        squeeze_max_delta=args.squeeze_max_delta,
        squeeze_steps=args.squeeze_steps,
        object_density=args.object_density,
    )
    metadata, state = evaluate_source_grasp(source, config=config)
    save_sampling_run(output_path, metadata=metadata, state=state)

    chosen = int(state.chosen_attempt_index)
    print(f"output path        : {output_path}")
    print(f"source result      : {source.source_path}")
    print(f"state / sample     : {source.state_name} / {source.sample_index}")
    print(f"attempt count      : {state.attempt_count}")
    print(f"chosen attempt     : {chosen}")
    print(f"chosen squeeze rad : {float(state.squeeze_deltas[chosen]):.6f}")
    print(f"chosen score       : {float(state.overall_scores[chosen]):.6f}")
    print(f"success attempts   : {int(np.count_nonzero(state.success))}/{state.attempt_count}")
    print(
        "initial overlap    : "
        f"{bool(state.initial_overlap[chosen])} "
        f"(contacts={int(state.initial_contact_count[chosen])}, "
        f"penetrations={int(state.initial_penetration_count[chosen])}, "
        f"depth_sum={float(state.initial_depth_sum[chosen]):.6f})"
    )
    print(
        "object in hand     : "
        f"pos={np.array2string(state.object_relative_pos, precision=4)} "
        f"quat={np.array2string(state.object_relative_quat, precision=4)}"
    )
    print("motion scores      :")
    for motion_meta, score, lost, fail in zip(metadata["motions"], state.motion_scores[chosen], state.motion_lost[chosen], state.motion_fail[chosen]):
        if not np.isfinite(score):
            print(f"  {motion_meta['name']:>4} : not-run")
            continue
        print(
            f"  {motion_meta['name']:>4} : score={float(score):.6f} "
            f"lost={bool(lost)} fail={bool(fail)}"
        )


if __name__ == "__main__":
    main()
