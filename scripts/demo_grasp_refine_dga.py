#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_refine import RefineConfig, refine_source_grasp, save_refine_run
from scripts._grasp_refine_bridge import load_source_artifact, make_single_callbacks, select_source_grasp


def _default_output_path(result_path: Path, state_name: str, sample_index: int) -> Path:
    sample_token = f"idx{sample_index}" if sample_index >= 0 else "bestauto"
    return ROOT / "outputs" / "grasp_refine" / f"{result_path.stem}_{state_name}_{sample_token}_dga_demo_refine.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a DexGrasp Anything-style refine result and open the viewer.")
    parser.add_argument("--source-result", type=Path, default=ROOT / "outputs" / "grasp_optimizer" / "drill_b64_s5000_latest.npz")
    parser.add_argument("--state", choices=("best", "current"), default="best")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--grad-scale", type=float, default=0.1)
    parser.add_argument("--noise-scale-start", type=float, default=5.0e-3)
    parser.add_argument("--noise-scale-end", type=float, default=1.0e-3)
    parser.add_argument("--surface-pull-weight", type=float, default=1.0)
    parser.add_argument("--external-repulsion-weight", type=float, default=0.3)
    parser.add_argument("--self-repulsion-weight", type=float, default=1.0)
    parser.add_argument("--surface-pull-threshold", type=float, default=2.0e-2)
    parser.add_argument("--self-repulsion-threshold", type=float, default=2.0e-2)
    parser.add_argument("--external-threshold", type=float, default=1.0e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--object-density", type=float, default=400.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pose", choices=("initial", "refined", "best"), default="best")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument("--bright-bg", action="store_true")
    return parser.parse_args()


def _generate_if_needed(args: argparse.Namespace) -> Path:
    output_path = args.output.resolve() if args.output is not None else _default_output_path(
        args.source_result.resolve(),
        args.state,
        int(args.index),
    )
    if output_path.exists() and not args.regenerate:
        print(f"reuse refine result : {output_path}")
        return output_path

    config = RefineConfig(
        steps=int(args.steps),
        guidance_scale=float(args.guidance_scale),
        grad_scale=float(args.grad_scale),
        noise_scale_start=float(args.noise_scale_start),
        noise_scale_end=float(args.noise_scale_end),
        surface_pull_weight=float(args.surface_pull_weight),
        external_repulsion_weight=float(args.external_repulsion_weight),
        self_repulsion_weight=float(args.self_repulsion_weight),
        surface_pull_threshold=float(args.surface_pull_threshold),
        self_repulsion_threshold=float(args.self_repulsion_threshold),
        external_threshold=float(args.external_threshold),
        grad_clip_norm=float(args.grad_clip_norm),
        actual_object_density=float(args.object_density),
        seed=int(args.seed),
    )

    source_artifact = load_source_artifact(args.source_result)
    source = select_source_grasp(source_artifact, state_name=args.state, index=int(args.index))
    metadata = dict(source_artifact.metadata)
    contact_target_local, callbacks = make_single_callbacks(source, metadata=metadata, config=config)
    artifact = refine_source_grasp(source, contact_target_local=contact_target_local, callbacks=callbacks, config=config)
    save_refine_run(output_path, metadata=artifact.metadata, state=artifact.state)

    print(f"generated refine result: {output_path}")
    print(f"initial total         : {artifact.state.initial_energy.total:.6f}")
    print(f"best total            : {artifact.state.best_energy.total:.6f}")
    print(f"best actual depth_sum : {artifact.state.best_actual_depth_sum:.6f}")
    return output_path


def main() -> None:
    args = parse_args()
    output_path = _generate_if_needed(args)
    if args.no_view:
        return

    view_argv = [
        sys.executable,
        str(ROOT / "scripts" / "view_grasp_refine.py"),
        "--result",
        str(output_path),
        "--pose",
        args.pose,
    ]
    if args.bright_bg:
        view_argv.append("--bright-bg")
    os.execv(sys.executable, view_argv)


if __name__ == "__main__":
    main()
