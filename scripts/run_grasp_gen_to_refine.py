#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_refine import format_command, run_grasp_gen_to_refine_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grasp_gen, select top grasps, train grasp_refine, and print viewer commands.")
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument("--object", choices=("cylinder", "cube", "drill", "decor01"), default="cylinder")
    parser.add_argument("--envs", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optimizer-backend", choices=("auto", "cpu", "gpu", "cuda", "tpu"), default="gpu")
    parser.add_argument("--refine-device", type=str, default="gpu")
    parser.add_argument("--refine-preset", type=str, default="auto")
    parser.add_argument("--refine-epochs", type=int, default=None)
    parser.add_argument("--refine-batch-size", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--regenerate-optimizer", action="store_true", help="Force a fresh grasp_gen optimizer run even if output already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.envs <= 0:
        raise SystemExit("--envs must be positive.")
    if args.steps <= 0:
        raise SystemExit("--steps must be positive.")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive.")

    result = run_grasp_gen_to_refine_pipeline(
        hand=args.hand,
        object_name=args.object,
        envs=int(args.envs),
        steps=int(args.steps),
        top_k=int(args.top_k),
        seed=int(args.seed),
        optimizer_backend=args.optimizer_backend,
        refine_device=args.refine_device,
        refine_preset=args.refine_preset,
        refine_epochs=None if args.refine_epochs is None else int(args.refine_epochs),
        refine_batch_size=None if args.refine_batch_size is None else int(args.refine_batch_size),
        regenerate_optimizer=bool(args.regenerate_optimizer),
        output_root=args.output_root,
    )

    print(f"pipeline root       : {result.paths.output_root}")
    print(f"optimizer artifact  : {result.paths.optimizer_artifact}")
    print(f"selected artifact   : {result.paths.selected_artifact}")
    print(f"selected count      : {result.subset.selected_count}")
    print(f"refine output dir   : {result.paths.refine_output_dir}")
    print(f"train config        : {result.paths.train_config_path}")
    print(f"viewer commands     : {result.paths.viewer_commands_path}")
    print("grasp_gen viewer    :")
    print(format_command(result.grasp_view_command))
    print("grasp_refine viewer :")
    print(format_command(result.model_view_command))


if __name__ == "__main__":
    main()
