#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


@dataclass(frozen=True)
class ObjectSpec:
    label: str
    object_name: str
    cube_size: float | None = None


OBJECT_SETS: dict[str, tuple[ObjectSpec, ...]] = {
    "stage1": (
        ObjectSpec(label="cylinder", object_name="cylinder"),
        ObjectSpec(label="cube_s006", object_name="cube", cube_size=0.06),
        ObjectSpec(label="cube_s008", object_name="cube", cube_size=0.08),
        ObjectSpec(label="drill", object_name="drill"),
        ObjectSpec(label="decor01", object_name="decor01"),
    ),
}


def _format_command(parts: list[str]) -> str:
    return " ".join(parts)


def _artifact_path(output_root: Path, spec: ObjectSpec) -> Path:
    return output_root / "optimizer" / spec.label / "grasp_optimizer.npz"


def _dataset_output_path(output_root: Path, object_set: str, state_name: str, coordinate_mode: str) -> Path:
    return output_root / "datasets" / f"{object_set}_{state_name}_{coordinate_mode}_normalized.npz"


def _optimizer_command(args: argparse.Namespace, spec: ObjectSpec, artifact_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_grasp_optimizer.py"),
        "--backend",
        args.backend,
        "--hand",
        args.hand,
        "--object",
        spec.object_name,
        "--batch",
        str(args.envs),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--output",
        str(artifact_path),
    ]
    if spec.cube_size is not None:
        command.extend(["--cube-size", str(spec.cube_size)])
    return command


def _dataset_command(args: argparse.Namespace, artifact_paths: list[Path], dataset_output: Path) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_grasp_refine_prepare_dga_dataset.py"),
        "--state",
        args.state,
        "--coordinate-mode",
        args.coordinate_mode,
        "--object-num-points",
        str(args.object_num_points),
        "--object-point-seed",
        str(args.object_point_seed),
        "--normalizer-padding",
        str(args.normalizer_padding),
        "--output",
        str(dataset_output),
    ]
    for artifact_path in artifact_paths:
        command.extend(["--artifact", str(artifact_path)])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a multi-object grasp_gen dataset plan for grasp_refine.")
    parser.add_argument("--object-set", choices=tuple(OBJECT_SETS.keys()), default="stage1")
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu", "cuda", "tpu"), default="gpu")
    parser.add_argument("--envs", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state", choices=("best", "current"), default="best")
    parser.add_argument("--coordinate-mode", choices=("hand_aligned_object", "world_object_rotated"), default="hand_aligned_object")
    parser.add_argument("--object-num-points", type=int, default=2048)
    parser.add_argument("--object-point-seed", type=int, default=13)
    parser.add_argument("--normalizer-padding", type=float, default=0.02)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "grasp_multi_object_stage1")
    parser.add_argument("--run-optimizer", action="store_true")
    parser.add_argument("--prepare-dataset", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    object_specs = OBJECT_SETS[str(args.object_set)]
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    artifact_paths = [_artifact_path(output_root, spec) for spec in object_specs]
    optimizer_commands = [_optimizer_command(args, spec, path) for spec, path in zip(object_specs, artifact_paths, strict=True)]
    dataset_output = _dataset_output_path(output_root, str(args.object_set), str(args.state), str(args.coordinate_mode))
    dataset_command = _dataset_command(args, artifact_paths, dataset_output)

    manifest = {
        "object_set": str(args.object_set),
        "hand": str(args.hand),
        "backend": str(args.backend),
        "envs": int(args.envs),
        "steps": int(args.steps),
        "seed": int(args.seed),
        "state": str(args.state),
        "coordinate_mode": str(args.coordinate_mode),
        "artifacts": [
            {
                **asdict(spec),
                "artifact_path": str(path),
            }
            for spec, path in zip(object_specs, artifact_paths, strict=True)
        ],
        "dataset_output": str(dataset_output),
        "optimizer_commands": [_format_command(command) for command in optimizer_commands],
        "dataset_command": _format_command(dataset_command),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if args.run_optimizer:
        for command, artifact_path in zip(optimizer_commands, artifact_paths, strict=True):
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[RUN] {_format_command(command)}")
            subprocess.run(command, check=True, cwd=ROOT)

    if args.prepare_dataset:
        dataset_output.parent.mkdir(parents=True, exist_ok=True)
        print(f"[RUN] {_format_command(dataset_command)}")
        subprocess.run(dataset_command, check=True, cwd=ROOT)

    print(f"manifest              : {manifest_path}")
    print(f"dataset output        : {dataset_output}")
    print("objects               :")
    for spec, path in zip(object_specs, artifact_paths, strict=True):
        if spec.cube_size is None:
            print(f"  - {spec.label:12s} object={spec.object_name:8s} artifact={path}")
        else:
            print(f"  - {spec.label:12s} object={spec.object_name:8s} cube_size={spec.cube_size:.3f} artifact={path}")
    print("optimizer commands    :")
    for command in optimizer_commands:
        print(f"  {_format_command(command)}")
    print("dataset command       :")
    print(f"  {_format_command(dataset_command)}")


if __name__ == "__main__":
    main()
