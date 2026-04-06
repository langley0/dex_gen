#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

DEFAULT_CONFIG_PATH = ROOT / "configs" / "grasp_gen" / "multi_object.toml"
SUPPORTED_OBJECTS = {"cylinder", "cube", "box", "drill", "decor01"}
SUPPORTED_PRIMITIVES = {"cylinder", "cube", "box"}


@dataclass(frozen=True)
class RunConfig:
    hand: str
    backend: str
    output_root: Path
    skip_existing: bool


@dataclass(frozen=True)
class OptimizerConfig:
    batch: int
    steps: int
    points: int
    contact_count: int
    seed: int
    offset: float
    distance_weight: float
    equilibrium_weight: float
    penetration_weight: float
    wrench_iters: int
    sdf_voxel_size: float
    sdf_padding: float
    bench_steps: int


@dataclass(frozen=True)
class TargetSpec:
    label: str
    object_name: str
    source_kind: str
    asset_name: str | None = None
    cube_size: float | None = None
    box_size: tuple[float, float, float] | None = None
    cylinder_radius: float | None = None
    cylinder_half_height: float | None = None
    output_name: str = "grasp_optimizer.npz"


@dataclass(frozen=True)
class BatchConfig:
    config_path: Path
    run: RunConfig
    optimizer: OptimizerConfig
    targets: tuple[TargetSpec, ...]


def _resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def _require_keys(raw: dict[str, Any], section_name: str, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in raw]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{section_name} is missing required keys: {joined}")


def _parse_box_size(raw_value: Any, *, label: str) -> tuple[float, float, float]:
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 3:
        raise ValueError(f"Target {label!r} must set box_size = [x, y, z].")
    values = tuple(float(value) for value in raw_value)
    if any(value <= 0.0 for value in values):
        raise ValueError(f"Target {label!r} box_size entries must be positive.")
    return values


def _load_target(raw_target: dict[str, Any], *, index: int) -> TargetSpec:
    _require_keys(raw_target, f"targets[{index}]", ("label",))
    label = str(raw_target["label"]).strip()
    if not label:
        raise ValueError(f"targets[{index}] label must not be empty.")

    asset_value = raw_target.get("asset")
    object_value = raw_target.get("object")
    primitive_value = raw_target.get("primitive")
    defined_sources = sum(value is not None for value in (asset_value, object_value, primitive_value))
    if defined_sources == 0:
        raise ValueError(f"Target {label!r} must define one of 'asset', 'object', or 'primitive'.")
    if defined_sources > 1:
        raise ValueError(f"Target {label!r} must define exactly one of 'asset', 'object', or 'primitive'.")

    asset_name = None
    if asset_value is not None:
        asset_name = str(asset_value).strip()
        if not asset_name:
            raise ValueError(f"Target {label!r} asset must not be empty.")
        object_name = "asset"
        source_kind = "asset"
    elif primitive_value is not None:
        object_name = str(primitive_value).strip().lower()
        if object_name not in SUPPORTED_PRIMITIVES:
            supported = ", ".join(sorted(SUPPORTED_PRIMITIVES))
            raise ValueError(f"Target {label!r} primitive must be one of: {supported}.")
        source_kind = "primitive"
    else:
        object_name = str(object_value).strip().lower()
        if object_name not in SUPPORTED_OBJECTS:
            supported = ", ".join(sorted(SUPPORTED_OBJECTS))
            raise ValueError(f"Target {label!r} object must be one of: {supported}.")
        source_kind = "object"

    cube_size = raw_target.get("cube_size")
    if cube_size is not None:
        cube_size = float(cube_size)
        if cube_size <= 0.0:
            raise ValueError(f"Target {label!r} cube_size must be positive.")

    box_size = None
    if "box_size" in raw_target:
        box_size = _parse_box_size(raw_target["box_size"], label=label)

    cylinder_radius = raw_target.get("cylinder_radius")
    if cylinder_radius is not None:
        cylinder_radius = float(cylinder_radius)
        if cylinder_radius <= 0.0:
            raise ValueError(f"Target {label!r} cylinder_radius must be positive.")

    cylinder_half_height = raw_target.get("cylinder_half_height")
    if cylinder_half_height is not None:
        cylinder_half_height = float(cylinder_half_height)
        if cylinder_half_height <= 0.0:
            raise ValueError(f"Target {label!r} cylinder_half_height must be positive.")

    if object_name == "cube" and cube_size is None:
        cube_size = 0.07
    if object_name == "box" and box_size is None:
        raise ValueError(f"Target {label!r} must define box_size for primitive/object 'box'.")
    if object_name == "cylinder" and cylinder_radius is None:
        cylinder_radius = 0.045
    if object_name == "cylinder" and cylinder_half_height is None:
        cylinder_half_height = 0.165

    output_name = str(raw_target.get("output_name", "grasp_optimizer.npz")).strip()
    if not output_name:
        raise ValueError(f"Target {label!r} output_name must not be empty.")

    return TargetSpec(
        label=label,
        object_name=object_name,
        source_kind=source_kind,
        asset_name=asset_name,
        cube_size=cube_size,
        box_size=box_size,
        cylinder_radius=cylinder_radius,
        cylinder_half_height=cylinder_half_height,
        output_name=output_name,
    )


def load_batch_config(config_path: Path) -> BatchConfig:
    resolved_config_path = config_path.expanduser().resolve()
    with resolved_config_path.open("rb") as stream:
        raw = tomllib.load(stream)

    run_raw = raw.get("run", {})
    optimizer_raw = raw.get("optimizer", {})
    targets_raw = raw.get("targets")

    _require_keys(run_raw, "run", ("hand", "backend", "output_root"))
    _require_keys(
        optimizer_raw,
        "optimizer",
        (
            "batch",
            "steps",
            "points",
            "contact_count",
            "seed",
            "offset",
            "distance_weight",
            "equilibrium_weight",
            "penetration_weight",
            "wrench_iters",
            "sdf_voxel_size",
            "sdf_padding",
            "bench_steps",
        ),
    )

    if not isinstance(targets_raw, list) or not targets_raw:
        raise ValueError("Config must define at least one [[targets]] entry.")

    run = RunConfig(
        hand=str(run_raw["hand"]).strip(),
        backend=str(run_raw["backend"]).strip(),
        output_root=_resolve_project_path(run_raw["output_root"]),
        skip_existing=bool(run_raw.get("skip_existing", True)),
    )
    optimizer = OptimizerConfig(
        batch=int(optimizer_raw["batch"]),
        steps=int(optimizer_raw["steps"]),
        points=int(optimizer_raw["points"]),
        contact_count=int(optimizer_raw["contact_count"]),
        seed=int(optimizer_raw["seed"]),
        offset=float(optimizer_raw["offset"]),
        distance_weight=float(optimizer_raw["distance_weight"]),
        equilibrium_weight=float(optimizer_raw["equilibrium_weight"]),
        penetration_weight=float(optimizer_raw["penetration_weight"]),
        wrench_iters=int(optimizer_raw["wrench_iters"]),
        sdf_voxel_size=float(optimizer_raw["sdf_voxel_size"]),
        sdf_padding=float(optimizer_raw["sdf_padding"]),
        bench_steps=int(optimizer_raw["bench_steps"]),
    )
    targets = tuple(_load_target(target_raw, index=index) for index, target_raw in enumerate(targets_raw))

    config = BatchConfig(
        config_path=resolved_config_path,
        run=run,
        optimizer=optimizer,
        targets=targets,
    )
    _validate_batch_config(config)
    return config


def _validate_batch_config(config: BatchConfig) -> None:
    if config.run.hand not in {"right", "left"}:
        raise ValueError("run.hand must be 'right' or 'left'.")
    if config.run.backend not in {"auto", "cpu", "gpu", "cuda", "tpu"}:
        raise ValueError("run.backend must be one of auto/cpu/gpu/cuda/tpu.")
    if config.optimizer.batch <= 0:
        raise ValueError("optimizer.batch must be positive.")
    if config.optimizer.steps < 0:
        raise ValueError("optimizer.steps must be non-negative.")
    if config.optimizer.points <= 0:
        raise ValueError("optimizer.points must be positive.")
    if config.optimizer.contact_count <= 0:
        raise ValueError("optimizer.contact_count must be positive.")
    if config.optimizer.offset <= 0.0:
        raise ValueError("optimizer.offset must be positive.")
    if config.optimizer.distance_weight < 0.0:
        raise ValueError("optimizer.distance_weight must be non-negative.")
    if config.optimizer.equilibrium_weight < 0.0:
        raise ValueError("optimizer.equilibrium_weight must be non-negative.")
    if config.optimizer.penetration_weight < 0.0:
        raise ValueError("optimizer.penetration_weight must be non-negative.")
    if config.optimizer.wrench_iters <= 0:
        raise ValueError("optimizer.wrench_iters must be positive.")
    if config.optimizer.sdf_voxel_size <= 0.0:
        raise ValueError("optimizer.sdf_voxel_size must be positive.")
    if config.optimizer.sdf_padding < 0.0:
        raise ValueError("optimizer.sdf_padding must be non-negative.")
    if config.optimizer.bench_steps < 0:
        raise ValueError("optimizer.bench_steps must be non-negative.")

    seen_labels: set[str] = set()
    for target in config.targets:
        if target.label in seen_labels:
            raise ValueError(f"Duplicate target label: {target.label!r}")
        seen_labels.add(target.label)


def _artifact_path(output_root: Path, target: TargetSpec) -> Path:
    return output_root / target.label / target.output_name


def _format_command(parts: list[str]) -> str:
    return shlex.join(parts)


def _normalize_jax_platforms(raw_value: str) -> str:
    value = raw_value.strip().lower()
    if not value or value == "auto":
        return ""
    alias_map = {
        "gpu": "cuda",
        "cuda": "cuda",
        "cpu": "cpu",
        "tpu": "tpu",
        "rocm": "rocm",
    }
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return ",".join(alias_map.get(part, part) for part in parts)


def _command_for_target(config: BatchConfig, target: TargetSpec, artifact_path: Path) -> list[str]:
    optimizer = config.optimizer
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_grasp_optimizer.py"),
        "--backend",
        config.run.backend,
        "--hand",
        config.run.hand,
        "--batch",
        str(optimizer.batch),
        "--steps",
        str(optimizer.steps),
        "--points",
        str(optimizer.points),
        "--contact-count",
        str(optimizer.contact_count),
        "--seed",
        str(optimizer.seed),
        "--offset",
        str(optimizer.offset),
        "--distance-weight",
        str(optimizer.distance_weight),
        "--equilibrium-weight",
        str(optimizer.equilibrium_weight),
        "--penetration-weight",
        str(optimizer.penetration_weight),
        "--wrench-iters",
        str(optimizer.wrench_iters),
        "--sdf-voxel-size",
        str(optimizer.sdf_voxel_size),
        "--sdf-padding",
        str(optimizer.sdf_padding),
        "--bench-steps",
        str(optimizer.bench_steps),
        "--output",
        str(artifact_path),
    ]
    if target.asset_name is not None:
        command.extend(["--asset", target.asset_name])
    else:
        command.extend(["--object", target.object_name])
    if target.cube_size is not None:
        command.extend(["--cube-size", str(target.cube_size)])
    if target.box_size is not None:
        command.extend(["--box-size", *(str(value) for value in target.box_size)])
    if target.cylinder_radius is not None:
        command.extend(["--cylinder-radius", str(target.cylinder_radius)])
    if target.cylinder_half_height is not None:
        command.extend(["--cylinder-half-height", str(target.cylinder_half_height)])
    return command


def _namespace_for_target(config: BatchConfig, target: TargetSpec, artifact_path: Path) -> argparse.Namespace:
    optimizer = config.optimizer
    return argparse.Namespace(
        backend=config.run.backend,
        hand=config.run.hand,
        asset=target.asset_name,
        object=target.object_name,
        cube_size=0.07 if target.cube_size is None else target.cube_size,
        box_size=target.box_size,
        cylinder_radius=0.045 if target.cylinder_radius is None else target.cylinder_radius,
        cylinder_half_height=0.165 if target.cylinder_half_height is None else target.cylinder_half_height,
        batch=optimizer.batch,
        steps=optimizer.steps,
        points=optimizer.points,
        contact_count=optimizer.contact_count,
        seed=optimizer.seed,
        offset=optimizer.offset,
        distance_weight=optimizer.distance_weight,
        equilibrium_weight=optimizer.equilibrium_weight,
        penetration_weight=optimizer.penetration_weight,
        wrench_iters=optimizer.wrench_iters,
        sdf_voxel_size=optimizer.sdf_voxel_size,
        sdf_padding=optimizer.sdf_padding,
        bench_steps=optimizer.bench_steps,
        output=artifact_path,
    )


def _optimizer_runner(backend: str):
    module_name = "scripts.run_grasp_optimizer"
    normalized_platforms = _normalize_jax_platforms(backend)
    existing = sys.modules.get(module_name)
    if existing is not None:
        active = os.environ.get("JAX_PLATFORMS", "")
        if active != normalized_platforms:
            raise RuntimeError(
                "In-process optimizer execution already initialized a different JAX backend. "
                f"active={active!r}, requested={normalized_platforms!r}"
            )
        return existing

    os.environ["JAX_PLATFORMS"] = normalized_platforms
    return importlib.import_module(module_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grasp_gen sequentially from a TOML batch config.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Batch config TOML path.")
    parser.add_argument(
        "--label",
        dest="labels",
        action="append",
        default=None,
        help="Run only the given target label. Repeatable.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved commands without executing them.")
    parser.add_argument("--force", action="store_true", help="Run even if the target output already exists.")
    parser.add_argument(
        "--execution-mode",
        choices=("direct", "subprocess"),
        default="direct",
        help="How to run each optimizer target. 'direct' calls the optimizer function in-process.",
    )
    return parser.parse_args()


def _select_targets(config: BatchConfig, labels: list[str] | None) -> tuple[TargetSpec, ...]:
    if not labels:
        return config.targets
    requested = {label.strip() for label in labels if label.strip()}
    selected = tuple(target for target in config.targets if target.label in requested)
    missing = sorted(requested.difference(target.label for target in selected))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Unknown target labels: {joined}")
    return selected


def _manifest(config: BatchConfig, targets: tuple[TargetSpec, ...]) -> dict[str, Any]:
    artifacts = [_artifact_path(config.run.output_root, target) for target in targets]
    commands = [_command_for_target(config, target, artifact_path) for target, artifact_path in zip(targets, artifacts, strict=True)]
    return {
        "config_path": str(config.config_path),
        "run": {
            **asdict(config.run),
            "output_root": str(config.run.output_root),
        },
        "optimizer": asdict(config.optimizer),
        "targets": [
            {
                **asdict(target),
                "artifact_path": str(artifact_path),
            }
            for target, artifact_path in zip(targets, artifacts, strict=True)
        ],
        "commands": [_format_command(command) for command in commands],
    }


def main() -> None:
    args = parse_args()
    config = load_batch_config(args.config)
    targets = _select_targets(config, args.labels)
    manifest = _manifest(config, targets)
    manifest_path = config.run.output_root / "manifest.json"

    print(f"config path        : {config.config_path}")
    print(f"output root        : {config.run.output_root}")
    print(f"target count       : {len(targets)}")
    print(f"skip existing      : {config.run.skip_existing}")
    print(f"force              : {args.force}")
    print(f"dry run            : {args.dry_run}")
    print(f"execution mode     : {args.execution_mode}")
    print("targets            :")
    for target in targets:
        artifact_path = _artifact_path(config.run.output_root, target)
        source_value = target.asset_name if target.asset_name is not None else target.object_name
        source = f"{target.source_kind}={source_value}"
        extra_parts: list[str] = []
        if target.cube_size is not None:
            extra_parts.append(f"cube_size={target.cube_size:.3f}")
        if target.box_size is not None:
            extra_parts.append(f"box_size={list(target.box_size)}")
        if target.cylinder_radius is not None:
            extra_parts.append(f"cylinder_radius={target.cylinder_radius:.3f}")
        if target.cylinder_half_height is not None:
            extra_parts.append(f"cylinder_half_height={target.cylinder_half_height:.3f}")
        extras = "" if not extra_parts else " " + " ".join(extra_parts)
        print(f"  - {target.label:16s} {source}{extras} -> {artifact_path}")

    if args.dry_run:
        print("commands           :")
        for command in manifest["commands"]:
            print(f"  {command}")
        return

    config.run.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest           : {manifest_path}")

    optimizer_module = None
    if args.execution_mode == "direct":
        optimizer_module = _optimizer_runner(config.run.backend)

    for target in targets:
        artifact_path = _artifact_path(config.run.output_root, target)
        command = _command_for_target(config, target, artifact_path)
        should_skip = config.run.skip_existing and artifact_path.exists() and not args.force
        if should_skip:
            print(f"[SKIP] {target.label} -> {artifact_path}")
            continue

        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[RUN]  {_format_command(command)}")
        if args.execution_mode == "subprocess":
            subprocess.run(command, check=True, cwd=ROOT)
            continue

        assert optimizer_module is not None
        namespace = _namespace_for_target(config, target, artifact_path)
        optimizer_module.validate_args(namespace)
        optimizer_module.run_optimizer(namespace)


if __name__ == "__main__":
    main()
