from __future__ import annotations

import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .artifact_subset import ArtifactSubsetResult, subset_grasp_artifact_topk


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PipelinePaths:
    output_root: Path
    optimizer_artifact: Path
    selected_artifact: Path
    refine_output_dir: Path
    train_config_path: Path
    viewer_commands_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class PipelineResult:
    paths: PipelinePaths
    subset: ArtifactSubsetResult
    optimizer_command: tuple[str, ...]
    refine_train_command: tuple[str, ...]
    grasp_view_command: tuple[str, ...]
    model_view_command: tuple[str, ...]


def default_pipeline_paths(*, object_name: str, envs: int, steps: int, top_k: int, seed: int) -> PipelinePaths:
    stem = f"{object_name}_e{envs}_s{steps}_top{top_k}_seed{seed}"
    output_root = ROOT / "outputs" / "grasp_pipeline" / stem
    return PipelinePaths(
        output_root=output_root,
        optimizer_artifact=output_root / "grasp_optimizer.npz",
        selected_artifact=output_root / f"grasp_optimizer_top{top_k}.npz",
        refine_output_dir=output_root / "grasp_refine_train",
        train_config_path=output_root / "grasp_refine_train_config.json",
        viewer_commands_path=output_root / "viewer_commands.sh",
        manifest_path=output_root / "pipeline_manifest.json",
    )


def _run_command(command: tuple[str, ...]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def format_command(command: tuple[str, ...]) -> str:
    return " ".join(shlex.quote(token) for token in command)


def build_optimizer_command(
    *,
    hand: str,
    object_name: str,
    envs: int,
    steps: int,
    seed: int,
    output_path: Path,
    backend: str,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_grasp_optimizer.py"),
        "--hand",
        hand,
        "--object",
        object_name,
        "--batch",
        str(envs),
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--output",
        str(output_path),
    ]
    if backend:
        command.extend(["--backend", backend])
    return tuple(command)


def build_refine_train_command(
    *,
    selected_artifact: Path,
    output_dir: Path,
    train_config_path: Path,
    device: str,
    preset: str,
    epochs: int | None,
    batch_size: int | None,
) -> tuple[str, ...]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_grasp_refine_train.py"),
        "--artifact",
        str(selected_artifact),
        "--device",
        device,
        "--output-dir",
        str(output_dir),
        "--write-config",
        str(train_config_path),
    ]
    if preset:
        command.extend(["--preset", preset])
    if epochs is not None:
        command.extend(["--epochs", str(epochs)])
    if batch_size is not None:
        command.extend(["--batch-size", str(batch_size)])
    return tuple(command)


def build_view_commands(
    *,
    selected_artifact: Path,
    refine_output_dir: Path,
    refine_device: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    grasp_view = (
        sys.executable,
        str(ROOT / "scripts" / "view_grasp_optimizer.py"),
        "--result",
        str(selected_artifact),
        "--state",
        "best",
        "--index",
        "0",
        "--bright-bg",
    )
    model_view = (
        sys.executable,
        str(ROOT / "scripts" / "view_grasp_refine_model.py"),
        "--checkpoint",
        str(refine_output_dir / "best.pkl"),
        "--artifact",
        str(selected_artifact),
        "--device",
        refine_device,
        "--num-generated",
        "3",
        "--pose",
        "cycle",
        "--object-num-points",
        "256",
        "--bright-bg",
    )
    return grasp_view, model_view


def write_viewer_commands(
    path: Path,
    *,
    grasp_view_command: tuple[str, ...],
    model_view_command: tuple[str, ...],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Viewer for the top-ranked grasp_gen sample used during refine training",
        format_command(grasp_view_command),
        "",
        "# Viewer for grasps sampled from the trained grasp_refine model",
        format_command(model_view_command),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_pipeline_manifest(path: Path, result: PipelineResult) -> Path:
    payload = {
        "paths": {
            "output_root": str(result.paths.output_root),
            "optimizer_artifact": str(result.paths.optimizer_artifact),
            "selected_artifact": str(result.paths.selected_artifact),
            "refine_output_dir": str(result.paths.refine_output_dir),
            "train_config_path": str(result.paths.train_config_path),
            "viewer_commands_path": str(result.paths.viewer_commands_path),
        },
        "subset": {
            "input_path": str(result.subset.input_path),
            "output_path": str(result.subset.output_path),
            "state_name": result.subset.state_name,
            "requested_top_k": result.subset.requested_top_k,
            "selected_count": result.subset.selected_count,
            "selected_indices": result.subset.selected_indices.tolist(),
            "selected_energies": result.subset.selected_energies.tolist(),
        },
        "commands": {
            "optimizer": format_command(result.optimizer_command),
            "refine_train": format_command(result.refine_train_command),
            "grasp_view": format_command(result.grasp_view_command),
            "model_view": format_command(result.model_view_command),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_grasp_gen_to_refine_pipeline(
    *,
    hand: str,
    object_name: str,
    envs: int,
    steps: int,
    top_k: int,
    seed: int,
    optimizer_backend: str,
    refine_device: str,
    refine_preset: str,
    refine_epochs: int | None,
    refine_batch_size: int | None,
    regenerate_optimizer: bool,
    output_root: Path | None,
) -> PipelineResult:
    if output_root is None:
        paths = default_pipeline_paths(object_name=object_name, envs=envs, steps=steps, top_k=top_k, seed=seed)
    else:
        root = Path(output_root).expanduser().resolve()
        paths = PipelinePaths(
            output_root=root,
            optimizer_artifact=root / "grasp_optimizer.npz",
            selected_artifact=root / f"grasp_optimizer_top{top_k}.npz",
            refine_output_dir=root / "grasp_refine_train",
            train_config_path=root / "grasp_refine_train_config.json",
            viewer_commands_path=root / "viewer_commands.sh",
            manifest_path=root / "pipeline_manifest.json",
        )
    paths.output_root.mkdir(parents=True, exist_ok=True)

    optimizer_command = build_optimizer_command(
        hand=hand,
        object_name=object_name,
        envs=envs,
        steps=steps,
        seed=seed,
        output_path=paths.optimizer_artifact,
        backend=optimizer_backend,
    )
    if regenerate_optimizer or not paths.optimizer_artifact.exists():
        _run_command(optimizer_command)

    subset = subset_grasp_artifact_topk(
        paths.optimizer_artifact,
        paths.selected_artifact,
        top_k=top_k,
        state_name="best",
    )

    refine_train_command = build_refine_train_command(
        selected_artifact=paths.selected_artifact,
        output_dir=paths.refine_output_dir,
        train_config_path=paths.train_config_path,
        device=refine_device,
        preset=refine_preset,
        epochs=refine_epochs,
        batch_size=refine_batch_size,
    )
    _run_command(refine_train_command)

    grasp_view_command, model_view_command = build_view_commands(
        selected_artifact=paths.selected_artifact,
        refine_output_dir=paths.refine_output_dir,
        refine_device=refine_device,
    )
    result = PipelineResult(
        paths=paths,
        subset=subset,
        optimizer_command=optimizer_command,
        refine_train_command=refine_train_command,
        grasp_view_command=grasp_view_command,
        model_view_command=model_view_command,
    )
    write_viewer_commands(
        paths.viewer_commands_path,
        grasp_view_command=grasp_view_command,
        model_view_command=model_view_command,
    )
    write_pipeline_manifest(paths.manifest_path, result)
    return result
