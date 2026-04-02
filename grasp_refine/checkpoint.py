from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax


@dataclass(frozen=True)
class CheckpointState:
    epoch: int
    step: int
    params: Any
    optimizer_state: Any


def _checkpoint_path(directory: Path, *, epoch: int, save_separately: bool) -> Path:
    if save_separately:
        return directory / f"model_{epoch}.pkl"
    return directory / "model.pkl"


def save_checkpoint(
    directory: str | Path,
    *,
    epoch: int,
    step: int,
    params: Any,
    optimizer_state: Any,
    save_separately: bool = True,
    keep_last: int = 5,
    save_scene_model: bool = True,
) -> Path:
    ckpt_dir = Path(directory).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(ckpt_dir, epoch=epoch, save_separately=save_separately)
    params_to_save = jax.device_get(params)
    if not save_scene_model and isinstance(params_to_save, dict) and "scene_model" in params_to_save:
        params_to_save = dict(params_to_save)
        params_to_save.pop("scene_model")

    payload = {
        "epoch": int(epoch),
        "step": int(step),
        "params": params_to_save,
        "optimizer_state": jax.device_get(optimizer_state),
    }
    with path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)

    if save_separately:
        checkpoint_files = sorted(
            ckpt_dir.glob("model_*.pkl"),
            key=lambda item: int(item.stem.split("_")[1]),
        )
        while len(checkpoint_files) > max(int(keep_last), 1):
            checkpoint_files[0].unlink(missing_ok=True)
            checkpoint_files.pop(0)
    return path


def load_checkpoint(
    directory: str | Path,
    *,
    save_separately: bool = True,
) -> CheckpointState | None:
    ckpt_dir = Path(directory).expanduser().resolve()
    if not ckpt_dir.exists():
        return None
    if save_separately:
        checkpoint_files = sorted(
            ckpt_dir.glob("model_*.pkl"),
            key=lambda item: int(item.stem.split("_")[1]),
        )
        if not checkpoint_files:
            return None
        path = checkpoint_files[-1]
    else:
        path = ckpt_dir / "model.pkl"
        if not path.exists():
            return None

    with path.open("rb") as stream:
        payload = pickle.load(stream)
    return CheckpointState(
        epoch=int(payload["epoch"]),
        step=int(payload["step"]),
        params=payload["params"],
        optimizer_state=payload["optimizer_state"],
    )
