from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .normalization import PoseNormalizer
from .types import TrainingConfig


def save_training_checkpoint(
    path: str | Path,
    *,
    params,
    optimizer_state,
    normalizer: PoseNormalizer,
    config: TrainingConfig,
    epoch: int,
    step: int,
    best_val_loss: float | None,
) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": params,
        "optimizer_state": optimizer_state,
        "normalizer": normalizer.state_dict(),
        "config": asdict(config),
        "epoch": int(epoch),
        "step": int(step),
        "best_val_loss": None if best_val_loss is None else float(best_val_loss),
    }
    with output_path.open("wb") as stream:
        pickle.dump(payload, stream)
    return output_path


def load_training_checkpoint(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("rb") as stream:
        payload = pickle.load(stream)
    payload["normalizer"] = PoseNormalizer.from_state_dict(payload["normalizer"])
    return payload
