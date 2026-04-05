from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ArtifactStateName = Literal["best", "current"]
CoordinateMode = Literal["hand_aligned_object", "world_object_rotated"]


@dataclass(frozen=True)
class DatasetConfig:
    artifact_paths: tuple[Path, ...] = ()
    artifact_glob: str | None = None
    state_name: ArtifactStateName = "best"
    max_samples_per_artifact: int | None = None
    object_num_points: int = 2048
    object_point_seed: int = 13
    coordinate_mode: CoordinateMode = "hand_aligned_object"
    drop_invalid_samples: bool = True
