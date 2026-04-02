from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import jax


def merge_param_tree(base: Any, updates: Any) -> Any:
    if isinstance(base, dict) and isinstance(updates, dict):
        merged = dict(base)
        for key, value in updates.items():
            if key in merged:
                merged[key] = merge_param_tree(merged[key], value)
            else:
                merged[key] = value
        return merged
    if isinstance(base, tuple) and isinstance(updates, tuple) and len(base) == len(updates):
        return tuple(merge_param_tree(base_item, update_item) for base_item, update_item in zip(base, updates, strict=True))
    return updates


def _extract_scene_model_tree(payload: Any) -> dict[str, object]:
    if isinstance(payload, dict) and "scene_model" in payload and isinstance(payload["scene_model"], dict):
        return payload["scene_model"]
    if (
        isinstance(payload, dict)
        and "params" in payload
        and isinstance(payload["params"], dict)
        and "scene_model" in payload["params"]
        and isinstance(payload["params"]["scene_model"], dict)
    ):
        return payload["params"]["scene_model"]
    if isinstance(payload, dict):
        return payload
    raise ValueError("Unsupported scene encoder pretrained payload. Expected a scene_model tree or checkpoint-like dict.")


def save_scene_encoder_pretrained_params(path: str | Path, scene_model_params: dict[str, object]) -> Path:
    resolved_path = Path(path).expanduser().resolve()
    if resolved_path.suffix.lower() not in {".pkl", ".pickle"}:
        raise ValueError(f"Scene encoder pretrained weights must be saved as .pkl, got {resolved_path.suffix!r}.")
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scene_model": jax.device_get(scene_model_params),
    }
    with resolved_path.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return resolved_path


def load_scene_encoder_pretrained_params(path: str | Path, template_params: dict[str, object]) -> dict[str, object]:
    resolved_path = Path(path).expanduser().resolve()
    if resolved_path.suffix.lower() not in {".pkl", ".pickle"}:
        raise ValueError(
            f"Scene encoder pretrained weights must be loaded from .pkl. "
            f"Use a pkl scene_model export or a grasp_refine checkpoint, got {resolved_path.suffix!r}."
        )
    with resolved_path.open("rb") as stream:
        payload = pickle.load(stream)
    loaded_tree = _extract_scene_model_tree(payload)
    return merge_param_tree(template_params, loaded_tree)
