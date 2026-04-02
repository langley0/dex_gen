from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


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


def _to_jax_weight(weight: Any, *, transpose: bool = True) -> jax.Array:
    array = np.asarray(weight, dtype=np.float32)
    if transpose and array.ndim == 2:
        array = array.T
    return jnp.asarray(array, dtype=jnp.float32)


def _to_jax_vector(value: Any) -> jax.Array:
    return jnp.asarray(np.asarray(value, dtype=np.float32), dtype=jnp.float32)


def _load_torch_state_dict(path: Path) -> dict[str, np.ndarray]:
    try:
        import torch
    except ModuleNotFoundError:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
    else:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        state_dict = payload["model"]
    else:
        state_dict = payload
    normalized: dict[str, np.ndarray] = {}
    for key, value in state_dict.items():
        if hasattr(value, "detach"):
            normalized[key] = value.detach().cpu().numpy()
        else:
            normalized[key] = np.asarray(value)
    return normalized


def _scene_substate_from_torch_state(state_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    prefixes = (
        "module.eps_model.scene_model.",
        "eps_model.scene_model.",
        "module.scene_model.",
        "scene_model.",
        "",
    )
    for prefix in prefixes:
        extracted = {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}
        if extracted and any(key.startswith("enc") for key in extracted):
            return extracted
    raise ValueError("Unable to find PointTransformer scene_model weights in checkpoint.")


def _load_pickled_tree(path: Path) -> Any:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if isinstance(payload, dict) and "scene_model" in payload:
        return payload["scene_model"]
    if isinstance(payload, dict) and "params" in payload and isinstance(payload["params"], dict) and "scene_model" in payload["params"]:
        return payload["params"]["scene_model"]
    return payload


def _load_npz_tree(path: Path) -> Any:
    payload = np.load(path, allow_pickle=True)
    if "scene_model" in payload:
        return payload["scene_model"].item()
    if "params" in payload:
        params = payload["params"].item()
        if isinstance(params, dict) and "scene_model" in params:
            return params["scene_model"]
    raise ValueError(f"Unable to find scene_model entry in {path}.")


def _map_transition(stage_index: int, transition: dict[str, object], state: dict[str, np.ndarray]) -> dict[str, object]:
    prefix = f"enc{stage_index + 1}.0"
    linear_weight = state[f"{prefix}.linear.weight"]
    bn_weight = state[f"{prefix}.bn.weight"]
    bn_bias = state[f"{prefix}.bn.bias"]
    updated = dict(transition)
    updated["linear"] = dict(updated["linear"])
    updated["linear"]["w"] = _to_jax_weight(linear_weight, transpose=True)
    updated["bn"] = dict(updated["bn"])
    updated["bn"]["scale"] = _to_jax_vector(bn_weight)
    updated["bn"]["bias"] = _to_jax_vector(bn_bias)
    return updated


def _map_block(stage_index: int, block_index: int, block: dict[str, object], state: dict[str, np.ndarray]) -> dict[str, object]:
    prefix = f"enc{stage_index + 1}.{block_index + 1}"
    updated = dict(block)
    updated["linear1"] = dict(updated["linear1"])
    updated["linear1"]["w"] = _to_jax_weight(state[f"{prefix}.linear1.weight"], transpose=True)
    updated["bn1"] = dict(updated["bn1"])
    updated["bn1"]["scale"] = _to_jax_vector(state[f"{prefix}.bn1.weight"])
    updated["bn1"]["bias"] = _to_jax_vector(state[f"{prefix}.bn1.bias"])
    transformer = dict(updated["transformer2"])
    transformer["linear_q"] = dict(transformer["linear_q"])
    transformer["linear_q"]["w"] = _to_jax_weight(state[f"{prefix}.transformer2.linear_q.weight"], transpose=True)
    transformer["linear_q"]["b"] = _to_jax_vector(state[f"{prefix}.transformer2.linear_q.bias"])
    transformer["linear_k"] = dict(transformer["linear_k"])
    transformer["linear_k"]["w"] = _to_jax_weight(state[f"{prefix}.transformer2.linear_k.weight"], transpose=True)
    transformer["linear_k"]["b"] = _to_jax_vector(state[f"{prefix}.transformer2.linear_k.bias"])
    transformer["linear_v"] = dict(transformer["linear_v"])
    transformer["linear_v"]["w"] = _to_jax_weight(state[f"{prefix}.transformer2.linear_v.weight"], transpose=True)
    transformer["linear_v"]["b"] = _to_jax_vector(state[f"{prefix}.transformer2.linear_v.bias"])
    transformer["linear_p1"] = dict(transformer["linear_p1"])
    transformer["linear_p1"]["w"] = _to_jax_weight(state[f"{prefix}.transformer2.linear_p.0.weight"], transpose=True)
    transformer["linear_p1"]["b"] = _to_jax_vector(state[f"{prefix}.transformer2.linear_p.0.bias"])
    transformer["bn_p"] = dict(transformer["bn_p"])
    transformer["bn_p"]["scale"] = _to_jax_vector(state[f"{prefix}.transformer2.linear_p.1.weight"])
    transformer["bn_p"]["bias"] = _to_jax_vector(state[f"{prefix}.transformer2.linear_p.1.bias"])
    transformer["linear_p2"] = dict(transformer["linear_p2"])
    transformer["linear_p2"]["w"] = _to_jax_weight(state[f"{prefix}.transformer2.linear_p.3.weight"], transpose=True)
    transformer["linear_p2"]["b"] = _to_jax_vector(state[f"{prefix}.transformer2.linear_p.3.bias"])
    transformer["bn_w1"] = dict(transformer["bn_w1"])
    transformer["bn_w1"]["scale"] = _to_jax_vector(state[f"{prefix}.transformer2.linear_w.0.weight"])
    transformer["bn_w1"]["bias"] = _to_jax_vector(state[f"{prefix}.transformer2.linear_w.0.bias"])
    transformer["linear_w1"] = dict(transformer["linear_w1"])
    transformer["linear_w1"]["w"] = _to_jax_weight(state[f"{prefix}.transformer2.linear_w.2.weight"], transpose=True)
    transformer["linear_w1"]["b"] = _to_jax_vector(state[f"{prefix}.transformer2.linear_w.2.bias"])
    transformer["bn_w2"] = dict(transformer["bn_w2"])
    transformer["bn_w2"]["scale"] = _to_jax_vector(state[f"{prefix}.transformer2.linear_w.3.weight"])
    transformer["bn_w2"]["bias"] = _to_jax_vector(state[f"{prefix}.transformer2.linear_w.3.bias"])
    transformer["linear_w2"] = dict(transformer["linear_w2"])
    transformer["linear_w2"]["w"] = _to_jax_weight(state[f"{prefix}.transformer2.linear_w.5.weight"], transpose=True)
    transformer["linear_w2"]["b"] = _to_jax_vector(state[f"{prefix}.transformer2.linear_w.5.bias"])
    updated["transformer2"] = transformer
    updated["bn2"] = dict(updated["bn2"])
    updated["bn2"]["scale"] = _to_jax_vector(state[f"{prefix}.bn2.weight"])
    updated["bn2"]["bias"] = _to_jax_vector(state[f"{prefix}.bn2.bias"])
    updated["linear3"] = dict(updated["linear3"])
    updated["linear3"]["w"] = _to_jax_weight(state[f"{prefix}.linear3.weight"], transpose=True)
    updated["bn3"] = dict(updated["bn3"])
    updated["bn3"]["scale"] = _to_jax_vector(state[f"{prefix}.bn3.weight"])
    updated["bn3"]["bias"] = _to_jax_vector(state[f"{prefix}.bn3.bias"])
    return updated


def load_scene_encoder_pretrained_params(path: str | Path, template_params: dict[str, object]) -> dict[str, object]:
    resolved_path = Path(path).expanduser().resolve()
    suffix = resolved_path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        loaded = _load_pickled_tree(resolved_path)
        if not isinstance(loaded, dict):
            raise ValueError(f"Unsupported pickled scene encoder payload in {resolved_path}.")
        return merge_param_tree(template_params, loaded)
    if suffix == ".npz":
        loaded = _load_npz_tree(resolved_path)
        if not isinstance(loaded, dict):
            raise ValueError(f"Unsupported npz scene encoder payload in {resolved_path}.")
        return merge_param_tree(template_params, loaded)
    if suffix in {".pth", ".pt"}:
        state = _scene_substate_from_torch_state(_load_torch_state_dict(resolved_path))
        updated = dict(template_params)
        stages = list(updated["stages"])
        for stage_index, stage in enumerate(stages):
            stage_updated = dict(stage)
            stage_updated["transition"] = _map_transition(stage_index, stage["transition"], state)
            stage_updated["blocks"] = tuple(
                _map_block(stage_index, block_index, block, state)
                for block_index, block in enumerate(stage["blocks"])
            )
            stages[stage_index] = stage_updated
        updated["stages"] = tuple(stages)
        return updated
    raise ValueError(f"Unsupported pretrained scene encoder format: {resolved_path.suffix}")
