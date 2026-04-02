#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import pickle

import jax
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_refine.scene_encoder_dga import init_scene_encoder_params
from grasp_refine.scene_encoder_pretrained import load_scene_encoder_pretrained_params


def _array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _export_fake_dga_state(scene_params: dict[str, object]) -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {}
    for stage_index, stage in enumerate(scene_params["stages"]):
        prefix = f"enc{stage_index + 1}"
        transition = stage["transition"]
        state[f"{prefix}.0.linear.weight"] = _array(transition["linear"]["w"]).T.copy()
        state[f"{prefix}.0.bn.weight"] = _array(transition["bn"]["scale"])
        state[f"{prefix}.0.bn.bias"] = _array(transition["bn"]["bias"])
        for block_index, block in enumerate(stage["blocks"]):
            block_prefix = f"{prefix}.{block_index + 1}"
            state[f"{block_prefix}.linear1.weight"] = _array(block["linear1"]["w"]).T.copy()
            state[f"{block_prefix}.bn1.weight"] = _array(block["bn1"]["scale"])
            state[f"{block_prefix}.bn1.bias"] = _array(block["bn1"]["bias"])
            transformer = block["transformer2"]
            state[f"{block_prefix}.transformer2.linear_q.weight"] = _array(transformer["linear_q"]["w"]).T.copy()
            state[f"{block_prefix}.transformer2.linear_q.bias"] = _array(transformer["linear_q"]["b"])
            state[f"{block_prefix}.transformer2.linear_k.weight"] = _array(transformer["linear_k"]["w"]).T.copy()
            state[f"{block_prefix}.transformer2.linear_k.bias"] = _array(transformer["linear_k"]["b"])
            state[f"{block_prefix}.transformer2.linear_v.weight"] = _array(transformer["linear_v"]["w"]).T.copy()
            state[f"{block_prefix}.transformer2.linear_v.bias"] = _array(transformer["linear_v"]["b"])
            state[f"{block_prefix}.transformer2.linear_p.0.weight"] = _array(transformer["linear_p1"]["w"]).T.copy()
            state[f"{block_prefix}.transformer2.linear_p.0.bias"] = _array(transformer["linear_p1"]["b"])
            state[f"{block_prefix}.transformer2.linear_p.1.weight"] = _array(transformer["bn_p"]["scale"])
            state[f"{block_prefix}.transformer2.linear_p.1.bias"] = _array(transformer["bn_p"]["bias"])
            state[f"{block_prefix}.transformer2.linear_p.3.weight"] = _array(transformer["linear_p2"]["w"]).T.copy()
            state[f"{block_prefix}.transformer2.linear_p.3.bias"] = _array(transformer["linear_p2"]["b"])
            state[f"{block_prefix}.transformer2.linear_w.0.weight"] = _array(transformer["bn_w1"]["scale"])
            state[f"{block_prefix}.transformer2.linear_w.0.bias"] = _array(transformer["bn_w1"]["bias"])
            state[f"{block_prefix}.transformer2.linear_w.2.weight"] = _array(transformer["linear_w1"]["w"]).T.copy()
            state[f"{block_prefix}.transformer2.linear_w.2.bias"] = _array(transformer["linear_w1"]["b"])
            state[f"{block_prefix}.transformer2.linear_w.3.weight"] = _array(transformer["bn_w2"]["scale"])
            state[f"{block_prefix}.transformer2.linear_w.3.bias"] = _array(transformer["bn_w2"]["bias"])
            state[f"{block_prefix}.transformer2.linear_w.5.weight"] = _array(transformer["linear_w2"]["w"]).T.copy()
            state[f"{block_prefix}.transformer2.linear_w.5.bias"] = _array(transformer["linear_w2"]["b"])
            state[f"{block_prefix}.bn2.weight"] = _array(block["bn2"]["scale"])
            state[f"{block_prefix}.bn2.bias"] = _array(block["bn2"]["bias"])
            state[f"{block_prefix}.linear3.weight"] = _array(block["linear3"]["w"]).T.copy()
            state[f"{block_prefix}.bn3.weight"] = _array(block["bn3"]["scale"])
            state[f"{block_prefix}.bn3.bias"] = _array(block["bn3"]["bias"])
    return state


def _max_abs_diff(a, b) -> float:
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    return max(float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in zip(leaves_a, leaves_b, strict=True))


def main() -> None:
    template = init_scene_encoder_params(jax.random.key(0), point_feature_dim=6, context_dim=512)
    state_dict = _export_fake_dga_state(template)
    with tempfile.TemporaryDirectory(prefix="grasp_refine_scene_pretrained_") as tmp_dir:
        ckpt_path = Path(tmp_dir) / "fake_pointtransformer.pth"
        with ckpt_path.open("wb") as stream:
            pickle.dump(state_dict, stream, protocol=pickle.HIGHEST_PROTOCOL)
        loaded = load_scene_encoder_pretrained_params(ckpt_path, init_scene_encoder_params(jax.random.key(1), point_feature_dim=6, context_dim=512))
    diff = _max_abs_diff(template, loaded)
    print(f"pretrained path      : synthetic torch state_dict")
    print(f"max abs diff         : {diff:.8f}")
    print(f"status               : {'PASS' if diff < 1.0e-6 else 'FAIL'}")


if __name__ == "__main__":
    main()
