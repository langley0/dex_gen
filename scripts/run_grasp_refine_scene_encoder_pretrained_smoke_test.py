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
from grasp_refine.scene_encoder_pretrained import load_scene_encoder_pretrained_params, save_scene_encoder_pretrained_params


def _max_abs_diff(a, b) -> float:
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    return max(float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in zip(leaves_a, leaves_b, strict=True))


def main() -> None:
    template = init_scene_encoder_params(jax.random.key(0), point_feature_dim=6, context_dim=512)
    with tempfile.TemporaryDirectory(prefix="grasp_refine_scene_pretrained_") as tmp_dir:
        export_path = Path(tmp_dir) / "scene_model.pkl"
        save_scene_encoder_pretrained_params(export_path, template)
        loaded = load_scene_encoder_pretrained_params(export_path, init_scene_encoder_params(jax.random.key(1), point_feature_dim=6, context_dim=512))

        checkpoint_like_path = Path(tmp_dir) / "checkpoint_like.pkl"
        with checkpoint_like_path.open("wb") as stream:
            pickle.dump({"params": {"scene_model": template}}, stream, protocol=pickle.HIGHEST_PROTOCOL)
        loaded_from_checkpoint = load_scene_encoder_pretrained_params(
            checkpoint_like_path,
            init_scene_encoder_params(jax.random.key(2), point_feature_dim=6, context_dim=512),
        )
    diff_export = _max_abs_diff(template, loaded)
    diff_checkpoint = _max_abs_diff(template, loaded_from_checkpoint)
    passed = diff_export < 1.0e-6 and diff_checkpoint < 1.0e-6
    print("pretrained format    : pkl")
    print(f"export roundtrip     : {diff_export:.8f}")
    print(f"checkpoint roundtrip : {diff_checkpoint:.8f}")
    print(f"status               : {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
