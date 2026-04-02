#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_refine import load_saved_dga_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit grasp_refine sample validity against prepared dataset ranges.")
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared normalized DGA dataset (.npz).")
    parser.add_argument("--sample", type=Path, required=True, help="Saved sample npz from run_grasp_refine_sample.py.")
    return parser.parse_args()


def _overshoot(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[int, int, float, float]:
    below = int((values < lower).sum())
    above = int((values > upper).sum())
    max_above = float(np.maximum(values - upper, 0.0).max())
    max_below = float(np.maximum(lower - values, 0.0).max())
    return below, above, max_above, max_below


def main() -> None:
    args = parse_args()
    dataset = load_saved_dga_dataset(args.dataset)
    sample_payload = np.load(args.sample.expanduser().resolve(), allow_pickle=False)
    samples = np.asarray(sample_payload["samples"], dtype=np.float32).reshape(-1, dataset.normalizer.pose_dim)
    train_pose = np.asarray(dataset.arrays.pose_raw, dtype=np.float32)

    trans = samples[:, :3]
    joints = samples[:, 3:]
    train_trans = train_pose[:, :3]
    train_joints = train_pose[:, 3:]

    trans_below, trans_above, max_trans_above, max_trans_below = _overshoot(
        trans,
        dataset.normalizer.translation_lower,
        dataset.normalizer.translation_upper,
    )
    joint_below, joint_above, max_joint_above, max_joint_below = _overshoot(
        joints,
        dataset.normalizer.joint_lower,
        dataset.normalizer.joint_upper,
    )

    print(f"dataset path          : {dataset.path}")
    print(f"sample path           : {args.sample.expanduser().resolve()}")
    print(f"sample count          : {samples.shape[0]}")
    print(f"finite                : {bool(np.isfinite(samples).all())}")
    print(f"translation below     : {trans_below}")
    print(f"translation above     : {trans_above}")
    print(f"joint below           : {joint_below}")
    print(f"joint above           : {joint_above}")
    print(f"max trans above       : {max_trans_above:.8f}")
    print(f"max trans below       : {max_trans_below:.8f}")
    print(f"max joint above       : {max_joint_above:.8f}")
    print(f"max joint below       : {max_joint_below:.8f}")
    print(f"sample mean distance  : {float(np.linalg.norm(samples.mean(axis=0) - train_pose.mean(axis=0))):.8f}")
    print(f"sample std distance   : {float(np.linalg.norm(samples.std(axis=0) - train_pose.std(axis=0))):.8f}")
    print(f"status                : {'PASS' if (trans_below + trans_above + joint_below + joint_above) == 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
