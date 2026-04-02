#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_refine import first_batch, load_dga_hand_point_spec
from grasp_refine.hand_points import full_pose_distance_points, full_pose_key_points, full_pose_surface_points
from grasp_refine.losses import erf_loss, spf_loss, srf_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved grasp_refine sample outputs with range and physics-style metrics.")
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared normalized DGA dataset (.npz).")
    parser.add_argument("--sample", type=Path, required=True, help="Saved sample npz from run_grasp_refine_sample.py.")
    parser.add_argument("--spf-threshold", type=float, default=0.02)
    parser.add_argument("--srf-threshold", type=float, default=0.02)
    return parser.parse_args()


def _overshoot(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[int, int, float, float]:
    below = int((values < lower).sum())
    above = int((values > upper).sum())
    max_above = float(np.maximum(values - upper, 0.0).max())
    max_below = float(np.maximum(lower - values, 0.0).max())
    return below, above, max_above, max_below


def main() -> None:
    args = parse_args()
    dataset, batch = first_batch(args.dataset, batch_size=1)
    sample_payload = np.load(args.sample.expanduser().resolve(), allow_pickle=False)
    samples = np.asarray(sample_payload["samples"], dtype=np.float32)
    samples_full = np.asarray(sample_payload["samples_full"], dtype=np.float32)
    train_pose = np.asarray(dataset.arrays.pose_raw, dtype=np.float32)

    batch_size, sample_count, pose_dim = samples.shape
    if samples_full.shape[:2] != samples.shape[:2]:
        raise SystemExit("samples and samples_full must share the same leading dimensions.")
    if batch_size <= 0 or sample_count <= 0:
        raise SystemExit("sample payload must contain at least one sample.")

    _, expected_batch = first_batch(args.dataset, batch_size=batch_size)
    flat_pose = samples.reshape(-1, pose_dim)
    flat_full = samples_full.reshape(-1, samples_full.shape[-1])

    trans = flat_pose[:, :3]
    joints = flat_pose[:, 3:]
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

    hand_side = str(expected_batch.hand_side[0])
    hand_spec = load_dga_hand_point_spec(hand_side)
    object_points = np.repeat(np.asarray(expected_batch.object_points, dtype=np.float32), sample_count, axis=0)
    object_normals = np.repeat(np.asarray(expected_batch.object_normals, dtype=np.float32), sample_count, axis=0)
    object_pcd_nor = jnp.concatenate(
        [jnp.asarray(object_points, dtype=jnp.float32), jnp.asarray(object_normals, dtype=jnp.float32)],
        axis=-1,
    )
    full_pose_jax = jnp.asarray(flat_full, dtype=jnp.float32)
    hand_surface = full_pose_surface_points(hand_spec, full_pose_jax)
    hand_distance = full_pose_distance_points(hand_spec, full_pose_jax)
    hand_key = full_pose_key_points(hand_spec, full_pose_jax)
    erf_value = float(erf_loss(object_pcd_nor, hand_surface))
    spf_value = float(spf_loss(hand_distance, jnp.asarray(object_points, dtype=jnp.float32), threshold=float(args.spf_threshold)))
    srf_value = float(srf_loss(hand_key, threshold=float(args.srf_threshold)))

    print(f"dataset path          : {dataset.path}")
    print(f"sample path           : {args.sample.expanduser().resolve()}")
    print(f"batch size            : {batch_size}")
    print(f"sample count          : {sample_count}")
    print(f"finite                : {bool(np.isfinite(samples).all() and np.isfinite(samples_full).all())}")
    print(f"translation below     : {trans_below}")
    print(f"translation above     : {trans_above}")
    print(f"joint below           : {joint_below}")
    print(f"joint above           : {joint_above}")
    print(f"max trans above       : {max_trans_above:.8f}")
    print(f"max trans below       : {max_trans_below:.8f}")
    print(f"max joint above       : {max_joint_above:.8f}")
    print(f"max joint below       : {max_joint_below:.8f}")
    print(f"sample mean distance  : {float(np.linalg.norm(flat_pose.mean(axis=0) - train_pose.mean(axis=0))):.8f}")
    print(f"sample std distance   : {float(np.linalg.norm(flat_pose.std(axis=0) - train_pose.std(axis=0))):.8f}")
    print(f"erf                   : {erf_value:.8f}")
    print(f"spf                   : {spf_value:.8f}")
    print(f"srf                   : {srf_value:.8f}")
    print(f"objective total       : {erf_value + spf_value + srf_value:.8f}")


if __name__ == "__main__":
    main()
