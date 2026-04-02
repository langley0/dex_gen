#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_refine import iterate_dga_batches, load_saved_dga_dataset, split_dga_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the saved DGA dataset loader and batch iterator.")
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared normalized DGA dataset (.npz).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-mode", choices=("sample", "object", "object_random", "object_fixed"), default="object")
    parser.add_argument("--train-object-key", action="append", default=None, help="Repeatable object key for object_fixed split.")
    parser.add_argument("--val-object-key", action="append", default=None, help="Repeatable object key for object_fixed split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_saved_dga_dataset(args.dataset)
    train_subset, val_subset = split_dga_dataset(
        dataset,
        train_fraction=float(args.train_fraction),
        seed=int(args.seed),
        split_mode=args.split_mode,
        train_object_keys=args.train_object_key,
        val_object_keys=args.val_object_key,
    )

    train_batch = next(
        iterate_dga_batches(
            train_subset,
            batch_size=int(args.batch_size),
            shuffle=True,
            seed=int(args.seed),
        )
    )
    restored_pose = dataset.normalizer.denormalize_numpy(train_batch.pose)
    roundtrip_max_abs_error = float(np.max(np.abs(restored_pose - train_batch.pose_raw)))

    print(f"dataset path             : {dataset.path}")
    print(f"dataset size             : {len(dataset)}")
    print(f"train size               : {len(train_subset)}")
    print(f"val size                 : {0 if val_subset is None else len(val_subset)}")
    print(f"split mode               : {args.split_mode}")
    print(f"batch pose shape         : {train_batch.pose.shape}")
    print(f"batch object shape       : {train_batch.object_points.shape}")
    print(f"batch normal shape       : {train_batch.object_normals.shape}")
    print(f"batch contact shape      : {train_batch.contact_indices.shape}")
    print(f"normalized pose range    : [{float(train_batch.pose.min()):.8f}, {float(train_batch.pose.max()):.8f}]")
    print(f"roundtrip max abs error  : {roundtrip_max_abs_error:.8f}")
    print(f"unique train object kinds: {sorted(set(str(value) for value in train_batch.object_kind))}")
    print(f"unique train object keys : {sorted(set(str(value) for value in train_batch.object_key))}")


if __name__ == "__main__":
    main()
