#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_refine import build_saved_object_key, load_saved_dga_dataset, split_dga_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit DGA-style split semantics on a prepared dataset.")
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared normalized DGA dataset (.npz).")
    parser.add_argument("--output", type=Path, default=None, help="Optional synthetic multi-object dataset path.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _clone_with_object_keys(source: Path, output: Path) -> Path:
    output_path = output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(source.expanduser().resolve(), allow_pickle=False) as payload:
        saved = {name: np.asarray(payload[name]) for name in payload.files}

    base_count = int(np.asarray(saved["pose"]).shape[0])
    chunk_count = min(3, max(base_count, 1))
    rows = np.arange(base_count, dtype=np.int32)
    partitions = [rows[index::chunk_count] for index in range(chunk_count)]
    labels = [
        ("cylinder", "cylinder", "cylinder:r=0.045:hh=0.165"),
        ("cube", "cube", "cube:size=0.08"),
        ("drill", "drill", "mesh:assets/drill/meshes/material_collision.stl"),
    ]

    object_kind = [str(value) for value in np.asarray(saved["object_kind"], dtype=np.str_)]
    object_name = [str(value) for value in np.asarray(saved["object_name"], dtype=np.str_)]
    if "object_key" in saved:
        object_key = [str(value) for value in np.asarray(saved["object_key"], dtype=np.str_)]
    else:
        object_key = [
            build_saved_object_key(object_kind=str(kind), object_name=str(name))
            for kind, name in zip(object_kind, object_name, strict=False)
        ]

    for label_index, indices in enumerate(partitions):
        kind, name, updated_object_key = labels[label_index]
        for index in indices:
            object_kind[int(index)] = kind
            object_name[int(index)] = name
            object_key[int(index)] = updated_object_key

    saved["object_kind"] = np.asarray(object_kind, dtype=np.str_)
    saved["object_name"] = np.asarray(object_name, dtype=np.str_)
    saved["object_key"] = np.asarray(object_key, dtype=np.str_)

    metadata = json.loads(str(np.asarray(saved["metadata_json"]).item()))
    metadata["split_audit"] = {
        "synthetic_object_keys": [label[2] for label in labels[:chunk_count]],
    }
    saved["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_)
    np.savez_compressed(output_path, **saved)
    return output_path


def _subset_keys(dataset, subset) -> list[str]:
    if subset is None or len(subset) == 0:
        return []
    return sorted({str(dataset.arrays.object_key[int(index)]) for index in subset.indices})


def main() -> None:
    args = parse_args()
    synthetic_path = args.output
    if synthetic_path is None:
        synthetic_path = ROOT / "outputs" / "grasp_refine_dga_data" / "split_audit_synthetic_multi_object.npz"
    synthetic_path = _clone_with_object_keys(args.dataset, synthetic_path)

    dataset = load_saved_dga_dataset(synthetic_path)
    object_keys = sorted(set(str(value) for value in dataset.arrays.object_key))

    random_train, random_val = split_dga_dataset(
        dataset,
        train_fraction=2.0 / 3.0,
        seed=int(args.seed),
        split_mode="object_random",
    )
    fixed_train_keys = object_keys[:2]
    fixed_val_keys = object_keys[2:]
    fixed_train, fixed_val = split_dga_dataset(
        dataset,
        train_fraction=1.0,
        seed=int(args.seed),
        split_mode="object_fixed",
        train_object_keys=fixed_train_keys,
        val_object_keys=fixed_val_keys,
    )

    print(f"synthetic dataset path   : {synthetic_path}")
    print(f"dataset size             : {len(dataset)}")
    print(f"object keys             : {object_keys}")
    print(f"random train object keys : {_subset_keys(dataset, random_train)}")
    print(f"random val object keys   : {_subset_keys(dataset, random_val)}")
    print(f"fixed train object keys  : {_subset_keys(dataset, fixed_train)}")
    print(f"fixed val object keys    : {_subset_keys(dataset, fixed_val)}")

    overlap = set(_subset_keys(dataset, fixed_train)) & set(_subset_keys(dataset, fixed_val))
    if overlap:
        raise SystemExit(f"Fixed split overlap detected: {sorted(overlap)}")
    if set(_subset_keys(dataset, fixed_train)) != set(fixed_train_keys):
        raise SystemExit("Fixed train split keys do not match the requested allowlist.")
    if set(_subset_keys(dataset, fixed_val)) != set(fixed_val_keys):
        raise SystemExit("Fixed val split keys do not match the requested allowlist.")


if __name__ == "__main__":
    main()
