#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_refine.object_mesh import load_object_mesh, sample_mesh_points
from grasp_refine.viewer import (
    add_marker,
    apply_hand_pose,
    build_view_model,
    configure_camera,
    set_bright_background,
)


OBJECT_POINT_COLOR = np.array([0.18, 0.64, 0.95, 0.85], dtype=float)
OBJECT_POINT_RADIUS = 0.0035
ROOT_MARKER_RADIUS = 0.007
GENERATED_ROOT_COLOR = np.array([0.18, 0.95, 0.35, 1.0], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View object-conditioned grasps sampled by run_grasp_refine_sample.py.")
    parser.add_argument("--result", type=Path, required=True, help="Sample result directory or res_diffuser.pkl path.")
    parser.add_argument("--object", type=str, default=None, help="Object key to view. Defaults to the first object.")
    parser.add_argument("--generated-index", type=int, default=0, help="Used when --pose=generated.")
    parser.add_argument("--pose", choices=("generated", "cycle"), default="cycle")
    parser.add_argument("--cycle-seconds", type=float, default=2.0)
    parser.add_argument("--object-num-points", type=int, default=256, help="Object points shown in the viewer.")
    parser.add_argument("--object-point-seed", type=int, default=13)
    parser.add_argument("--bright-bg", action="store_true")
    parser.add_argument("--list-objects", action="store_true")
    parser.add_argument("--no-view", action="store_true", help="Print a summary and exit.")
    return parser.parse_args()


def _stable_kind_seed(object_kind: str, base_seed: int) -> int:
    stable_hash = sum((index + 1) * ord(char) for index, char in enumerate(object_kind))
    return int(base_seed) + stable_hash % 100_000


def _resolve_result_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "res_diffuser.pkl"
    if not resolved.exists():
        raise SystemExit(f"Result file was not found: {resolved}")
    return resolved


def _load_result(path: Path) -> dict:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise SystemExit(f"Unexpected result payload type: {type(payload)!r}")
    required = {"sample_qpos", "object_metadata", "hand_side"}
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise SystemExit(f"Result payload is missing required keys: {', '.join(missing)}")
    return payload


def _available_object_keys(payload: dict) -> list[str]:
    sample_qpos = payload.get("sample_qpos", {})
    if not isinstance(sample_qpos, dict) or not sample_qpos:
        raise SystemExit("No sampled objects were found in result payload.")
    return sorted(str(key) for key in sample_qpos.keys())


def _select_object_key(payload: dict, requested: str | None) -> str:
    keys = _available_object_keys(payload)
    if requested is None:
        return keys[0]
    if requested in keys:
        return requested
    raise SystemExit(f"Unknown --object {requested!r}. Available objects: {', '.join(keys)}")


def _sample_object_points(meta: dict[str, object], *, num_points: int, seed: int) -> np.ndarray:
    mesh = load_object_mesh(meta)
    object_kind = str(meta.get("kind", "unknown"))
    sampled_seed = _stable_kind_seed(object_kind, seed)
    points, _ = sample_mesh_points(mesh, int(num_points), seed=sampled_seed)
    return np.asarray(points, dtype=np.float32)


def _print_summary(payload: dict, object_key: str, generated_poses: np.ndarray) -> None:
    object_name = payload.get("object_name", {}).get(object_key, object_key)
    object_kind = payload.get("object_kind", {}).get(object_key, "unknown")
    artifact_path = payload.get("artifact_path", {}).get(object_key, "(unknown)")
    joint_losses = np.asarray(payload.get("joint_limit_loss", {}).get(object_key, []), dtype=np.float32)
    root_losses = np.asarray(payload.get("root_distance_loss", {}).get(object_key, []), dtype=np.float32)

    print(f"result              : {payload.get('checkpoint', '(unknown checkpoint)')}")
    print(f"object key          : {object_key}")
    print(f"object              : {object_name} ({object_kind})")
    print(f"artifact            : {artifact_path}")
    print(f"hand side           : {payload.get('hand_side', 'right')}")
    print(f"generated samples   : {generated_poses.shape[0]}")
    for index in range(generated_poses.shape[0]):
        joint_text = f"{float(joint_losses[index]):.6f}" if index < joint_losses.shape[0] else "n/a"
        root_text = f"{float(root_losses[index]):.6f}" if index < root_losses.shape[0] else "n/a"
        print(f"generated[{index}]      : joint={joint_text} root={root_text}")


def _pose_sequence(generated_poses: np.ndarray, pose_mode: str, generated_index: int) -> list[tuple[str, np.ndarray]]:
    if pose_mode == "generated":
        index = int(generated_index) % generated_poses.shape[0]
        return [(f"generated[{index}]", generated_poses[index])]
    return [(f"generated[{index}]", generated_poses[index]) for index in range(generated_poses.shape[0])]


def main() -> None:
    args = parse_args()
    if args.object_num_points <= 0:
        raise SystemExit("--object-num-points must be positive.")
    if args.cycle_seconds <= 0.0:
        raise SystemExit("--cycle-seconds must be positive.")

    result_path = _resolve_result_path(args.result)
    payload = _load_result(result_path)
    if args.list_objects:
        for key in _available_object_keys(payload):
            print(key)
        return

    object_key = _select_object_key(payload, args.object)
    generated_poses = np.asarray(payload["sample_qpos"][object_key], dtype=np.float32)
    if generated_poses.ndim != 2 or generated_poses.shape[0] == 0:
        raise SystemExit(f"No generated poses were found for {object_key!r}.")
    object_meta = dict(payload["object_metadata"][object_key])
    object_points = _sample_object_points(
        object_meta,
        num_points=int(args.object_num_points),
        seed=int(args.object_point_seed),
    )

    _print_summary(payload, object_key, generated_poses)
    if args.no_view:
        return

    hand_side = str(payload.get("hand_side", "right"))
    model, data, root_body_id = build_view_model(hand_side)
    if args.bright_bg:
        set_bright_background(model)

    sequence = _pose_sequence(generated_poses, args.pose, args.generated_index)
    point_stride = max(1, object_points.shape[0] // 200)
    shown_points = object_points[::point_stride]

    current_index = 0
    last_switch = time.monotonic()
    active_name, active_pose = sequence[current_index]
    apply_hand_pose(model, data, root_body_id, active_pose)
    print(f"active pose         : {active_name}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_camera(viewer.cam)
        while viewer.is_running():
            now = time.monotonic()
            if len(sequence) > 1 and args.pose == "cycle" and now - last_switch >= args.cycle_seconds:
                current_index = (current_index + 1) % len(sequence)
                active_name, active_pose = sequence[current_index]
                apply_hand_pose(model, data, root_body_id, active_pose)
                print(f"active pose         : {active_name}")
                last_switch = now

            scene = viewer.user_scn
            idx = 0
            for point in shown_points:
                idx = add_marker(scene, idx, point, OBJECT_POINT_RADIUS, OBJECT_POINT_COLOR)
            idx = add_marker(scene, idx, active_pose[:3], ROOT_MARKER_RADIUS, GENERATED_ROOT_COLOR)
            scene.ngeom = idx
            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
