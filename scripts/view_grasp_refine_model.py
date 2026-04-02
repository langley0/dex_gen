#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_refine.viewer import (
    add_marker,
    apply_hand_pose,
    build_view_model,
    configure_camera,
    load_viewer_state,
    set_bright_background,
)


OBJECT_POINT_COLOR = np.array([0.18, 0.64, 0.95, 0.85], dtype=float)
OBJECT_POINT_RADIUS = 0.0035
ROOT_MARKER_RADIUS = 0.007
SOURCE_ROOT_COLOR = np.array([0.98, 0.98, 0.98, 1.0], dtype=float)
GENERATED_ROOT_COLOR = np.array([0.18, 0.95, 0.35, 1.0], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View grasps generated from a trained grasp_refine checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Training checkpoint (.pkl).")
    parser.add_argument("--artifact", type=Path, required=True, help="Source grasp_gen artifact (.npz) used for object conditioning.")
    parser.add_argument("--state", choices=("best", "current"), default="best")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed for generated grasps.")
    parser.add_argument("--num-generated", type=int, default=1, help="How many generated grasps to sample from the model.")
    parser.add_argument("--pose", choices=("source", "generated", "cycle"), default="cycle")
    parser.add_argument("--generated-index", type=int, default=0, help="Used when --pose=generated.")
    parser.add_argument("--cycle-seconds", type=float, default=2.0)
    parser.add_argument("--device", type=str, default=None, help="Optional override for sampling device, for example cpu or gpu.")
    parser.add_argument("--object-num-points", type=int, default=256, help="Object points shown in the viewer and used for conditioning.")
    parser.add_argument("--bright-bg", action="store_true")
    parser.add_argument("--no-view", action="store_true", help="Load the checkpoint, sample grasps, print a summary, and exit.")
    return parser.parse_args()


def _print_summary(args: argparse.Namespace, state) -> None:
    print(f"checkpoint          : {state.checkpoint_path}")
    print(f"artifact            : {state.artifact_path}")
    print(f"object              : {state.object_name} ({state.object_kind})")
    print(f"hand side           : {state.hand_side}")
    print(f"source joint loss   : {state.source_joint_limit_loss:.6f}")
    print(f"source root loss    : {state.source_root_distance_loss:.6f}")
    print(f"generated samples   : {state.generated_poses.shape[0]}")
    for index in range(state.generated_poses.shape[0]):
        print(
            f"generated[{index}]      : "
            f"joint={state.generated_joint_limit_loss[index]:.6f} "
            f"root={state.generated_root_distance_loss[index]:.6f}"
        )
    print(f"pose mode           : {args.pose}")


def _pose_sequence(state, pose_mode: str, generated_index: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
    sequence: list[tuple[str, np.ndarray, np.ndarray]] = []
    if pose_mode in {"source", "cycle"}:
        sequence.append(("source", state.source_pose, SOURCE_ROOT_COLOR))
    if pose_mode == "generated":
        index = int(generated_index) % state.generated_poses.shape[0]
        sequence.append((f"generated[{index}]", state.generated_poses[index], GENERATED_ROOT_COLOR))
    elif pose_mode == "cycle":
        for index in range(state.generated_poses.shape[0]):
            sequence.append((f"generated[{index}]", state.generated_poses[index], GENERATED_ROOT_COLOR))
    return sequence


def main() -> None:
    args = parse_args()
    if args.num_generated <= 0:
        raise SystemExit("--num-generated must be positive.")
    if args.object_num_points <= 0:
        raise SystemExit("--object-num-points must be positive.")
    if args.cycle_seconds <= 0.0:
        raise SystemExit("--cycle-seconds must be positive.")

    state = load_viewer_state(
        args.checkpoint,
        args.artifact,
        state_name=args.state,
        sample_index=int(args.sample_index),
        seed=int(args.seed),
        num_generated=int(args.num_generated),
        object_num_points=int(args.object_num_points),
        device=args.device,
    )
    _print_summary(args, state)
    if args.no_view:
        return

    model, data, root_body_id = build_view_model(state.hand_side)
    if args.bright_bg:
        set_bright_background(model)

    sequence = _pose_sequence(state, args.pose, args.generated_index)
    point_stride = max(1, state.object_points.shape[0] // 200)
    shown_points = state.object_points[::point_stride]

    current_index = 0
    last_switch = time.monotonic()
    active_name, active_pose, active_color = sequence[current_index]
    apply_hand_pose(model, data, root_body_id, active_pose)
    print(f"active pose         : {active_name}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        configure_camera(viewer.cam)
        while viewer.is_running():
            now = time.monotonic()
            if len(sequence) > 1 and args.pose == "cycle" and now - last_switch >= args.cycle_seconds:
                current_index = (current_index + 1) % len(sequence)
                active_name, active_pose, active_color = sequence[current_index]
                apply_hand_pose(model, data, root_body_id, active_pose)
                print(f"active pose         : {active_name}")
                last_switch = now

            scene = viewer.user_scn
            idx = 0
            for point in shown_points:
                idx = add_marker(scene, idx, point, OBJECT_POINT_RADIUS, OBJECT_POINT_COLOR)
            idx = add_marker(scene, idx, active_pose[:3], ROOT_MARKER_RADIUS, active_color)
            scene.ngeom = idx
            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
