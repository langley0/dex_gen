#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from generate_grasp_candidates import (
    build_hand_only_scene,
    describe_hand_actuators,
    sample_initial_pose_sequence,
)
from generate_grasp_data import DEFAULT_CONFIG_PATH, load_optimizer_config
from view_franka_inspire import PICK_CYLINDER_POS
from view_grasp_candidate_env import _configure_candidate_camera, save_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the optimizer's sampled initial hand pose before any gradient updates."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--sample-index", type=int, default=0, help="0-based initial pose sample index.")
    parser.add_argument("--snapshot", type=Path, default=None, help="Optional PNG output path.")
    parser.add_argument(
        "--cycle",
        action="store_true",
        help="Cycle through all sampled initial poses in the viewer.",
    )
    parser.add_argument(
        "--cycle-interval",
        type=float,
        default=1.2,
        help="Seconds to hold each initial pose while cycling.",
    )
    parser.add_argument(
        "--print-joints",
        action="store_true",
        help="Print sampled actuator targets for the chosen initial pose.",
    )
    return parser.parse_args()


def _load_pose_samples(config_path: Path):
    config = load_optimizer_config(config_path)
    pose_samples = sample_initial_pose_sequence(config)
    if not pose_samples:
        raise ValueError("No initial poses were sampled.")
    _, hand_reference_model, _ = build_hand_only_scene(config, pose_samples[0])
    actuator_specs = describe_hand_actuators(hand_reference_model)
    return config, actuator_specs, pose_samples


def _add_marker_body(
    spec: mujoco.MjSpec,
    name: str,
    rgba: np.ndarray,
) -> None:
    body = spec.worldbody.add_body()
    body.name = name
    geom = body.add_geom()
    geom.name = f"{name}_geom"
    geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
    geom.size = np.array([0.008, 0.0, 0.0], dtype=float)
    geom.contype = 0
    geom.conaffinity = 0
    geom.rgba = rgba.astype(float)


def _apply_pose_sample_to_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    pose_sample,
) -> None:
    root_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"inspire_{side}_hand_base"))
    model.body_pos[root_body_id] = pose_sample.root_pos.astype(float)
    model.body_quat[root_body_id] = pose_sample.root_quat.astype(float)
    data.qpos[:] = pose_sample.hand_qpos.astype(float)
    data.qvel[:] = 0.0
    if model.nu > 0:
        data.ctrl[:] = np.clip(pose_sample.hand_ctrl.astype(float), model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])

    for marker_name, marker_pos in (
        ("anchor_marker", pose_sample.anchor_world_pos),
        ("surface_marker", pose_sample.surface_world_pos),
        ("palm_marker", pose_sample.palm_world_pos),
    ):
        marker_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, marker_name))
        if marker_body_id >= 0:
            model.body_pos[marker_body_id] = np.asarray(marker_pos, dtype=float)

    mujoco.mj_forward(model, data)


def _build_pose_view_model(config, pose_sample):
    spec, _, _ = build_hand_only_scene(config, pose_sample)
    _add_marker_body(spec, "anchor_marker", np.array([1.0, 0.2, 0.2, 1.0], dtype=float))
    _add_marker_body(spec, "surface_marker", np.array([0.2, 0.7, 1.0, 1.0], dtype=float))
    _add_marker_body(spec, "palm_marker", np.array([0.2, 1.0, 0.4, 1.0], dtype=float))
    model = spec.compile()
    data = mujoco.MjData(model)
    _apply_pose_sample_to_scene(model, data, config.scene.hand, pose_sample)
    return model, data


def _print_pose_summary(config_path: Path, config, pose_sample, actuator_specs, print_joints: bool) -> None:
    print(f"Config path         : {config_path}")
    print(f"Hand side           : {config.scene.hand}")
    print(f"Sample index        : {pose_sample.sample_index}")
    print(f"Orientation score   : {pose_sample.orientation_score:.6f}")
    print(f"Palm roll           : {pose_sample.palm_roll_deg:.3f} deg")
    print(
        f"Anchor world pos    : [{pose_sample.anchor_world_pos[0]:.6f}, {pose_sample.anchor_world_pos[1]:.6f}, {pose_sample.anchor_world_pos[2]:.6f}]"
    )
    print(
        f"Surface world pos   : [{pose_sample.surface_world_pos[0]:.6f}, {pose_sample.surface_world_pos[1]:.6f}, {pose_sample.surface_world_pos[2]:.6f}]"
    )
    print(
        f"Surface normal      : [{pose_sample.surface_world_normal[0]:.6f}, {pose_sample.surface_world_normal[1]:.6f}, {pose_sample.surface_world_normal[2]:.6f}]"
    )
    print(
        f"Palm world pos      : [{pose_sample.palm_world_pos[0]:.6f}, {pose_sample.palm_world_pos[1]:.6f}, {pose_sample.palm_world_pos[2]:.6f}]"
    )
    print(
        f"Palm world normal   : [{pose_sample.palm_world_normal[0]:.6f}, {pose_sample.palm_world_normal[1]:.6f}, {pose_sample.palm_world_normal[2]:.6f}]"
    )
    print(
        f"Hand root pos       : [{pose_sample.root_pos[0]:.6f}, {pose_sample.root_pos[1]:.6f}, {pose_sample.root_pos[2]:.6f}]"
    )
    print(
        f"Hand root quat      : [{pose_sample.root_quat[0]:.6f}, {pose_sample.root_quat[1]:.6f}, {pose_sample.root_quat[2]:.6f}, {pose_sample.root_quat[3]:.6f}]"
    )

    if print_joints:
        print("Initial joint targets")
        for spec in actuator_specs:
            print(
                f"{spec.actuator_index:02d}. {spec.joint_name} "
                f"ctrl={pose_sample.hand_ctrl[spec.actuator_index]:.6f} "
                f"range=[{spec.ctrl_min:.3f}, {spec.ctrl_max:.3f}]"
            )


def _run_cycle_viewer(model: mujoco.MjModel, data: mujoco.MjData, side: str, pose_samples: list, cycle_interval: float) -> None:
    object_center = PICK_CYLINDER_POS.copy()
    interval = max(cycle_interval, 0.1)
    pose_count = len(pose_samples)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        _configure_candidate_camera(viewer.cam, object_center)
        start_time = time.monotonic()
        current_index = -1
        while viewer.is_running():
            next_index = int((time.monotonic() - start_time) / interval) % pose_count
            if next_index != current_index:
                current_index = next_index
                _apply_pose_sample_to_scene(model, data, side, pose_samples[current_index])
                print(
                    f"[cycle] sample={pose_samples[current_index].sample_index:02d} "
                    f"score={pose_samples[current_index].orientation_score:.6f} "
                    f"roll={pose_samples[current_index].palm_roll_deg:.1f}deg"
                )
            viewer.sync()


def main() -> None:
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else DEFAULT_CONFIG_PATH.parent.parent / args.config
    config, actuator_specs, pose_samples = _load_pose_samples(config_path)
    if args.sample_index < 0 or args.sample_index >= len(pose_samples):
        raise ValueError(f"sample-index must be in [0, {len(pose_samples) - 1}]")
    pose_sample = pose_samples[args.sample_index]
    _print_pose_summary(config_path, config, pose_sample, actuator_specs, print_joints=args.print_joints)

    model, data = _build_pose_view_model(config, pose_sample)
    object_center = PICK_CYLINDER_POS.copy()

    if args.snapshot is not None:
        snapshot_path = args.snapshot if args.snapshot.is_absolute() else config_path.parent.parent / args.snapshot
        save_snapshot(model, data, object_center, snapshot_path)
        print(f"Saved snapshot      : {snapshot_path}")
        return

    if args.cycle:
        _run_cycle_viewer(model, data, config.scene.hand, pose_samples, cycle_interval=args.cycle_interval)
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        _configure_candidate_camera(viewer.cam, object_center)
        while viewer.is_running():
            viewer.sync()


if __name__ == "__main__":
    main()
