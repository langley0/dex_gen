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

from grasp_gen.hand import Hand
from grasp_gen.prop_assets import prop_from_metadata
from grasp_sampling.io import load_sampling_run
from grasp_sampling.scene import build_physics_scene
from grasp_sampling.types import MotionSpec


def _bright_bg(model: mujoco.MjModel) -> None:
    model.vis.rgba.haze[:] = np.array([0.97, 0.97, 0.99, 1.0], dtype=float)
    model.vis.rgba.fog[:] = np.array([0.97, 0.97, 0.99, 1.0], dtype=float)
    model.vis.headlight.ambient[:] = np.array([0.55, 0.55, 0.55], dtype=float)
    model.vis.headlight.diffuse[:] = np.array([0.85, 0.85, 0.85], dtype=float)
    model.vis.headlight.specular[:] = np.array([0.15, 0.15, 0.15], dtype=float)


def _cam(cam: mujoco.MjvCamera) -> None:
    cam.lookat[:] = np.array([0.0, 0.0, 0.0], dtype=float)
    cam.distance = 0.72
    cam.azimuth = 145.0
    cam.elevation = -18.0


def _unit_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float).reshape(4).copy()
    norm = np.linalg.norm(quat)
    if norm < 1.0e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    quat /= norm
    if quat[0] < 0.0:
        quat *= -1.0
    return quat


def _ortho6d_to_matrix_np(ortho6d: np.ndarray) -> np.ndarray:
    ortho6d = np.asarray(ortho6d, dtype=float).reshape(6)
    first = ortho6d[:3]
    first /= max(float(np.linalg.norm(first)), 1.0e-8)
    second = ortho6d[3:6] - first * float(np.dot(first, ortho6d[3:6]))
    second /= max(float(np.linalg.norm(second)), 1.0e-8)
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=1)


def _matrix_to_quat_np(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s,
            ],
            dtype=float,
        )
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        quat = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / s,
                0.25 * s,
                (rotation[0, 1] + rotation[1, 0]) / s,
                (rotation[0, 2] + rotation[2, 0]) / s,
            ],
            dtype=float,
        )
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        quat = np.array(
            [
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[0, 1] + rotation[1, 0]) / s,
                0.25 * s,
                (rotation[1, 2] + rotation[2, 1]) / s,
            ],
            dtype=float,
        )
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        quat = np.array(
            [
                (rotation[1, 0] - rotation[0, 1]) / s,
                (rotation[0, 2] + rotation[2, 0]) / s,
                (rotation[1, 2] + rotation[2, 1]) / s,
                0.25 * s,
            ],
            dtype=float,
        )
    return _unit_quat(quat)


def _quat_mul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = _unit_quat(lhs)
    w2, x2, y2, z2 = _unit_quat(rhs)
    return _unit_quat(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=float,
        )
    )


def _axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float).reshape(3)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-8)
    half = 0.5 * float(angle)
    return _unit_quat(np.array([np.cos(half), *(np.sin(half) * axis)], dtype=float))


def _motion_pose(base_root_pos: np.ndarray, base_root_quat: np.ndarray, motion: MotionSpec, alpha: float, translation_delta: float, rotation_delta_rad: float) -> tuple[np.ndarray, np.ndarray]:
    if motion.kind == "translation":
        delta = float(motion.direction) * float(translation_delta) * np.asarray(motion.axis, dtype=float)
        return base_root_pos + float(alpha) * delta, base_root_quat.copy()
    return base_root_pos.copy(), _quat_mul(base_root_quat, _axis_angle_quat(motion.axis, float(motion.direction) * float(alpha) * rotation_delta_rad))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View a saved grasp_sampling physics evaluation result.")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=-1, help="Attempt index, or -1 for chosen attempt.")
    parser.add_argument("--motion", type=str, default="", help="Motion name to replay, or empty for a static pose.")
    parser.add_argument("--bright-bg", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = load_sampling_run(args.result)
    meta = artifact.metadata
    state = artifact.state

    attempt_index = int(state.chosen_attempt_index) if args.attempt < 0 else int(args.attempt)
    if not 0 <= attempt_index < state.attempt_count:
        raise SystemExit(f"--attempt must be in [0, {state.attempt_count - 1}] or -1.")

    hand = Hand(str(meta["hand"]["side"]))
    prop = prop_from_metadata(dict(meta["prop"]))
    hand_pose = np.asarray(state.base_hand_pose, dtype=float)
    root_pos = hand_pose[:3].copy()
    root_quat = _matrix_to_quat_np(_ortho6d_to_matrix_np(hand_pose[3:9]))
    qpos_target = np.asarray(state.qpos_targets[attempt_index], dtype=float)

    motion_meta = {item["name"]: item for item in meta["motions"]}
    selected_motion = None
    if args.motion:
        if args.motion not in motion_meta:
            raise SystemExit(f"Unknown motion {args.motion!r}. Available: {', '.join(sorted(motion_meta))}")
        item = motion_meta[args.motion]
        selected_motion = MotionSpec(
            name=str(item["name"]),
            kind=str(item["kind"]),
            axis_name=str(item["axis_name"]),
            axis=np.asarray(item["axis"], dtype=np.float32),
            direction=int(item["direction"]),
        )

    scene = build_physics_scene(
        hand,
        prop,
        dict(meta["prop"]),
        root_pos=root_pos,
        root_quat=root_quat,
        qpos_target=qpos_target,
        timestep=float(meta["eval"]["timestep"]),
        density=float(meta["eval"]["object_density"]),
    )
    if args.bright_bg:
        _bright_bg(scene.model)

    print(f"source result      : {meta['source']['result_path']}")
    print(f"sample / attempt   : {meta['source']['sample_index']} / {attempt_index}")
    print(f"chosen attempt     : {state.chosen_attempt_index}")
    print(f"squeeze delta rad  : {float(state.squeeze_deltas[attempt_index]):.6f}")
    print(f"overall score      : {float(state.overall_scores[attempt_index]):.6f}")
    print(
        "initial overlap    : "
        f"{bool(state.initial_overlap[attempt_index])} "
        f"(contacts={int(state.initial_contact_count[attempt_index])}, "
        f"penetrations={int(state.initial_penetration_count[attempt_index])}, "
        f"depth_sum={float(state.initial_depth_sum[attempt_index]):.6f}, "
        f"max_depth={float(state.initial_max_depth[attempt_index]):.6f})"
    )
    print(
        "object in hand     : "
        f"pos={np.array2string(state.object_relative_pos, precision=4)} "
        f"quat={np.array2string(state.object_relative_quat, precision=4)}"
    )
    for motion_info, score, lost, fail in zip(meta["motions"], state.motion_scores[attempt_index], state.motion_lost[attempt_index], state.motion_fail[attempt_index]):
        if not np.isfinite(score):
            print(f"  {motion_info['name']:>4} : not-run lost={bool(lost)} fail={bool(fail)}")
            continue
        print(f"  {motion_info['name']:>4} : score={float(score):.6f} lost={bool(lost)} fail={bool(fail)}")

    with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
        _cam(viewer.cam)
        scene.reset(qpos_target=qpos_target)
        start_time = time.time()
        last_elapsed = 0.0
        rotation_delta_rad = np.deg2rad(float(meta["eval"]["rotation_delta_deg"]))
        translation_delta = float(meta["eval"]["translation_delta"])
        motion_time = float(meta["eval"]["motion_time"])
        while viewer.is_running():
            if selected_motion is None:
                mujoco.mj_forward(scene.model, scene.data)
            else:
                cycle = max(motion_time, 1.0e-6)
                elapsed = (time.time() - start_time) % cycle
                if elapsed < last_elapsed:
                    scene.reset(qpos_target=qpos_target)
                last_elapsed = elapsed
                if motion_time <= 1.0e-6:
                    alpha = 1.0
                else:
                    phase = elapsed / motion_time
                    alpha = 0.5 - 0.5 * np.cos(np.pi * phase)
                motion_root_pos, motion_root_quat = _motion_pose(
                    root_pos,
                    root_quat,
                    selected_motion,
                    alpha,
                    translation_delta,
                    rotation_delta_rad,
                )
                scene.model.body_pos[scene.root_body_id] = motion_root_pos
                scene.model.body_quat[scene.root_body_id] = motion_root_quat
                scene.step(root_pos=motion_root_pos, root_quat=motion_root_quat, qpos_target=qpos_target)
            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
