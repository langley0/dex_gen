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

from grasp_gen.grasp_optimizer_io import load_grasp_run
from grasp_gen.hand import Hand
from grasp_gen.prop_assets import prop_from_metadata
from grasp_refine.batch import load_refine_batch


TARGET = np.zeros(3, dtype=float)


def _bright_bg(model: mujoco.MjModel) -> None:
    model.vis.rgba.haze[:] = np.array([0.97, 0.97, 0.99, 1.0], dtype=float)
    model.vis.rgba.fog[:] = np.array([0.97, 0.97, 0.99, 1.0], dtype=float)
    model.vis.headlight.ambient[:] = np.array([0.55, 0.55, 0.55], dtype=float)
    model.vis.headlight.diffuse[:] = np.array([0.85, 0.85, 0.85], dtype=float)
    model.vis.headlight.specular[:] = np.array([0.15, 0.15, 0.15], dtype=float)


def _cam(cam: mujoco.MjvCamera) -> None:
    cam.lookat[:] = TARGET
    cam.distance = 0.72
    cam.azimuth = 145.0
    cam.elevation = -18.0


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
    quat /= max(float(np.linalg.norm(quat)), 1.0e-8)
    if quat[0] < 0.0:
        quat *= -1.0
    return quat


def _build_scene(hand: Hand, prop) -> tuple[mujoco.MjModel, mujoco.MjData, int]:
    spec = hand.mjcf()
    prop.add_to(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    root_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{hand.side}_hand_base"))
    return model, data, root_body_id


def _apply_pose(model: mujoco.MjModel, data: mujoco.MjData, root_body_id: int, hand_pose: np.ndarray) -> None:
    root_pos = np.asarray(hand_pose[:3], dtype=float)
    root_quat = _matrix_to_quat_np(_ortho6d_to_matrix_np(np.asarray(hand_pose[3:9], dtype=float)))
    qpos = np.asarray(hand_pose[9:], dtype=float)
    model.body_pos[root_body_id] = root_pos
    model.body_quat[root_body_id] = root_quat
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    if model.nu > 0:
        data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def _select_index(refine_state: dict[str, np.ndarray], index: int) -> int:
    if index >= 0:
        return int(index)
    actual_fixed = np.asarray(refine_state.get("actual_fixed_mask", np.zeros_like(refine_state["fixed_mask"])), dtype=bool)
    fixed = np.asarray(refine_state["fixed_mask"], dtype=bool)
    best_total = np.asarray(refine_state["best_total"], dtype=np.float32)
    if np.any(actual_fixed):
        return int(np.argmin(np.where(actual_fixed, best_total, np.inf)))
    if np.any(fixed):
        return int(np.argmin(np.where(fixed, best_total, np.inf)))
    return int(np.argmin(best_total))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare source grasp and refined grasp by flipping between them.")
    parser.add_argument("--refine-result", type=Path, required=True)
    parser.add_argument("--index", type=int, default=-1)
    parser.add_argument("--interval", type=float, default=1.5)
    parser.add_argument("--bright-bg", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    refine_meta, refine_state = load_refine_batch(args.refine_result)
    source_meta = dict(refine_meta["source"])
    source_artifact = load_grasp_run(Path(source_meta["result_path"]))
    state_name = str(source_meta["state"])
    sample_index = _select_index(refine_state, int(args.index))

    if state_name == "best":
        source_pose_batch = np.asarray(source_artifact.state.best_hand_pose, dtype=np.float32)
        source_total = np.asarray(source_artifact.state.best_energy.total, dtype=np.float32)
    else:
        source_pose_batch = np.asarray(source_artifact.state.hand_pose, dtype=np.float32)
        source_total = np.asarray(source_artifact.state.energy.total, dtype=np.float32)

    source_hand_pose = np.asarray(source_pose_batch[sample_index], dtype=np.float32)
    refined_best_pose = np.asarray(refine_state["best_hand_pose"][sample_index], dtype=np.float32)
    refined_initial_pose = np.asarray(refine_state["initial_hand_pose"][sample_index], dtype=np.float32)

    hand = Hand(str(refine_meta["hand"]["side"]))
    prop = prop_from_metadata(dict(refine_meta["prop"]))
    model, data, root_body_id = _build_scene(hand, prop)
    if args.bright_bg:
        _bright_bg(model)

    print(f"source result       : {source_meta['result_path']}")
    print(f"state               : {state_name}")
    print(f"sample index        : {sample_index}")
    print(f"source total        : {float(source_total[sample_index]):.6f}")
    print(f"refine init total   : {float(refine_state['initial_total'][sample_index]):.6f}")
    print(f"refine best total   : {float(refine_state['best_total'][sample_index]):.6f}")
    if "initial_actual_depth_sum" in refine_state:
        print(
            "actual depth        : "
            f"{float(refine_state['initial_actual_depth_sum'][sample_index]):.6f} -> "
            f"{float(refine_state['best_actual_depth_sum'][sample_index]):.6f}"
        )
    print("viewer              : flips between source and refined-best")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        _cam(viewer.cam)
        start = time.time()
        while viewer.is_running():
            elapsed = time.time() - start
            use_refined = (int(elapsed / max(float(args.interval), 1.0e-3)) % 2) == 1
            hand_pose = refined_best_pose if use_refined else source_hand_pose
            _apply_pose(model, data, root_body_id, hand_pose)
            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
