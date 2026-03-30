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

from grasp_gen.grasp_energy import GraspEnergyModel
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


def _add_marker(scene, idx: int, pos: np.ndarray, radius: float, rgba: np.ndarray) -> int:
    limit = int(getattr(scene, "maxgeom", len(scene.geoms)))
    if idx >= limit:
        return idx
    mujoco.mjv_initGeom(
        scene.geoms[idx],
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        np.array([radius, 0.0, 0.0], dtype=float),
        np.asarray(pos, dtype=float),
        np.eye(3, dtype=float).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    return idx + 1


def _select_sample(state: dict[str, np.ndarray], index: int) -> int:
    actual_fixed_mask = np.asarray(state.get("actual_fixed_mask", np.zeros_like(state["fixed_mask"])), dtype=bool)
    fixed_mask = np.asarray(state["fixed_mask"], dtype=bool)
    if index >= 0:
        return int(index)
    if np.any(actual_fixed_mask):
        best_total = np.asarray(state["best_total"], dtype=np.float32)
        return int(np.argmin(np.where(actual_fixed_mask, best_total, np.inf)))
    if np.any(fixed_mask):
        best_total = np.asarray(state["best_total"], dtype=np.float32)
        return int(np.argmin(np.where(fixed_mask, best_total, np.inf)))
    return int(np.argmin(np.asarray(state["best_total"], dtype=np.float32)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare before/after from a batch grasp_refine result.")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--index", type=int, default=-1, help="Sample index, or -1 for best fixed case.")
    parser.add_argument("--interval", type=float, default=1.5, help="Seconds between before/after flips.")
    parser.add_argument("--bright-bg", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata, state = load_refine_batch(args.result)
    sample_index = _select_sample(state, int(args.index))

    hand = Hand(str(metadata["hand"]["side"]))
    prop = prop_from_metadata(dict(metadata["prop"]))
    model, data, root_body_id = _build_scene(hand, prop)
    if args.bright_bg:
        _bright_bg(model)

    initial_hand_pose = np.asarray(state["initial_hand_pose"][sample_index], dtype=np.float32)
    best_hand_pose = np.asarray(state["best_hand_pose"][sample_index], dtype=np.float32)
    contact_indices = np.asarray(state["contact_indices"][sample_index], dtype=np.int32)
    contact_target_local = np.asarray(state["contact_target_local"][sample_index], dtype=np.float32)
    energy_model = GraspEnergyModel(hand, prop)

    print(f"source result       : {metadata['source']['result_path']}")
    print(f"sample index        : {sample_index}")
    print(f"fixed               : {bool(state['fixed_mask'][sample_index])}")
    if "actual_fixed_mask" in state:
        print(f"actual fixed        : {bool(state['actual_fixed_mask'][sample_index])}")
    print(
        "initial -> best pen : "
        f"{float(state['initial_penetration'][sample_index]):.6f} -> {float(state['best_penetration'][sample_index]):.6f}"
    )
    if "initial_actual_depth_sum" in state:
        print(
            "initial -> best act : "
            f"{float(state['initial_actual_depth_sum'][sample_index]):.6f} -> "
            f"{float(state['best_actual_depth_sum'][sample_index]):.6f}"
        )
    print(
        "initial -> best tot : "
        f"{float(state['initial_total'][sample_index]):.6f} -> {float(state['best_total'][sample_index]):.6f}"
    )
    print("viewer              : flips between initial and best pose")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        _cam(viewer.cam)
        start = time.time()
        while viewer.is_running():
            elapsed = time.time() - start
            use_best = (int(elapsed / max(float(args.interval), 1.0e-3)) % 2) == 1
            current_pose = best_hand_pose if use_best else initial_hand_pose
            _apply_pose(model, data, root_body_id, current_pose)

            diagnostics = energy_model.diagnostics(
                np.asarray(current_pose[None, :], dtype=np.float32),
                np.asarray(contact_indices[None, :], dtype=np.int32),
            )
            target_world = (
                np.asarray(prop.pos, dtype=np.float32)[None, :]
                + np.einsum("ij,nj->ni", np.asarray(energy_model.prop_mesh.rotation_world), contact_target_local)
            )
            penetrating = np.asarray(diagnostics.cloud_world_positions[0], dtype=np.float32)[
                np.asarray(diagnostics.penetration_depths[0], dtype=np.float32) > 1.0e-6
            ]
            nearest = np.asarray(diagnostics.nearest_world_positions[0], dtype=np.float32)

            scene = viewer.user_scn
            idx = 0
            color = np.array([0.15, 0.75, 1.0, 1.0], dtype=float) if use_best else np.array([1.0, 0.55, 0.15, 1.0], dtype=float)
            for point in nearest:
                idx = _add_marker(scene, idx, point, 0.006, color)
            for point in target_world:
                idx = _add_marker(scene, idx, point, 0.0045, np.array([0.15, 0.95, 0.25, 0.95], dtype=float))
            for point in penetrating:
                idx = _add_marker(scene, idx, point, 0.0035, np.array([1.0, 0.1, 0.85, 0.9], dtype=float))
            scene.ngeom = idx
            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
