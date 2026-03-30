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
from grasp_refine.io import load_refine_run


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


def _apply_pose(model: mujoco.MjModel, data: mujoco.MjData, root_body_id: int, hand_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    return root_pos, root_quat


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View a saved grasp_refine result.")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--pose", choices=("initial", "refined", "best"), default="best")
    parser.add_argument("--bright-bg", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = load_refine_run(args.result)
    meta = artifact.metadata
    state = artifact.state

    hand = Hand(str(meta["hand"]["side"]))
    prop = prop_from_metadata(dict(meta["prop"]))
    model, data, root_body_id = _build_scene(hand, prop)
    if args.bright_bg:
        _bright_bg(model)

    hand_pose = {
        "initial": np.asarray(state.initial_hand_pose, dtype=np.float32),
        "refined": np.asarray(state.refined_hand_pose, dtype=np.float32),
        "best": np.asarray(state.best_hand_pose, dtype=np.float32),
    }[args.pose]
    _apply_pose(model, data, root_body_id, hand_pose)

    energy_model = GraspEnergyModel(hand, prop)
    diagnostics = energy_model.diagnostics(
        np.asarray(hand_pose[None, :], dtype=np.float32),
        np.asarray(state.contact_indices[None, :], dtype=np.int32),
    )
    contact_indices = np.asarray(state.contact_indices, dtype=np.int32)
    selected_contacts = diagnostics.nearest_world_positions[0]
    cloud_world_positions = np.asarray(diagnostics.cloud_world_positions[0], dtype=np.float32)
    penetration_depths = np.asarray(diagnostics.penetration_depths[0], dtype=np.float32)
    target_world = (
        np.asarray(prop.pos, dtype=np.float32)[None, :]
        + np.einsum("ij,nj->ni", np.asarray(energy_model.prop_mesh.rotation_world), np.asarray(state.contact_target_local, dtype=np.float32))
    )

    print(f"source result       : {meta['source']['result_path']}")
    print(f"state / sample      : {meta['source']['state']} / {meta['source']['sample_index']}")
    print(f"pose                : {args.pose}")
    print(f"initial total       : {state.initial_energy.total:.6f}")
    print(f"final total         : {state.final_energy.total:.6f}")
    print(f"best total          : {state.best_energy.total:.6f}")
    print(
        "initial actual      : "
        f"contacts={state.initial_actual_contact_count} penetrations={state.initial_actual_penetration_count} depth_sum={state.initial_actual_depth_sum:.6f}"
    )
    print(
        "best actual         : "
        f"contacts={state.best_actual_contact_count} penetrations={state.best_actual_penetration_count} depth_sum={state.best_actual_depth_sum:.6f}"
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        _cam(viewer.cam)
        while viewer.is_running():
            scene = viewer.user_scn
            idx = 0
            for point in np.asarray(selected_contacts, dtype=np.float32):
                idx = _add_marker(scene, idx, point, 0.006, np.array([1.0, 0.65, 0.15, 1.0], dtype=float))
            for point in np.asarray(target_world, dtype=np.float32):
                idx = _add_marker(scene, idx, point, 0.005, np.array([0.15, 0.95, 0.25, 0.95], dtype=float))
            penetrating = cloud_world_positions[penetration_depths > 1.0e-6]
            for point in penetrating:
                idx = _add_marker(scene, idx, point, 0.0035, np.array([1.0, 0.1, 0.85, 0.9], dtype=float))
            scene.ngeom = idx
            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
