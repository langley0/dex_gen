#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import NamedTuple

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_gen.grasp_optimizer_io import load_grasp_run
from grasp_gen.hand import Hand
from grasp_gen.hand_contacts import ContactConfig, ContactRecord
from grasp_gen.mesh_primitives import cylinder_mesh
from grasp_gen.prop import Prop


TARGET = np.zeros(3, dtype=float)


class ViewBatch(NamedTuple):
    hand_pose: np.ndarray
    contact_indices: np.ndarray
    energy_total: np.ndarray
    energy_distance: np.ndarray
    step_index: int


class ViewSample(NamedTuple):
    root_pos: np.ndarray
    root_quat: np.ndarray
    qpos: np.ndarray
    all_contacts: list[ContactRecord]
    selected_contacts: list[ContactRecord]
    nearest_prop_points: np.ndarray
    contact_indices: np.ndarray
    energy_total: float
    energy_distance: float


def _cam(cam: mujoco.MjvCamera) -> None:
    cam.lookat[:] = TARGET
    cam.distance = 0.72
    cam.azimuth = 145.0
    cam.elevation = -18.0


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


def _matrix_to_rpy_deg_np(rotation: np.ndarray) -> np.ndarray:
    sy = float(np.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0]))
    singular = sy < 1.0e-8
    if not singular:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.rad2deg(np.array([roll, pitch, yaw], dtype=float))


def _closest_point_on_triangle(point: np.ndarray, tri: np.ndarray) -> np.ndarray:
    a, b, c = tri
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a.copy()

    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b.copy()

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / max(d1 - d3, 1.0e-12)
        return a + v * ab

    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c.copy()

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / max(d2 - d6, 1.0e-12)
        return a + w * ac

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        edge = c - b
        w = (d4 - d3) / max((d4 - d3) + (d5 - d6), 1.0e-12)
        return b + w * edge

    denom = max(va + vb + vc, 1.0e-12)
    v = vb / denom
    w = vc / denom
    return a + ab * v + ac * w


def _prop_triangles_world(prop: Prop) -> np.ndarray:
    quat = np.asarray(prop.quat, dtype=float)
    matrix = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(matrix, quat)
    rotation = matrix.reshape(3, 3)
    triangles_local = np.asarray(prop.vertices[prop.faces], dtype=float)
    return prop.pos[None, None, :] + np.einsum("ij,nkj->nki", rotation, triangles_local)


def _nearest_prop_points(triangles_world: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    nearest = np.zeros((len(points_world), 3), dtype=float)
    for point_index, point in enumerate(np.asarray(points_world, dtype=float)):
        best_dist2 = np.inf
        best_point = point.copy()
        for tri in triangles_world:
            candidate = _closest_point_on_triangle(point, tri)
            dist2 = float(np.sum((point - candidate) ** 2))
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_point = candidate
        nearest[point_index] = best_point
    return nearest


def _make_prop(prop_meta: dict[str, object]) -> Prop:
    kind = str(prop_meta.get("kind", ""))
    if kind != "cylinder":
        raise SystemExit(f"Unsupported prop kind in result file: {kind!r}")
    vertices, faces = cylinder_mesh(
        float(prop_meta["radius"]),
        float(prop_meta["half_height"]),
        sides=int(prop_meta["sides"]),
    )
    return Prop(
        vertices,
        faces,
        pos=np.asarray(prop_meta["pos"], dtype=float),
        quat=np.asarray(prop_meta["quat"], dtype=float),
        name=str(prop_meta.get("name", "cylinder")),
    )


def _build_scene(hand: Hand, prop: Prop) -> tuple[mujoco.MjModel, mujoco.MjData, int]:
    spec = hand.mjcf()
    prop.add_to(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    root_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{hand.side}_hand_base"))
    return model, data, root_body_id


def _apply_pose_to_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_body_id: int,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    qpos: np.ndarray,
) -> None:
    model.body_pos[root_body_id] = np.asarray(root_pos, dtype=float)
    model.body_quat[root_body_id] = np.asarray(root_quat, dtype=float)
    data.qpos[:] = np.asarray(qpos, dtype=float)
    data.qvel[:] = 0.0
    if model.nu > 0:
        data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def _select_view_batch(artifact, state_name: str) -> ViewBatch:
    state = artifact.state
    if state_name == "best":
        energy = state.best_energy
        return ViewBatch(
            hand_pose=np.asarray(state.best_hand_pose, dtype=np.float32),
            contact_indices=np.asarray(state.best_contact_indices, dtype=np.int32),
            energy_total=np.asarray(energy.total, dtype=np.float32),
            energy_distance=np.asarray(energy.distance, dtype=np.float32),
            step_index=int(np.asarray(state.step_index)),
        )

    energy = state.energy
    return ViewBatch(
        hand_pose=np.asarray(state.hand_pose, dtype=np.float32),
        contact_indices=np.asarray(state.contact_indices, dtype=np.int32),
        energy_total=np.asarray(energy.total, dtype=np.float32),
        energy_distance=np.asarray(energy.distance, dtype=np.float32),
        step_index=int(np.asarray(state.step_index)),
    )


def _resolve_index(batch: ViewBatch, index: int) -> int:
    if index < 0:
        return int(np.argmin(batch.energy_total))
    if not 0 <= index < int(batch.hand_pose.shape[0]):
        raise SystemExit(f"--index must be in [0, {int(batch.hand_pose.shape[0]) - 1}] or -1.")
    return int(index)


def _sample_view(
    hand: Hand,
    prop_triangles_world: np.ndarray,
    batch: ViewBatch,
    sample_index: int,
    contact_cfg: ContactConfig,
) -> ViewSample:
    hand_pose = np.asarray(batch.hand_pose[sample_index], dtype=float)
    root_pos = hand_pose[:3].copy()
    root_rot = _ortho6d_to_matrix_np(hand_pose[3:9])
    root_quat = _matrix_to_quat_np(root_rot)
    qpos = hand_pose[9:].copy()

    hand.apply_state(qpos=qpos, ctrl=np.zeros(hand.model.nu, dtype=float), root_pos=root_pos, root_quat=root_quat)
    all_contacts = hand.contacts(cfg=contact_cfg)
    contact_indices = np.asarray(batch.contact_indices[sample_index], dtype=np.int32)
    selected_contacts = [all_contacts[int(index)] for index in contact_indices.tolist()]
    nearest_points = _nearest_prop_points(
        prop_triangles_world,
        np.asarray([record.world_pos for record in selected_contacts], dtype=float),
    )
    return ViewSample(
        root_pos=root_pos,
        root_quat=root_quat,
        qpos=qpos,
        all_contacts=all_contacts,
        selected_contacts=selected_contacts,
        nearest_prop_points=nearest_points,
        contact_indices=contact_indices,
        energy_total=float(batch.energy_total[sample_index]),
        energy_distance=float(batch.energy_distance[sample_index]),
    )


def _state_text(hand: Hand, sample: ViewSample, batch: ViewBatch, sample_index: int) -> str:
    root_rpy = _matrix_to_rpy_deg_np(_ortho6d_to_matrix_np(batch.hand_pose[sample_index, 3:9]))
    lines = [
        (
            f"sample={sample_index} step={batch.step_index} "
            f"energy={sample.energy_total:.6f} "
            f"(distance={sample.energy_distance:.6f})"
        ),
        (
            "root 6dof      : "
            f"xyz=[{sample.root_pos[0]: .4f}, {sample.root_pos[1]: .4f}, {sample.root_pos[2]: .4f}] "
            f"rpy_deg=[{root_rpy[0]: .1f}, {root_rpy[1]: .1f}, {root_rpy[2]: .1f}]"
        ),
        "selected points :",
    ]
    for slot, (point_index, record) in enumerate(zip(sample.contact_indices.tolist(), sample.selected_contacts)):
        lines.append(f"  {slot}: idx={point_index:02d} {record.body_name} ({record.finger}/{record.role})")
    lines.append("joints         :")
    for actuator in hand.actuators:
        rad = float(sample.qpos[actuator.qpos_index])
        deg = float(np.rad2deg(rad))
        lines.append(f"  {actuator.joint_name}: {rad: .4f} rad ({deg: .1f} deg)")
    return "\n".join(lines)


def _overlay(viewer, sample: ViewSample, *, show_all: bool) -> None:
    scene = viewer.user_scn
    idx = 0
    idx = _add_marker(scene, idx, TARGET, 0.015, np.array([1.0, 0.2, 0.2, 1.0], dtype=float))
    idx = _add_marker(scene, idx, sample.root_pos, 0.006, np.array([1.0, 1.0, 1.0, 1.0], dtype=float))
    if show_all:
        for record in sample.all_contacts:
            idx = _add_marker(scene, idx, record.world_pos, 0.0045, np.array([0.2, 0.85, 1.0, 0.25], dtype=float))
    for record in sample.selected_contacts:
        idx = _add_marker(scene, idx, record.world_pos, 0.007, np.array([1.0, 0.72, 0.2, 1.0], dtype=float))
    for point in sample.nearest_prop_points:
        idx = _add_marker(scene, idx, point, 0.006, np.array([0.30, 1.0, 0.40, 1.0], dtype=float))
    scene.ngeom = idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View a saved grasp optimizer result.")
    parser.add_argument("--result", type=Path, required=True, help="Saved optimizer .npz artifact.")
    parser.add_argument("--state", choices=("best", "current"), default="best", help="Which saved state to render.")
    parser.add_argument("--index", type=int, default=-1, help="Batch sample index. Use -1 to pick the lowest-energy sample.")
    parser.add_argument("--show-all", action="store_true", help="Overlay all candidate hand contact points.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = load_grasp_run(args.result)
    metadata = artifact.metadata

    hand_side = str(metadata["hand"]["side"])
    hand = Hand(hand_side)
    prop = _make_prop(metadata["prop"])
    prop_triangles_world = _prop_triangles_world(prop)
    batch = _select_view_batch(artifact, args.state)
    sample_index = _resolve_index(batch, args.index)

    contact_meta = metadata["contact"]
    contact_cfg = ContactConfig(
        n_per_seg=int(contact_meta["n_per_seg"]),
        thumb_weight=float(contact_meta["thumb_weight"]),
        palm_clearance=float(contact_meta["palm_clearance"]),
    )

    model, data, root_body_id = _build_scene(hand, prop)
    sample = _sample_view(hand, prop_triangles_world, batch, sample_index, contact_cfg)
    _apply_pose_to_scene(model, data, root_body_id, sample.root_pos, sample.root_quat, sample.qpos)

    run_meta = metadata["run"]
    result_meta = metadata.get("result", {})
    print(f"result path      : {artifact.path}")
    print(f"hand side        : {hand_side}")
    print(f"jax backend      : {run_meta.get('backend', 'unknown')}")
    print(f"saved step count : {run_meta.get('steps', 'unknown')}")
    print(f"batch size       : {run_meta.get('batch', 'unknown')}")
    print(f"view state       : {args.state}")
    print(f"sample index     : {sample_index}")
    if "best_sample_index" in result_meta:
        print(f"best sample idx  : {result_meta['best_sample_index']}")
    print("viewer colors    : red=target/prop center, white=root, orange=selected hand contacts, green=nearest prop points")
    if args.show_all:
        print("viewer colors    : cyan=all candidate hand contacts")
    print(_state_text(hand, sample, batch, sample_index), flush=True)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        _cam(viewer.cam)
        while viewer.is_running():
            _overlay(viewer, sample, show_all=args.show_all)
            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
