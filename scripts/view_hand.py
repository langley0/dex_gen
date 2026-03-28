#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_gen.hand import Hand
from grasp_gen.hand_contacts import ContactConfig, ContactRecord, build_contacts
from grasp_gen.hand_pose import HandPose, InitPoseConfig, apply_pose, make_home_pose, make_init_pose
from grasp_gen.mesh_primitives import cylinder_mesh
from grasp_gen.prop import Prop


PICK_CYLINDER_RADIUS = 0.045
PICK_CYLINDER_HALF_HEIGHT = 0.165
PICK_CYLINDER_POS = np.zeros(3, dtype=float)


def _cam(cam: mujoco.MjvCamera, lookat: np.ndarray) -> None:
    cam.lookat[:] = np.asarray(lookat, dtype=float)
    cam.distance = 0.65
    cam.azimuth = 145.0
    cam.elevation = -18.0


def _make_prop() -> Prop:
    vertices, faces = cylinder_mesh(PICK_CYLINDER_RADIUS, PICK_CYLINDER_HALF_HEIGHT, sides=32)
    return Prop(vertices, faces, pos=PICK_CYLINDER_POS, name="pickup_cylinder")


def _build_scene(hand: Hand, pose: HandPose, prop: Prop) -> tuple[mujoco.MjModel, mujoco.MjData]:
    spec = mujoco.MjSpec()
    spec.modelname = f"{hand.side}_hand_with_prop"
    spec.stat.center = 0.5 * (prop.pos + pose.palm_pos)
    spec.stat.extent = 0.95

    light = spec.worldbody.add_light()
    light.name = "key_light"
    light.pos = np.array([0.3, -0.6, 1.3], dtype=float)
    light.dir = np.array([0.0, 0.2, -1.0], dtype=float)
    light.diffuse = np.array([0.95, 0.95, 0.95], dtype=float)
    light.specular = np.array([0.25, 0.25, 0.25], dtype=float)
    light.castshadow = True

    prop.add_to(spec)

    mount = spec.worldbody.add_site()
    mount.name = "hand_mount"
    mount.pos = np.zeros(3, dtype=float)
    mount.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    spec.attach(hand.mjcf(root_pos=pose.root_pos, root_quat=pose.root_quat), site=mount)

    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(pose.qpos, dtype=float)
    data.qvel[:] = 0.0
    if model.nu > 0:
        ctrl = np.asarray(pose.ctrl, dtype=float)
        data.ctrl[:] = np.clip(ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    mujoco.mj_forward(model, data)
    return model, data


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


def _overlay(viewer, anchor: np.ndarray, contacts: list[ContactRecord]) -> None:
    scene = viewer.user_scn
    idx = 0
    idx = _add_marker(scene, idx, anchor, 0.015, np.array([1.0, 0.2, 0.2, 1.0], dtype=float))
    for record in contacts:
        color = np.array([1.0, 0.7, 0.2, 1.0], dtype=float) if record.finger == "thumb" else np.array(
            [0.2, 0.85, 1.0, 0.95],
            dtype=float,
        )
        idx = _add_marker(scene, idx, record.world_pos, 0.006, color)
    scene.ngeom = idx


def run(hand: Hand, pose: HandPose, prop: Prop, contacts: list[ContactRecord]) -> None:
    model, data = _build_scene(hand, pose, prop)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        _cam(viewer.cam, 0.5 * (prop.pos + pose.palm_pos))
        while viewer.is_running():
            _overlay(viewer, prop.pos, contacts)
            viewer.sync()
            time.sleep(1.0 / 60.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the standalone grasp_gen Hand in the MuJoCo viewer.")
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument("--pose", choices=("init", "home"), default="init")
    parser.add_argument("--points", type=int, default=10, help="Contact points per segment used for the init pose.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offset", type=float, default=0.30, help="Anchor-to-palm distance for the init pose.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hand = Hand(args.hand)
    prop = _make_prop()
    if args.pose == "init":
        pose = make_init_pose(
            hand,
            anchor=prop.pos.copy(),
            cfg=InitPoseConfig(contact_n_per_seg=args.points, palm_offset=args.offset),
            seed=args.seed,
        )
    else:
        pose = make_home_pose(hand)
    apply_pose(hand, pose)
    contacts = build_contacts(
        hand,
        qpos=pose.qpos,
        ctrl=pose.ctrl,
        cfg=ContactConfig(n_per_seg=args.points, thumb_weight=4.0),
    )
    body_counts = Counter(record.body_name for record in contacts)
    weights = np.asarray([record.weight for record in contacts], dtype=float)

    print(f"Hand side      : {args.hand}")
    print(f"Pose mode      : {args.pose}")
    print(f"qpos / ctrl    : {hand.model.nq} / {hand.model.nu}")
    print(f"contact points : {len(contacts)}")
    print(f"per segment    : {args.points}")
    print(f"weight range   : [{float(weights.min()):.3f}, {float(weights.max()):.3f}]")
    print(f"prop mesh      : {len(prop.vertices)} verts / {len(prop.faces)} faces")
    print(f"prop center    : [{prop.pos[0]:.3f}, {prop.pos[1]:.3f}, {prop.pos[2]:.3f}]")
    print(f"anchor         : [{pose.anchor[0]:.3f}, {pose.anchor[1]:.3f}, {pose.anchor[2]:.3f}]")
    print(f"root pos       : [{pose.root_pos[0]:.3f}, {pose.root_pos[1]:.3f}, {pose.root_pos[2]:.3f}]")
    print(
        f"root quat      : [{pose.root_quat[0]:.3f}, {pose.root_quat[1]:.3f}, "
        f"{pose.root_quat[2]:.3f}, {pose.root_quat[3]:.3f}]"
    )
    print("palm included  : no")
    print("point bodies   :")
    for body_name, count in body_counts.items():
        print(f"  {body_name}: {count}")
    print("viewer colors  : red=prop center, orange=thumb contact candidates, cyan=other candidates")
    print("Close the viewer window to exit.")

    run(hand, pose, prop, contacts)


if __name__ == "__main__":
    main()
