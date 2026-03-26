#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PANDA_XML_PATH = PROJECT_ROOT / "third_party/mujoco_menagerie/franka_emika_panda" / "panda_nohand.xml"
PANDA_ASSET_DIR = PANDA_XML_PATH.parent / "assets"
INSPIRE_XML_PATHS = {
    "right": PROJECT_ROOT / "assets" / "inspire" / "right.xml",
    "left": PROJECT_ROOT / "assets" / "inspire" / "left.xml",
}
INSPIRE_ASSET_DIR = PROJECT_ROOT / "assets" / "inspire" / "assets"

ARM_ACTUATOR_COUNT = 7
HAND_ACTUATOR_COUNT = 12

ARM_HOME_QPOS = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], dtype=float)
STANDARD_GRIPPER_MOUNT_TO_INSPIRE_QUAT = np.array([0.70710678, -0.70710678, 0.0, 0.0], dtype=float)
PICK_CYLINDER_RADIUS = 0.045
PICK_CYLINDER_HALF_HEIGHT = 0.165
PICK_CYLINDER_MASS = 0.84375
PICK_CYLINDER_POS = np.array([0.65, 0.06, PICK_CYLINDER_HALF_HEIGHT], dtype=float)
ARM_PICK_APPROACH_QPOS = np.array([0.049857, 0.258130, 0.047492, -1.341370, 0.008979, 1.783554, -0.7853], dtype=float)
ARM_PICK_QPOS = np.array([0.049843, 0.281704, 0.047219, -1.405452, 0.008817, 1.751973, -0.7853], dtype=float)
ARM_PICK_LIFT_QPOS = np.array([0.050507, 0.248268, 0.048200, -1.210721, 0.009345, 1.860283, -0.7853], dtype=float)
DIRECT_ATTACH_HAND_CLOSE_FRACTIONS = np.array(
    [0.70, 0.55, 0.70, 0.75, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85],
    dtype=float,
)


def _smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _lerp(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    return start + _smoothstep(alpha) * (end - start)


def _absolutize_mesh_paths(spec: mujoco.MjSpec, asset_dir: Path) -> None:
    for mesh in spec.meshes:
        if mesh.file and not Path(mesh.file).is_absolute():
            mesh.file = str((asset_dir / mesh.file).resolve())


def _enable_inspire_contacts(hand_spec: mujoco.MjSpec) -> None:
    for geom in hand_spec.geoms:
        if not geom.name:
            continue
        if not geom.name.startswith("collision_hand_"):
            continue
        geom.conaffinity = 1
        geom.contype = 0
        geom.friction = np.array([1.0, 0.05, 0.01], dtype=float)


def _remove_inspire_root_dofs(hand_spec: mujoco.MjSpec, side: str) -> None:
    root_actuators = [
        f"{side}_pos_x_position",
        f"{side}_pos_y_position",
        f"{side}_pos_z_position",
        f"{side}_rot_x_position",
        f"{side}_rot_y_position",
        f"{side}_rot_z_position",
    ]
    root_joints = [
        f"{side}_pos_x",
        f"{side}_pos_y",
        f"{side}_pos_z",
        f"{side}_rot_x",
        f"{side}_rot_y",
        f"{side}_rot_z",
    ]

    for actuator_name in root_actuators:
        actuator = hand_spec.actuator(actuator_name)
        if actuator is not None:
            hand_spec.delete(actuator)

    for joint_name in root_joints:
        joint = hand_spec.joint(joint_name)
        if joint is not None:
            hand_spec.delete(joint)


def _load_hand_spec(side: str) -> mujoco.MjSpec:
    hand_spec = mujoco.MjSpec.from_file(str(INSPIRE_XML_PATHS[side]))
    _absolutize_mesh_paths(hand_spec, INSPIRE_ASSET_DIR)
    _enable_inspire_contacts(hand_spec)
    _remove_inspire_root_dofs(hand_spec, side)

    for light in list(hand_spec.lights):
        hand_spec.delete(light)

    root_body = hand_spec.body(f"{side}_hand_base")
    root_body.gravcomp = 0.0
    # Match the stock Panda gripper mount so the Inspire fingertips point downward
    # in the standard home pose while keeping the hand base directly on attachment_site.
    root_body.quat = STANDARD_GRIPPER_MOUNT_TO_INSPIRE_QUAT.copy()
    hand_spec.modelname = f"inspire_{side}_attached"
    return hand_spec


def _add_floor(spec: mujoco.MjSpec) -> None:
    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = np.array([0.0, 0.0, 0.05], dtype=float)
    floor.pos = np.array([0.0, 0.0, 0.0], dtype=float)
    floor.rgba = np.array([0.92, 0.93, 0.95, 1.0], dtype=float)
    floor.friction = np.array([1.0, 0.05, 0.01], dtype=float)


def _add_pick_cylinder(spec: mujoco.MjSpec) -> None:
    body = spec.worldbody.add_body()
    body.name = "pickup_cylinder"
    body.pos = PICK_CYLINDER_POS.copy()
    body.add_freejoint(name="pickup_cylinder_freejoint")

    geom = body.add_geom()
    geom.name = "pickup_cylinder_geom"
    geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    geom.size = np.array([PICK_CYLINDER_RADIUS, PICK_CYLINDER_HALF_HEIGHT, 0.0], dtype=float)
    geom.mass = PICK_CYLINDER_MASS
    geom.condim = 4
    geom.friction = np.array([1.1, 0.05, 0.01], dtype=float)
    geom.rgba = np.array([0.91, 0.58, 0.19, 1.0], dtype=float)

    top_site = body.add_site()
    top_site.name = "pickup_cylinder_top"
    top_site.type = mujoco.mjtGeom.mjGEOM_SPHERE
    top_site.size = np.array([0.009, 0.0, 0.0], dtype=float)
    top_site.pos = np.array([0.0, 0.0, PICK_CYLINDER_HALF_HEIGHT], dtype=float)
    top_site.rgba = np.array([0.96, 0.91, 0.82, 0.8], dtype=float)


def build_franka_inspire_spec(side: str) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(PANDA_XML_PATH))
    _absolutize_mesh_paths(spec, PANDA_ASSET_DIR)

    spec.modelname = f"franka_inspire_{side}"
    spec.stat.center = np.array([0.3, 0.0, 0.4], dtype=float)
    spec.stat.extent = 1.0
    spec.visual.global_.offwidth = 1600
    spec.visual.global_.offheight = 1200

    _add_floor(spec)
    _add_pick_cylinder(spec)

    hand_spec = _load_hand_spec(side)
    spec.attach(hand_spec, prefix="inspire_", site=spec.site("attachment_site"))
    return spec


def build_model(side: str, dump_xml_path: Path | None) -> tuple[mujoco.MjSpec, mujoco.MjModel, mujoco.MjData]:
    spec = build_franka_inspire_spec(side)
    model = spec.compile()
    data = mujoco.MjData(model)

    if dump_xml_path is not None:
        dump_xml_path.parent.mkdir(parents=True, exist_ok=True)
        dump_xml_path.write_text(spec.to_xml(), encoding="utf-8")

    initialize_state(model, data)
    return spec, model, data


def initialize_state(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    qpos = model.qpos0.copy()
    qpos[:ARM_ACTUATOR_COUNT] = ARM_HOME_QPOS
    data.qpos[:] = qpos
    data.qvel[:] = 0.0

    ctrl = np.zeros(model.nu, dtype=float)
    ctrl[:ARM_ACTUATOR_COUNT] = ARM_HOME_QPOS
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)


def _hand_ctrl_ranges(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    ctrl_range = model.actuator_ctrlrange[ARM_ACTUATOR_COUNT:]
    lower = ctrl_range[:, 0]
    upper = ctrl_range[:, 1]
    open_pose = np.clip(np.zeros_like(lower), lower, upper)
    close_pose = lower + DIRECT_ATTACH_HAND_CLOSE_FRACTIONS * (upper - lower)
    return open_pose, np.clip(close_pose, lower, upper)


def build_demo_ctrl(model: mujoco.MjModel, sim_time: float) -> np.ndarray:
    open_hand, closed_hand = _hand_ctrl_ranges(model)

    phase = sim_time % 8.0
    if phase < 2.0:
        arm_target = _lerp(ARM_HOME_QPOS, ARM_PICK_APPROACH_QPOS, phase / 2.0)
        hand_target = open_hand
    elif phase < 3.5:
        arm_target = _lerp(ARM_PICK_APPROACH_QPOS, ARM_PICK_QPOS, (phase - 2.0) / 1.5)
        hand_target = open_hand
    elif phase < 4.75:
        arm_target = ARM_PICK_QPOS
        hand_target = _lerp(open_hand, closed_hand, (phase - 3.5) / 1.25)
    elif phase < 6.25:
        arm_target = _lerp(ARM_PICK_QPOS, ARM_PICK_LIFT_QPOS, (phase - 4.75) / 1.5)
        hand_target = closed_hand
    elif phase < 7.0:
        arm_target = ARM_PICK_LIFT_QPOS
        hand_target = closed_hand
    else:
        arm_target = _lerp(ARM_PICK_LIFT_QPOS, ARM_HOME_QPOS, (phase - 7.0) / 1.0)
        hand_target = _lerp(closed_hand, open_hand, (phase - 7.0) / 1.0)

    return np.concatenate([arm_target, hand_target])


def _configure_free_camera(cam: mujoco.MjvCamera) -> None:
    cam.lookat = np.array([0.36, 0.0, 0.26], dtype=float)
    cam.distance = 1.25
    cam.azimuth = 160.0
    cam.elevation = -16.0


def save_snapshot(model: mujoco.MjModel, data: mujoco.MjData, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=720, width=960)
    camera = mujoco.MjvCamera()
    _configure_free_camera(camera)
    renderer.update_scene(data, camera=camera)
    Image.fromarray(renderer.render()).save(output_path)
    renderer.close()


def actuator_summary(model: mujoco.MjModel) -> str:
    lines = []
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        ctrl_min, ctrl_max = model.actuator_ctrlrange[actuator_id]
        lines.append(f"{actuator_id:02d}: {name} [{ctrl_min:.4f}, {ctrl_max:.4f}]")
    return "\n".join(lines)


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData, demo: bool) -> None:
    timestep = model.opt.timestep
    start_wall_time = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        _configure_free_camera(viewer.cam)
        while viewer.is_running():
            elapsed = time.time() - start_wall_time
            if demo:
                data.ctrl[:] = build_demo_ctrl(model, elapsed)
            mujoco.mj_step(model, data)
            viewer.sync()

            remaining = timestep - (time.time() - start_wall_time - elapsed)
            if remaining > 0.0:
                time.sleep(remaining)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and visualize a Franka Panda arm with an attached Inspire hand."
    )
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument(
        "--mode",
        choices=("demo", "hold"),
        default="demo",
        help="demo: reach-close-lift loop, hold: keep the home pose.",
    )
    parser.add_argument(
        "--dump-xml",
        type=Path,
        default=None,
        help="Optional path to save the generated combined MJCF.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Render one offscreen image and exit instead of launching the interactive viewer.",
    )
    parser.add_argument(
        "--print-actuators",
        action="store_true",
        help="Print all actuator names and control ranges before launching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, model, data = build_model(args.hand, args.dump_xml)

    if args.print_actuators:
        print(f"Arm resource : {PANDA_XML_PATH}")
        print(f"Hand resource: {INSPIRE_XML_PATHS[args.hand]}")
        print(actuator_summary(model))

    if args.snapshot is not None:
        if args.mode == "demo":
            data.ctrl[:] = build_demo_ctrl(model, 4.75)
            for _ in range(250):
                mujoco.mj_step(model, data)
        save_snapshot(model, data, args.snapshot)
        print(f"Saved snapshot to: {args.snapshot}")
        if args.dump_xml is not None:
            print(f"Saved combined MJCF to: {args.dump_xml}")
        return

    if args.dump_xml is not None:
        print(f"Saved combined MJCF to: {args.dump_xml}")
    print(f"Arm resource : {PANDA_XML_PATH}")
    print(f"Hand resource: {INSPIRE_XML_PATHS[args.hand]}")
    print(
        f"Actuator count: {model.nu} "
        f"(arm {ARM_ACTUATOR_COUNT} + hand {model.nu - ARM_ACTUATOR_COUNT})"
    )
    print("Close the viewer window to exit.")
    run_viewer(model, data, demo=args.mode == "demo")


if __name__ == "__main__":
    main()
