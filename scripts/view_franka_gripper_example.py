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
MENAGERIE_DIR = PROJECT_ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda"
SCENE_XML_PATH = MENAGERIE_DIR / "scene.xml"
ROBOT_XML_PATH = MENAGERIE_DIR / "panda.xml"
HOME_KEY_NAME = "home"


def _resource_path(resource: str) -> Path:
    if resource == "scene":
        return SCENE_XML_PATH
    if resource == "robot":
        return ROBOT_XML_PATH
    raise ValueError(f"Unsupported resource: {resource}")


def load_model(resource: str) -> tuple[Path, mujoco.MjModel, mujoco.MjData]:
    xml_path = _resource_path(resource)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    return xml_path, model, data


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, HOME_KEY_NAME)
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def actuator_summary(model: mujoco.MjModel) -> str:
    lines = []
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        ctrl_min, ctrl_max = model.actuator_ctrlrange[actuator_id]
        lines.append(f"{actuator_id:02d}: {name} [{ctrl_min:.4f}, {ctrl_max:.4f}]")
    return "\n".join(lines)


def home_ctrl(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    ctrl = np.zeros(model.nu, dtype=float)
    if model.nu > 0:
        ctrl[:] = data.ctrl[: model.nu]
    return ctrl


def build_gripper_demo_ctrl(model: mujoco.MjModel, base_ctrl: np.ndarray, sim_time: float) -> np.ndarray:
    ctrl = base_ctrl.copy()
    if model.nu == 0:
        return ctrl

    grip_open = model.actuator_ctrlrange[-1, 1]
    grip_closed = model.actuator_ctrlrange[-1, 0]
    cycle = sim_time % 4.0

    if cycle < 1.5:
        alpha = cycle / 1.5
        grip_target = grip_open + alpha * (grip_closed - grip_open)
    elif cycle < 2.5:
        grip_target = grip_closed
    else:
        alpha = (cycle - 2.5) / 1.5
        grip_target = grip_closed + alpha * (grip_open - grip_closed)

    ctrl[-1] = grip_target
    return ctrl


def _configure_camera(cam: mujoco.MjvCamera) -> None:
    cam.lookat = np.array([0.3, 0.0, 0.42], dtype=float)
    cam.distance = 1.35
    cam.azimuth = 120.0
    cam.elevation = -20.0


def save_snapshot(model: mujoco.MjModel, data: mujoco.MjData, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera = mujoco.MjvCamera()
    _configure_camera(camera)
    renderer.update_scene(data, camera=camera)
    Image.fromarray(renderer.render()).save(output_path)
    renderer.close()


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData, demo: bool) -> None:
    timestep = model.opt.timestep
    wall_start = time.time()
    base_ctrl = home_ctrl(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        _configure_camera(viewer.cam)
        while viewer.is_running():
            elapsed = time.time() - wall_start
            if demo:
                data.ctrl[:] = build_gripper_demo_ctrl(model, base_ctrl, elapsed)
            mujoco.mj_step(model, data)
            viewer.sync()

            next_sleep = timestep - (time.time() - wall_start - elapsed)
            if next_sleep > 0.0:
                time.sleep(next_sleep)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View the existing MuJoCo Menagerie Franka Panda example with the stock gripper."
    )
    parser.add_argument(
        "--resource",
        choices=("scene", "robot"),
        default="scene",
        help="scene: load the full existing scene.xml, robot: load panda.xml only.",
    )
    parser.add_argument(
        "--mode",
        choices=("hold", "demo"),
        default="hold",
        help="hold: keep the home pose, demo: repeatedly open and close the stock gripper.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Render one snapshot and exit instead of launching the interactive viewer.",
    )
    parser.add_argument(
        "--print-actuators",
        action="store_true",
        help="Print actuator names and control ranges.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xml_path, model, data = load_model(args.resource)

    print(f"Using existing resource: {xml_path}")
    print(f"Home keyframe present: {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, HOME_KEY_NAME) >= 0}")

    if args.print_actuators:
        print(actuator_summary(model))

    if args.snapshot is not None:
        if args.mode == "demo":
            base_ctrl = home_ctrl(model, data)
            data.ctrl[:] = build_gripper_demo_ctrl(model, base_ctrl, 1.8)
            for _ in range(250):
                mujoco.mj_step(model, data)
        save_snapshot(model, data, args.snapshot)
        print(f"Saved snapshot to: {args.snapshot}")
        return

    print(f"Actuator count: {model.nu}")
    print("Close the viewer window to exit.")
    run_viewer(model, data, demo=args.mode == "demo")


if __name__ == "__main__":
    main()
