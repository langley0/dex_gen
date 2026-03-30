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

from grasp_gen.prop import Prop
from grasp_gen.prop_assets import make_drill_prop
from grasp_gen.prop_sdf import PropSDFConfig, build_prop_sdf_grid, build_sdf_surface_mesh


DRILL_GROUP = 0
SDF_GROUP = 3


def _bright_bg(model: mujoco.MjModel) -> None:
    model.vis.rgba.haze[:] = np.array([0.97, 0.97, 0.99, 1.0], dtype=float)
    model.vis.rgba.fog[:] = np.array([0.97, 0.97, 0.99, 1.0], dtype=float)
    model.vis.headlight.ambient[:] = np.array([0.55, 0.55, 0.55], dtype=float)
    model.vis.headlight.diffuse[:] = np.array([0.85, 0.85, 0.85], dtype=float)
    model.vis.headlight.specular[:] = np.array([0.15, 0.15, 0.15], dtype=float)


def _camera(cam: mujoco.MjvCamera) -> None:
    cam.lookat[:] = np.zeros(3, dtype=float)
    cam.distance = 0.48
    cam.azimuth = 140.0
    cam.elevation = -18.0


def _geomgroup(mode: str) -> np.ndarray:
    if mode == "object":
        return np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    if mode == "sdf":
        return np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
    return np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8)


def _build_scene(voxel_size: float, padding: float) -> tuple[mujoco.MjModel, mujoco.MjData, int, int]:
    drill, _ = make_drill_prop()
    sdf_grid = build_prop_sdf_grid(
        drill,
        cfg=PropSDFConfig(voxel_size=float(voxel_size), padding=float(padding)),
    )
    sdf_vertices, sdf_faces = build_sdf_surface_mesh(sdf_grid)
    sdf_prop = Prop(
        sdf_vertices,
        sdf_faces,
        pos=drill.pos,
        quat=drill.quat,
        com_local=drill.com_local,
        name="drill_sdf",
        rgba=np.array([0.16, 0.48, 1.0, 0.28], dtype=float),
        condim=1,
    )

    spec = mujoco.MjSpec()
    spec.modelname = "drill_sdf_view"

    _, drill_geom = drill.add_to(spec, body_name="drill_body", geom_name="drill_geom", mesh_name="drill_mesh")
    drill_geom.group = DRILL_GROUP
    drill_geom.contype = 0
    drill_geom.conaffinity = 0

    _, sdf_geom = sdf_prop.add_to(spec, body_name="drill_sdf_body", geom_name="drill_sdf_geom", mesh_name="drill_sdf_mesh")
    sdf_geom.group = SDF_GROUP
    sdf_geom.contype = 0
    sdf_geom.conaffinity = 0

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, sdf_vertices.shape[0], sdf_faces.shape[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View the drill mesh and its SDF surface mesh by geom group.")
    parser.add_argument("--show", choices=("both", "object", "sdf"), default="both")
    parser.add_argument("--sdf-voxel-size", type=float, default=3.0e-3)
    parser.add_argument("--sdf-padding", type=float, default=1.0e-2)
    parser.add_argument("--bright-bg", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sdf_voxel_size <= 0.0:
        raise SystemExit("--sdf-voxel-size must be positive.")
    if args.sdf_padding < 0.0:
        raise SystemExit("--sdf-padding must be non-negative.")

    model, data, vertex_count, face_count = _build_scene(args.sdf_voxel_size, args.sdf_padding)
    if args.bright_bg:
        _bright_bg(model)

    print(f"groups: drill={DRILL_GROUP}, sdf={SDF_GROUP}")
    print(f"sdf mesh: vertices={vertex_count}, faces={face_count}")
    print(f"show mode: {args.show}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        _camera(viewer.cam)
        viewer.opt.geomgroup[:] = _geomgroup(args.show)
        while viewer.is_running():
            viewer.sync()
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
