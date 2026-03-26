#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

from generate_grasp_candidates import SELECTED_HAND_POINT_RADIUS, SELECTED_OBJECT_POINT_COLOR, SELECTED_OBJECT_POINT_RADIUS
from view_franka_inspire import (
    PICK_CYLINDER_HALF_HEIGHT,
    PICK_CYLINDER_POS,
    PICK_CYLINDER_RADIUS,
    _load_hand_spec,
    build_model,
)
from view_surface_points import HAND_ROLE_COLORS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CANDIDATE_JSON = PROJECT_ROOT / "generated" / "grasp_candidates_right.json"
FLOOR_RGBA = np.array([0.92, 0.93, 0.95, 1.0], dtype=float)


def _json_vec3(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"Expected a length-3 vector, got shape {array.shape}.")
    return array


def _json_vec(values: list[float], length: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (length,):
        raise ValueError(f"Expected a length-{length} vector, got shape {array.shape}.")
    return array


def _add_marker_geom_to_body(
    body: mujoco.MjsBody,
    name: str,
    pos: np.ndarray,
    radius: float,
    rgba: np.ndarray,
) -> None:
    geom = body.add_geom()
    geom.name = name
    geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
    geom.pos = pos.astype(float)
    geom.size = np.array([radius, 0.0, 0.0], dtype=float)
    geom.contype = 0
    geom.conaffinity = 0
    geom.group = 2
    geom.rgba = rgba.astype(float)


def _configure_candidate_camera(cam: mujoco.MjvCamera, object_center: np.ndarray) -> None:
    cam.lookat = object_center.astype(float)
    cam.distance = 0.72
    cam.azimuth = 148.0
    cam.elevation = -20.0


def _initialize_static_state(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    if model.nu > 0:
        zero_ctrl = np.zeros(model.nu, dtype=float)
        ctrl_range = model.actuator_ctrlrange
        data.ctrl[:] = np.clip(zero_ctrl, ctrl_range[:, 0], ctrl_range[:, 1])
    mujoco.mj_forward(model, data)


def _apply_hand_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_qpos: np.ndarray | None,
    hand_ctrl: np.ndarray | None,
) -> None:
    qpos = model.qpos0.copy()
    if hand_qpos is not None:
        qpos[:] = hand_qpos.astype(float)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    if model.nu > 0:
        ctrl_range = model.actuator_ctrlrange
        ctrl = np.zeros(model.nu, dtype=float) if hand_ctrl is None else hand_ctrl.astype(float)
        data.ctrl[:] = np.clip(ctrl, ctrl_range[:, 0], ctrl_range[:, 1])
    mujoco.mj_forward(model, data)


def _load_candidate_payload(candidate_json: Path) -> dict[str, Any]:
    return json.loads(candidate_json.read_text(encoding="utf-8"))


def _find_candidate(payload: dict[str, Any], rank: int) -> dict[str, Any]:
    candidates = payload["candidates"]
    for candidate in candidates:
        if int(candidate["rank"]) == rank:
            return candidate
    raise ValueError(f"Candidate rank {rank} was not found in {len(candidates)} saved candidates.")


def _candidate_hand_side(payload: dict[str, Any]) -> str:
    return str(payload["config"]["scene"]["hand"])


def _candidate_object_body_name(payload: dict[str, Any]) -> str:
    return str(payload["config"]["scene"].get("object_body_name", "pickup_cylinder"))


def _candidate_alignment_translation(candidate: dict[str, Any], align_mode: str) -> np.ndarray:
    if align_mode == "none":
        return np.zeros(3, dtype=float)
    if align_mode != "mean_translation":
        raise ValueError(f"Unsupported align mode: {align_mode}")

    deltas = [
        _json_vec3(contact["object_world_pos"]) - _json_vec3(contact["hand_world_pos"])
        for contact in candidate["contacts"]
    ]
    return np.mean(np.stack(deltas, axis=0), axis=0)


def _candidate_alignment_rms(candidate: dict[str, Any], translation: np.ndarray) -> float:
    errors = []
    for contact in candidate["contacts"]:
        hand_world_pos = _json_vec3(contact["hand_world_pos"]) + translation
        object_world_pos = _json_vec3(contact["object_world_pos"])
        errors.append(np.linalg.norm(object_world_pos - hand_world_pos))
    return float(np.sqrt(np.mean(np.square(errors))))


def _reference_hand_base_pose(side: str) -> tuple[np.ndarray, np.ndarray]:
    _, model, data = build_model(side, None)
    body_name = f"inspire_{side}_hand_base"
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Reference hand base body '{body_name}' was not found.")
    return data.xpos[body_id].copy(), data.xquat[body_id].copy()


def _candidate_root_pose(
    side: str,
    candidate: dict[str, Any],
    align_mode: str,
    trace_step: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trace_entry = _candidate_trace_entry(candidate, trace_step)
    if trace_entry is not None:
        return (
            _json_vec3(trace_entry["hand_root_pos"]),
            _json_vec(trace_entry["hand_root_quat"], 4),
            np.zeros(3, dtype=float),
        )
    if "hand_root_pos" in candidate and "hand_root_quat" in candidate:
        return (
            _json_vec3(candidate["hand_root_pos"]),
            np.asarray(candidate["hand_root_quat"], dtype=float),
            np.zeros(3, dtype=float),
        )

    translation = _candidate_alignment_translation(candidate, align_mode=align_mode)
    reference_pos, reference_quat = _reference_hand_base_pose(side)
    return reference_pos + translation, reference_quat, translation


def _candidate_hand_state(candidate: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    hand_qpos = None
    hand_ctrl = None
    if "hand_qpos" in candidate:
        hand_qpos = np.asarray(candidate["hand_qpos"], dtype=float)
    if "hand_ctrl" in candidate:
        hand_ctrl = np.asarray(candidate["hand_ctrl"], dtype=float)
    return hand_qpos, hand_ctrl


def _candidate_trace_entry(candidate: dict[str, Any], trace_step: int | None) -> dict[str, Any] | None:
    if trace_step is None:
        return None
    trace = candidate.get("optimization_trace")
    if not trace:
        return None

    step_index = trace_step
    if step_index < 0:
        step_index = len(trace) + step_index
    step_index = int(np.clip(step_index, 0, len(trace) - 1))
    return trace[step_index]


def build_candidate_env_model(
    side: str,
    object_body_name: str,
    candidate: dict[str, Any],
    align_mode: str,
    trace_step: int | None,
) -> tuple[mujoco.MjModel, mujoco.MjData, np.ndarray, np.ndarray, float]:
    hand_root_pos, hand_root_quat, translation = _candidate_root_pose(
        side,
        candidate,
        align_mode=align_mode,
        trace_step=trace_step,
    )
    trace_entry = _candidate_trace_entry(candidate, trace_step)
    if trace_entry is not None:
        hand_qpos = _json_vec(trace_entry["hand_qpos"], len(trace_entry["hand_qpos"]))
        hand_ctrl = hand_qpos.copy()
    else:
        hand_qpos, hand_ctrl = _candidate_hand_state(candidate)
    rms_error = _candidate_alignment_rms(candidate, translation)

    spec = mujoco.MjSpec()
    spec.modelname = f"grasp_candidate_env_{side}_rank_{candidate['rank']}"
    spec.stat.center = np.array([0.58, 0.02, 0.33], dtype=float)
    spec.stat.extent = 0.9
    spec.visual.global_.offwidth = 1600
    spec.visual.global_.offheight = 1200

    light = spec.worldbody.add_light()
    light.name = "key_light"
    light.pos = np.array([0.3, -0.6, 1.3], dtype=float)
    light.dir = np.array([0.0, 0.2, -1.0], dtype=float)
    light.diffuse = np.array([0.95, 0.95, 0.95], dtype=float)
    light.specular = np.array([0.25, 0.25, 0.25], dtype=float)
    light.castshadow = True

    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = np.array([0.0, 0.0, 0.05], dtype=float)
    floor.pos = np.array([0.0, 0.0, 0.0], dtype=float)
    floor.rgba = FLOOR_RGBA.copy()
    floor.friction = np.array([1.0, 0.05, 0.01], dtype=float)

    object_body = spec.worldbody.add_body()
    object_body.name = object_body_name
    object_body.pos = PICK_CYLINDER_POS.copy()
    object_geom = object_body.add_geom()
    object_geom.name = "pickup_cylinder_geom"
    object_geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    object_geom.size = np.array([PICK_CYLINDER_RADIUS, PICK_CYLINDER_HALF_HEIGHT, 0.0], dtype=float)
    object_geom.condim = 4
    object_geom.friction = np.array([1.1, 0.05, 0.01], dtype=float)
    object_geom.rgba = np.array([0.91, 0.58, 0.19, 1.0], dtype=float)

    hand_spec = _load_hand_spec(side)
    hand_root = hand_spec.body(f"{side}_hand_base")
    hand_root.pos = hand_root_pos.copy()
    hand_root.quat = hand_root_quat.copy()

    mount_site = spec.worldbody.add_site()
    mount_site.name = "hand_mount_site"
    mount_site.pos = np.zeros(3, dtype=float)
    mount_site.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    spec.attach(hand_spec, prefix="inspire_", site=mount_site)

    object_body = spec.body(object_body_name)
    for contact_index, contact in enumerate(candidate["contacts"]):
        hand_body = spec.body(contact["body_name"])
        hand_local_pos = _json_vec3(contact["hand_local_pos"])
        object_local_pos = _json_vec3(contact["object_local_pos"])
        role = str(contact["role"])

        _add_marker_geom_to_body(
            hand_body,
            name=f"candidate_hand_contact_{contact_index:02d}",
            pos=hand_local_pos,
            radius=SELECTED_HAND_POINT_RADIUS,
            rgba=HAND_ROLE_COLORS[role],
        )
        _add_marker_geom_to_body(
            object_body,
            name=f"candidate_object_contact_{contact_index:02d}",
            pos=object_local_pos,
            radius=SELECTED_OBJECT_POINT_RADIUS,
            rgba=SELECTED_OBJECT_POINT_COLOR,
        )

    model = spec.compile()
    data = mujoco.MjData(model)
    _apply_hand_state(model, data, hand_qpos=hand_qpos, hand_ctrl=hand_ctrl)

    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)
    object_center = data.xpos[object_body_id].copy()
    return model, data, translation, object_center, rms_error


def save_snapshot(model: mujoco.MjModel, data: mujoco.MjData, object_center: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=720, width=960)
    camera = mujoco.MjvCamera()
    _configure_candidate_camera(camera, object_center)
    renderer.update_scene(data, camera=camera)
    Image.fromarray(renderer.render()).save(output_path)
    renderer.close()


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData, object_center: np.ndarray) -> None:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        _configure_candidate_camera(viewer.cam, object_center)
        while viewer.is_running():
            viewer.sync()


def _contact_labels(candidate: dict[str, Any]) -> str:
    labels = [
        f"{contact['point_index']}:{contact['finger']}_{contact['role']}"
        for contact in candidate["contacts"]
    ]
    return ", ".join(labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a grasp candidate in a hand-only environment with a fixed object."
    )
    parser.add_argument("--candidate-json", type=Path, default=DEFAULT_CANDIDATE_JSON)
    parser.add_argument("--rank", type=int, default=1, help="1-based candidate rank to visualize.")
    parser.add_argument(
        "--align-mode",
        choices=("mean_translation", "none"),
        default="mean_translation",
        help="How to apply the candidate to the hand-only environment.",
    )
    parser.add_argument(
        "--trace-step",
        type=int,
        default=None,
        help="Render a specific optimization trace step if the candidate stores optimization_trace.",
    )
    parser.add_argument("--snapshot", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_json = args.candidate_json if args.candidate_json.is_absolute() else PROJECT_ROOT / args.candidate_json
    payload = _load_candidate_payload(candidate_json)
    side = _candidate_hand_side(payload)
    object_body_name = _candidate_object_body_name(payload)
    candidate = _find_candidate(payload, rank=args.rank)
    model, data, translation, object_center, rms_error = build_candidate_env_model(
        side=side,
        object_body_name=object_body_name,
        candidate=candidate,
        align_mode=args.align_mode,
        trace_step=args.trace_step,
    )

    print(f"Candidate JSON      : {candidate_json}")
    print(f"Candidate rank      : {candidate['rank']}")
    print(f"Hand side           : {side}")
    print(f"Score               : {candidate['score']:.6f}")
    print(
        "Loss terms          : "
        f"e_dis={candidate['e_dis']:.6f}, "
        f"e_tq={candidate['e_tq']:.6f}, "
        f"e_align={candidate.get('e_align', 0.0):.6f}, "
        f"e_palm={candidate.get('e_palm', 0.0):.6f}, "
        f"e_qpos={candidate.get('e_qpos', 0.0):.6f}, "
        f"e_pen={candidate['e_pen']:.1f}"
    )
    print(f"Alignment mode      : {args.align_mode}")
    if args.trace_step is not None:
        print(f"Trace step          : {args.trace_step}")
    print(f"Hand translation    : [{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}]")
    print(f"Residual RMS        : {rms_error:.6f} m")
    print(f"Contacts            : {_contact_labels(candidate)}")

    if args.snapshot is not None:
        snapshot_path = args.snapshot if args.snapshot.is_absolute() else PROJECT_ROOT / args.snapshot
        save_snapshot(model, data, object_center, snapshot_path)
        print(f"Saved snapshot      : {snapshot_path}")
        return

    print("Close the viewer window to exit.")
    run_viewer(model, data, object_center)


if __name__ == "__main__":
    main()
