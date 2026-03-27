#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from generate_grasp_candidates import (
    PROJECT_ROOT,
    CYLINDER_RGBA,
    FLOOR_RGBA,
    HandActuatorSpec,
    HandInitConfig,
    HandPointProbe,
    HandPoseConfig,
    HandPoseSample,
    GeneMetrics,
    LossConfig,
    OutputConfig,
    SceneConfig,
    SELECTED_HAND_POINT_RADIUS,
    SELECTED_OBJECT_POINT_COLOR,
    SELECTED_OBJECT_POINT_RADIUS,
    _normalize,
    _path_from_config,
    _smoothstep_unit,
    build_hand_only_scene,
    closest_surface_on_object,
    compute_finger_qpos_penalties,
    describe_hand_actuators,
    parse_hand_init_config,
    parse_qpos_excluded_joint_names,
    sample_initial_pose_sequence,
    scene_penetration_depth,
)
from view_surface_points import HAND_ROLE_COLORS, SEGMENT_SPECS, compute_finger_surface_point_records
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "grasp_data_search.toml"
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
HAND_COLLISION_GEOMGROUP = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
PATH_BLOCK_RAY_EPS = 1.0e-4


@dataclass(frozen=True)
class ContactConfig:
    contact_count: int
    switch_possibility: float


@dataclass(frozen=True)
class RunConfig:
    random_seed: int
    candidate_limit: int


@dataclass(frozen=True)
class OptimizeConfig:
    max_steps: int
    starting_temperature: float
    temperature_decay: float
    annealing_period: int
    step_size: float
    stepsize_period: int
    mu: float
    position_epsilon: float
    quat_epsilon: float
    qpos_epsilon: float
    gradient_clip: float
    root_position_margin: float
    trace_stride: int
    log_period: int


@dataclass(frozen=True)
class OptimizerConfig:
    scene: SceneConfig
    hand_pose: HandPoseConfig
    hand_init: HandInitConfig
    contacts: ContactConfig
    run: RunConfig
    optimize: OptimizeConfig
    loss: LossConfig
    output: OutputConfig

    @property
    def search(self) -> RunConfig:
        return self.run


@dataclass(frozen=True)
class HandState:
    root_pos: np.ndarray
    root_quat: np.ndarray
    hand_qpos: np.ndarray
    hand_ctrl: np.ndarray


@dataclass
class OptimizationScene:
    model: mujoco.MjModel
    data: mujoco.MjData
    root_body_id: int
    palm_site_id: int
    object_body_id: int
    object_geom_ids: tuple[int, ...]
    point_records: list[dict[str, Any]]
    point_body_ids: tuple[int, ...]
    point_local_positions: np.ndarray
    point_local_normals: np.ndarray
    actuator_specs: list[HandActuatorSpec]
    qpos_lower: np.ndarray
    qpos_upper: np.ndarray
    object_center: np.ndarray
    object_geom_type: int | None
    object_geom_size: np.ndarray | None
    object_geom_pos: np.ndarray | None
    object_geom_rot: np.ndarray | None
    initial_state: HandState


@dataclass(frozen=True)
class AnnealingOptimizationResult:
    contact_indices: tuple[int, ...]
    initial_contact_indices: tuple[int, ...]
    final_state: HandState
    metrics: GeneMetrics
    contacts: list[HandPointProbe]
    trace: list[dict[str, Any]]
    optimize_steps: int
    optimize_stop_reason: str
    fingers: tuple[str, ...]
    accepted_steps: int
    rejected_steps: int
    final_temperature: float
    final_step_size: float


@dataclass(frozen=True)
class PoseSearchResult:
    pose_sample: HandPoseSample
    result: AnnealingOptimizationResult
    step_stats: list[dict[str, Any]]


def load_optimizer_config(config_path: Path) -> OptimizerConfig:
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    scene_raw = raw["scene"]
    hand_pose_raw = raw["hand_pose"]
    hand_init_raw = raw["hand_init"]
    contacts_raw = raw.get("contacts", raw.get("gene", {}))
    run_raw = raw.get("run", raw.get("search", {}))
    optimize_raw = raw["optimize"]
    loss_raw = raw["loss"]
    output_raw = raw["output"]

    scene = SceneConfig(
        hand=str(scene_raw["hand"]),
        hand_points_per_joint=int(scene_raw["hand_points_per_joint"]),
        object_body_name=str(scene_raw["object_body_name"]),
        object_geom_names=tuple(str(name) for name in scene_raw["object_geom_names"]),
    )
    hand_pose = HandPoseConfig(
        sample_count=int(hand_pose_raw["sample_count"]),
        surface_offset=float(hand_pose_raw["surface_offset"]),
        surface_candidate_count=int(hand_pose_raw.get("surface_candidate_count", 32)),
        roll_search_steps=int(hand_pose_raw.get("roll_search_steps", 72)),
        distance_std_weight=float(hand_pose_raw.get("distance_std_weight", 0.5)),
        distance_max_weight=float(hand_pose_raw.get("distance_max_weight", 0.0)),
        accessible_normal_min_z=float(hand_pose_raw.get("accessible_normal_min_z", -0.25)),
        thumb_contact_weight=float(hand_pose_raw.get("thumb_contact_weight", 4.0)),
    )
    hand_init = parse_hand_init_config(hand_init_raw)
    contacts = ContactConfig(
        contact_count=int(contacts_raw.get("contact_count", contacts_raw.get("min_contact_count", 4))),
        switch_possibility=float(contacts_raw.get("switch_possibility", contacts_raw.get("mutation_probability", 0.5))),
    )
    run = RunConfig(
        random_seed=int(run_raw.get("random_seed", 7)),
        candidate_limit=int(run_raw.get("candidate_limit", output_raw.get("candidate_limit", hand_pose.sample_count))),
    )
    optimize = OptimizeConfig(
        max_steps=int(optimize_raw["max_steps"]),
        starting_temperature=float(optimize_raw.get("starting_temperature", 1.0)),
        temperature_decay=float(optimize_raw.get("temperature_decay", 0.95)),
        annealing_period=int(optimize_raw.get("annealing_period", 50)),
        step_size=float(optimize_raw.get("step_size", 0.02)),
        stepsize_period=int(optimize_raw.get("stepsize_period", 50)),
        mu=float(optimize_raw.get("mu", 0.1)),
        position_epsilon=float(optimize_raw["position_epsilon"]),
        quat_epsilon=float(optimize_raw["quat_epsilon"]),
        qpos_epsilon=float(optimize_raw["qpos_epsilon"]),
        gradient_clip=float(optimize_raw["gradient_clip"]),
        root_position_margin=float(optimize_raw["root_position_margin"]),
        trace_stride=int(optimize_raw["trace_stride"]),
        log_period=int(optimize_raw.get("log_period", 100)),
    )
    loss = LossConfig(
        distance_weight=float(loss_raw["distance_weight"]),
        torque_weight=float(loss_raw["torque_weight"]),
        penetration_weight=float(loss_raw["penetration_weight"]),
        penetration_penalty_value=float(loss_raw["penetration_penalty_value"]),
        qpos_weight=float(loss_raw["qpos_weight"]),
        qpos_edge_margin=float(loss_raw["qpos_edge_margin"]),
        qpos_excluded_joint_names=parse_qpos_excluded_joint_names(loss_raw),
        contact_alignment_weight=float(loss_raw.get("contact_alignment_weight", 0.0)),
        contact_alignment_min_dot=float(loss_raw.get("contact_alignment_min_dot", 0.0)),
        palm_alignment_weight=float(loss_raw.get("palm_alignment_weight", 0.0)),
        palm_alignment_min_dot=float(loss_raw.get("palm_alignment_min_dot", -1.0)),
    )
    output = OutputConfig(
        candidate_json=_path_from_config(PROJECT_ROOT, str(output_raw["candidate_json"])),
        snapshot=_path_from_config(PROJECT_ROOT, output_raw.get("snapshot")),
    )

    config = OptimizerConfig(
        scene=scene,
        hand_pose=hand_pose,
        hand_init=hand_init,
        contacts=contacts,
        run=run,
        optimize=optimize,
        loss=loss,
        output=output,
    )
    _validate_config(config)
    return config


def _validate_config(config: OptimizerConfig) -> None:
    if config.scene.hand not in {"right", "left"}:
        raise ValueError("scene.hand must be 'right' or 'left'.")
    if config.scene.hand_points_per_joint <= 0:
        raise ValueError("scene.hand_points_per_joint must be positive.")
    if config.hand_pose.sample_count <= 0:
        raise ValueError("hand_pose.sample_count must be positive.")
    if config.hand_pose.surface_offset <= 0.0:
        raise ValueError("hand_pose.surface_offset must be positive.")
    if config.hand_pose.surface_candidate_count <= 0:
        raise ValueError("hand_pose.surface_candidate_count must be positive.")
    if config.hand_pose.roll_search_steps <= 0:
        raise ValueError("hand_pose.roll_search_steps must be positive.")
    if config.hand_pose.thumb_contact_weight <= 0.0:
        raise ValueError("hand_pose.thumb_contact_weight must be positive.")
    if config.contacts.contact_count <= 0:
        raise ValueError("contacts.contact_count must be positive.")
    if not 0.0 <= config.contacts.switch_possibility <= 1.0:
        raise ValueError("contacts.switch_possibility must be in [0, 1].")
    if config.run.candidate_limit <= 0:
        raise ValueError("run.candidate_limit must be positive.")
    if config.optimize.max_steps <= 0:
        raise ValueError("optimize.max_steps must be positive.")
    if config.optimize.starting_temperature <= 0.0:
        raise ValueError("optimize.starting_temperature must be positive.")
    if not 0.0 < config.optimize.temperature_decay <= 1.0:
        raise ValueError("optimize.temperature_decay must be in (0, 1].")
    if config.optimize.annealing_period <= 0:
        raise ValueError("optimize.annealing_period must be positive.")
    if config.optimize.step_size <= 0.0:
        raise ValueError("optimize.step_size must be positive.")
    if config.optimize.stepsize_period <= 0:
        raise ValueError("optimize.stepsize_period must be positive.")
    if not 0.0 < config.optimize.mu <= 1.0:
        raise ValueError("optimize.mu must be in (0, 1].")
    if config.optimize.gradient_clip <= 0.0:
        raise ValueError("optimize.gradient_clip must be positive.")
    if config.optimize.root_position_margin <= 0.0:
        raise ValueError("optimize.root_position_margin must be positive.")
    if config.optimize.trace_stride <= 0:
        raise ValueError("optimize.trace_stride must be positive.")
    if config.optimize.log_period <= 0:
        raise ValueError("optimize.log_period must be positive.")


def _json_array(value: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(value, dtype=float).tolist()]


def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float).copy()
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        quat /= norm
    if quat[0] < 0.0:
        quat *= -1.0
    return quat


def _state_from_pose_sample(pose_sample: HandPoseSample) -> HandState:
    return HandState(
        root_pos=np.asarray(pose_sample.root_pos, dtype=float).copy(),
        root_quat=np.asarray(pose_sample.root_quat, dtype=float).copy(),
        hand_qpos=np.asarray(pose_sample.hand_qpos, dtype=float).copy(),
        hand_ctrl=np.asarray(pose_sample.hand_ctrl, dtype=float).copy(),
    )


def _copy_state(state: HandState) -> HandState:
    return HandState(
        root_pos=state.root_pos.copy(),
        root_quat=state.root_quat.copy(),
        hand_qpos=state.hand_qpos.copy(),
        hand_ctrl=state.hand_ctrl.copy(),
    )


def _pack_state(state: HandState) -> np.ndarray:
    return np.concatenate([state.root_pos, state.root_quat, state.hand_qpos], axis=0)


def _unpack_state(vector: np.ndarray) -> HandState:
    vector = np.asarray(vector, dtype=float)
    hand_qpos = vector[7:].copy()
    return HandState(
        root_pos=vector[:3].copy(),
        root_quat=vector[3:7].copy(),
        hand_qpos=hand_qpos,
        hand_ctrl=hand_qpos.copy(),
    )


def _project_state(scene: OptimizationScene, state: HandState, config: OptimizerConfig) -> HandState:
    root_pos = state.root_pos.copy()
    offset = root_pos - scene.object_center
    offset_norm = np.linalg.norm(offset)
    if offset_norm > config.optimize.root_position_margin:
        root_pos = scene.object_center + offset / offset_norm * config.optimize.root_position_margin
    root_pos[2] = max(root_pos[2], 0.03)

    root_quat = _normalize_quaternion(state.root_quat)
    hand_qpos = np.clip(state.hand_qpos, scene.qpos_lower, scene.qpos_upper)
    hand_ctrl = hand_qpos.copy()
    return HandState(root_pos=root_pos, root_quat=root_quat, hand_qpos=hand_qpos, hand_ctrl=hand_ctrl)


def _apply_state(scene: OptimizationScene, state: HandState) -> None:
    scene.model.body_pos[scene.root_body_id] = state.root_pos.astype(float)
    scene.model.body_quat[scene.root_body_id] = _normalize_quaternion(state.root_quat)
    scene.data.qpos[:] = state.hand_qpos.astype(float)
    scene.data.qvel[:] = 0.0
    if scene.model.nu > 0:
        scene.data.ctrl[:] = np.clip(state.hand_ctrl, scene.qpos_lower, scene.qpos_upper)
    mujoco.mj_kinematics(scene.model, scene.data)
    mujoco.mj_comPos(scene.model, scene.data)


def build_optimization_scene(
    config: OptimizerConfig,
    pose_sample: HandPoseSample,
) -> OptimizationScene:
    _, model, data = build_hand_only_scene(config, pose_sample)
    hand_point_records, _ = compute_finger_surface_point_records(
        model,
        data,
        side=config.scene.hand,
        total_point_count=config.scene.hand_points_per_joint * len(SEGMENT_SPECS),
        point_count_per_segment=config.scene.hand_points_per_joint,
    )

    root_body_name = f"inspire_{config.scene.hand}_hand_base"
    root_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name))
    if root_body_id < 0:
        raise ValueError(f"Root body '{root_body_name}' was not found.")

    palm_site_name = f"inspire_{config.scene.hand}_palm"
    palm_site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, palm_site_name))
    if palm_site_id < 0:
        raise ValueError(f"Palm site '{palm_site_name}' was not found.")

    object_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.scene.object_body_name))
    if object_body_id < 0:
        raise ValueError(f"Object body '{config.scene.object_body_name}' was not found.")

    object_geom_ids = tuple(
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name))
        for geom_name in config.scene.object_geom_names
    )
    if any(geom_id < 0 for geom_id in object_geom_ids):
        raise ValueError("One or more object geom names were not found in the optimization scene.")

    point_body_ids = tuple(
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(point_record["body_name"])))
        for point_record in hand_point_records
    )
    if any(body_id < 0 for body_id in point_body_ids):
        raise ValueError("A hand point body name could not be resolved in the optimization scene.")
    point_local_positions = np.asarray(
        [np.asarray(point_record["local_pos"], dtype=float) for point_record in hand_point_records],
        dtype=float,
    )
    point_local_normals = np.asarray(
        [np.asarray(point_record["local_normal"], dtype=float) for point_record in hand_point_records],
        dtype=float,
    )

    actuator_specs = describe_hand_actuators(model)
    qpos_lower = np.array([spec.ctrl_min for spec in actuator_specs], dtype=float)
    qpos_upper = np.array([spec.ctrl_max for spec in actuator_specs], dtype=float)

    object_geom_type: int | None = None
    object_geom_size: np.ndarray | None = None
    object_geom_pos: np.ndarray | None = None
    object_geom_rot: np.ndarray | None = None
    if len(object_geom_ids) == 1:
        geom_id = object_geom_ids[0]
        geom_type = int(model.geom_type[geom_id])
        if geom_type in {
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
            int(mujoco.mjtGeom.mjGEOM_BOX),
            int(mujoco.mjtGeom.mjGEOM_SPHERE),
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        }:
            object_geom_type = geom_type
            object_geom_size = model.geom_size[geom_id].copy()
            object_geom_pos = data.geom_xpos[geom_id].copy()
            object_geom_rot = data.geom_xmat[geom_id].reshape(3, 3).copy()

    initial_state = _state_from_pose_sample(pose_sample)
    return OptimizationScene(
        model=model,
        data=data,
        root_body_id=root_body_id,
        palm_site_id=palm_site_id,
        object_body_id=object_body_id,
        object_geom_ids=object_geom_ids,
        point_records=hand_point_records,
        point_body_ids=point_body_ids,
        point_local_positions=point_local_positions,
        point_local_normals=point_local_normals,
        actuator_specs=actuator_specs,
        qpos_lower=qpos_lower,
        qpos_upper=qpos_upper,
        object_center=data.xipos[object_body_id].copy(),
        object_geom_type=object_geom_type,
        object_geom_size=object_geom_size,
        object_geom_pos=object_geom_pos,
        object_geom_rot=object_geom_rot,
        initial_state=initial_state,
    )


def _point_finger(point_records: list[dict[str, Any]], point_index: int) -> str:
    return str(point_records[point_index]["finger"])


def _contact_fingers(contact_indices: tuple[int, ...], point_records: list[dict[str, Any]]) -> tuple[str, ...]:
    selected = {_point_finger(point_records, point_index) for point_index in contact_indices}
    return tuple(finger for finger in FINGER_ORDER if finger in selected)


def make_random_contact_indices(
    rng: np.random.Generator,
    point_count: int,
    config: ContactConfig,
) -> tuple[int, ...]:
    if point_count <= 0:
        raise ValueError("point_count must be positive.")
    selected_indices = rng.integers(0, point_count, size=config.contact_count, endpoint=False)
    return tuple(int(index) for index in selected_indices.tolist())


def resample_contact_indices(
    contact_indices: tuple[int, ...],
    point_count: int,
    rng: np.random.Generator,
    switch_possibility: float,
) -> tuple[int, ...]:
    resampled = list(contact_indices)

    for slot in range(len(resampled)):
        if rng.random() >= switch_possibility:
            continue
        resampled[slot] = int(rng.integers(0, point_count))

    return tuple(int(index) for index in resampled)


def _path_blocked_by_hand(
    scene: OptimizationScene,
    start_world_pos: np.ndarray,
    target_world_pos: np.ndarray,
) -> bool:
    path_vec = np.asarray(target_world_pos, dtype=float) - np.asarray(start_world_pos, dtype=float)
    path_length = float(np.linalg.norm(path_vec))
    if path_length <= PATH_BLOCK_RAY_EPS:
        return False

    hit_geom_id = np.array([-1], dtype=np.int32)
    ray_distance = mujoco.mj_ray(
        scene.model,
        scene.data,
        np.asarray(start_world_pos, dtype=float),
        path_vec / path_length,
        HAND_COLLISION_GEOMGROUP,
        1,
        -1,
        hit_geom_id,
        None,
    )
    return 0.0 <= float(ray_distance) < (path_length - PATH_BLOCK_RAY_EPS)


def _batch_box_signed_distances(points_local: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    q = np.abs(points_local) - half_extents[None, :]
    outside = np.maximum(q, 0.0)
    outside_distance = np.linalg.norm(outside, axis=1)
    inside_distance = np.minimum(np.max(q, axis=1), 0.0)
    return outside_distance + inside_distance


def _batch_sphere_signed_distances(points_local: np.ndarray, radius: float) -> np.ndarray:
    return np.linalg.norm(points_local, axis=1) - radius


def _batch_capsule_signed_distances(points_local: np.ndarray, radius: float, half_length: float) -> np.ndarray:
    segment_z = np.clip(points_local[:, 2], -half_length, half_length)
    delta = points_local - np.stack(
        [
            np.zeros_like(segment_z),
            np.zeros_like(segment_z),
            segment_z,
        ],
        axis=1,
    )
    return np.linalg.norm(delta, axis=1) - radius


def _batch_cylinder_signed_distances(points_local: np.ndarray, radius: float, half_height: float) -> np.ndarray:
    radial = np.linalg.norm(points_local[:, :2], axis=1)
    radial_delta = radial - radius
    top_delta = points_local[:, 2] - half_height
    bottom_delta = -half_height - points_local[:, 2]

    outside = np.stack(
        [
            np.maximum(radial_delta, 0.0),
            np.maximum(top_delta, 0.0),
            np.maximum(bottom_delta, 0.0),
        ],
        axis=1,
    )
    signed = np.linalg.norm(outside, axis=1)
    inside_mask = (radial_delta <= 0.0) & (top_delta <= 0.0) & (bottom_delta <= 0.0)
    if np.any(inside_mask):
        signed[inside_mask] = -np.minimum.reduce(
            [
                radius - radial[inside_mask],
                half_height - points_local[inside_mask, 2],
                half_height + points_local[inside_mask, 2],
            ]
        )
    return signed


def _distance_only_contact_metrics_fast(
    scene: OptimizationScene,
    contact_indices: tuple[int, ...],
) -> tuple[np.ndarray, bool] | None:
    if (
        scene.object_geom_type is None
        or scene.object_geom_size is None
        or scene.object_geom_pos is None
        or scene.object_geom_rot is None
    ):
        return None
    if len(contact_indices) == 0:
        return np.zeros(0, dtype=float), False

    contact_index_array = np.asarray(contact_indices, dtype=np.intp)
    body_index_array = np.asarray(scene.point_body_ids, dtype=np.intp)[contact_index_array]
    body_positions = scene.data.xpos[body_index_array]
    body_rotations = scene.data.xmat[body_index_array].reshape(-1, 3, 3)
    local_positions = scene.point_local_positions[contact_index_array]
    world_positions = body_positions + np.einsum("nij,nj->ni", body_rotations, local_positions)
    points_local = (world_positions - scene.object_geom_pos[None, :]) @ scene.object_geom_rot

    geom_type = scene.object_geom_type
    geom_size = scene.object_geom_size
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        signed_distances = _batch_cylinder_signed_distances(
            points_local,
            radius=float(geom_size[0]),
            half_height=float(geom_size[1]),
        )
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        signed_distances = _batch_box_signed_distances(points_local, geom_size[:3])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        signed_distances = _batch_sphere_signed_distances(points_local, radius=float(geom_size[0]))
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        signed_distances = _batch_capsule_signed_distances(
            points_local,
            radius=float(geom_size[0]),
            half_length=float(geom_size[1]),
        )
    else:
        return None

    return np.abs(signed_distances), bool(np.any(signed_distances < 0.0))


def _current_contact_probes(
    scene: OptimizationScene,
    contact_indices: tuple[int, ...],
    object_body_pos: np.ndarray,
    object_body_rot: np.ndarray,
    object_center: np.ndarray,
    include_contacts: bool,
) -> tuple[list[HandPointProbe], np.ndarray, np.ndarray, np.ndarray, bool, int, np.ndarray]:
    contacts: list[HandPointProbe] = []
    distances = []
    signed_distances = []
    torque_vectors = []
    force_vectors = []
    contact_alignment_dots = []
    blocked_path_count = 0

    for point_index in contact_indices:
        point_record = scene.point_records[point_index]
        body_id = scene.point_body_ids[point_index]
        body_pos = scene.data.xpos[body_id].copy()
        body_rot = scene.data.xmat[body_id].reshape(3, 3)

        local_pos = np.asarray(point_record["local_pos"], dtype=float)
        local_normal = np.asarray(point_record["local_normal"], dtype=float)
        world_pos = body_pos + body_rot @ local_pos
        world_normal = _normalize(body_rot @ local_normal)

        query = closest_surface_on_object(
            scene.model,
            scene.data,
            scene.object_geom_ids,
            world_pos,
        )
        path_blocked = _path_blocked_by_hand(scene, world_pos, query.world_pos)
        if path_blocked:
            blocked_path_count += 1
        contact_vec = query.world_pos - world_pos
        contact_vec_norm = np.linalg.norm(contact_vec)
        if contact_vec_norm > 1e-9:
            contact_alignment_dots.append(float(np.dot(world_normal, contact_vec / contact_vec_norm)))
        else:
            contact_alignment_dots.append(1.0)
        force_vector = -query.world_normal
        torque_vector = np.cross(query.world_pos - object_center, force_vector)
        distances.append(float(np.linalg.norm(world_pos - query.world_pos)))
        signed_distances.append(float(query.signed_distance))
        torque_vectors.append(torque_vector)
        force_vectors.append(force_vector)

        if include_contacts:
            object_local_pos = object_body_rot.T @ (query.world_pos - object_body_pos)
            contacts.append(
                HandPointProbe(
                    point_index=point_index,
                    finger=str(point_record["finger"]),
                    segment=str(point_record["segment"]),
                    role=str(point_record["role"]),
                    body_name=str(point_record["body_name"]),
                    local_pos=local_pos,
                    local_normal=local_normal,
                    world_pos=world_pos,
                    world_normal=world_normal,
                    object_geom_name=query.geom_name,
                    object_body_name=query.body_name,
                    object_world_pos=query.world_pos,
                    object_world_normal=query.world_normal,
                    object_local_pos=object_local_pos,
                    distance=distances[-1],
                    signed_distance=signed_distances[-1],
                    torque_vector=torque_vector,
                    force_vector=force_vector,
                    path_blocked=path_blocked,
                )
            )

    return (
        contacts,
        np.asarray(distances, dtype=float),
        np.asarray(torque_vectors, dtype=float),
        np.asarray(force_vectors, dtype=float),
        any(value < 0.0 for value in signed_distances),
        blocked_path_count,
        np.asarray(contact_alignment_dots, dtype=float),
    )


def calculate_energy(
    scene: OptimizationScene,
    contact_indices: tuple[int, ...],
    state: HandState,
    config: OptimizerConfig,
    include_contacts: bool = False,
) -> tuple[GeneMetrics, list[HandPointProbe]]:
    projected_state = _project_state(scene, state, config)
    _apply_state(scene, projected_state)

    if not include_contacts:
        fast_result = _distance_only_contact_metrics_fast(scene, contact_indices)
        if fast_result is not None:
            distances, selected_penetration = fast_result
            e_dis = float(np.sum(distances))
            score = float(config.loss.distance_weight * e_dis)
            metrics = GeneMetrics(
                point_indices=contact_indices,
                score=score,
                e_dis=e_dis,
                e_tq=0.0,
                e_pen=0.0,
                e_qpos=0.0,
                e_force=0.0,
                scene_penetration_depth=0.0,
                selected_penetration=bool(selected_penetration),
                selected_path_blocked_count=0,
                e_align=0.0,
                e_palm=0.0,
            )
            return metrics, []

    object_center = scene.data.xipos[scene.object_body_id].copy()
    object_body_pos = scene.data.xpos[scene.object_body_id].copy()
    object_body_rot = scene.data.xmat[scene.object_body_id].reshape(3, 3)
    contacts, distances, _, _, selected_penetration, blocked_path_count, _ = _current_contact_probes(
        scene,
        contact_indices,
        object_body_pos=object_body_pos,
        object_body_rot=object_body_rot,
        object_center=object_center,
        include_contacts=include_contacts,
    )
    e_dis = float(np.sum(distances))
    score = float(config.loss.distance_weight * e_dis)
    metrics = GeneMetrics(
        point_indices=contact_indices,
        score=score,
        e_dis=e_dis,
        e_tq=0.0,
        e_pen=0.0,
        e_qpos=0.0,
        e_force=0.0,
        scene_penetration_depth=0.0,
        selected_penetration=bool(selected_penetration),
        selected_path_blocked_count=int(blocked_path_count),
        e_align=0.0,
        e_palm=0.0,
    )
    return metrics, contacts


def _state_trace_entry(
    step: int,
    state: HandState,
    contact_indices: tuple[int, ...],
    metrics: GeneMetrics,
    temperature: float,
    step_size: float,
    accepted: bool | None,
) -> dict[str, Any]:
    return {
        "step": int(step),
        "score": float(metrics.score),
        "e_dis": float(metrics.e_dis),
        "e_tq": float(metrics.e_tq),
        "e_align": float(metrics.e_align),
        "e_palm": float(metrics.e_palm),
        "e_pen": float(metrics.e_pen),
        "e_qpos": float(metrics.e_qpos),
        "blocked_path_count": int(metrics.selected_path_blocked_count),
        "point_indices": [int(value) for value in contact_indices],
        "temperature": float(temperature),
        "step_size": float(step_size),
        "accepted": None if accepted is None else bool(accepted),
        "hand_root_pos": _json_array(state.root_pos),
        "hand_root_quat": _json_array(state.root_quat),
        "hand_qpos": _json_array(state.hand_qpos),
    }


def _state_epsilons(scene: OptimizationScene, config: OptimizerConfig) -> np.ndarray:
    return np.concatenate(
        [
            np.full(3, config.optimize.position_epsilon, dtype=float),
            np.full(4, config.optimize.quat_epsilon, dtype=float),
            np.full(scene.model.nu, config.optimize.qpos_epsilon, dtype=float),
        ]
    )


def finite_difference_gradient(
    scene: OptimizationScene,
    contact_indices: tuple[int, ...],
    state: HandState,
    base_score: float,
    config: OptimizerConfig,
) -> np.ndarray:
    base_vector = _pack_state(state)
    epsilons = _state_epsilons(scene, config)
    gradient = np.zeros_like(base_vector)

    for dim_index, epsilon in enumerate(epsilons):
        trial_vector = base_vector.copy()
        trial_vector[dim_index] += epsilon
        trial_state = _project_state(scene, _unpack_state(trial_vector), config)
        projected_vector = _pack_state(trial_state)
        actual_delta = projected_vector[dim_index] - base_vector[dim_index]
        if abs(actual_delta) < 1e-9:
            continue
        trial_metrics, _ = calculate_energy(
            scene,
            contact_indices,
            trial_state,
            config,
            include_contacts=False,
        )
        gradient[dim_index] = (trial_metrics.score - base_score) / actual_delta

    grad_norm = float(np.linalg.norm(gradient))
    if grad_norm > config.optimize.gradient_clip:
        gradient *= config.optimize.gradient_clip / grad_norm
    return gradient


class AnnealingRMSPropOptimizer:
    def __init__(
        self,
        scene: OptimizationScene,
        config: OptimizerConfig,
        rng: np.random.Generator,
        initial_state: HandState,
        initial_contact_indices: tuple[int, ...],
        point_count: int,
    ) -> None:
        self.scene = scene
        self.config = config
        self.rng = rng
        self.point_count = point_count

        self.current_state = _project_state(scene, initial_state, config)
        self.current_contact_indices = tuple(sorted(initial_contact_indices))
        self.current_metrics, _ = calculate_energy(
            scene,
            self.current_contact_indices,
            self.current_state,
            config,
            include_contacts=False,
        )
        self.best_state = _copy_state(self.current_state)
        self.best_contact_indices = self.current_contact_indices
        self.best_metrics = self.current_metrics

        self.temperature = float(config.optimize.starting_temperature)
        self.step_size = float(config.optimize.step_size)
        self.ema_grad_hand_pose = np.zeros_like(_pack_state(self.current_state))

        self.old_state = _copy_state(self.current_state)
        self.old_contact_indices = self.current_contact_indices
        self.old_metrics = self.current_metrics
        self.proposed_state = _copy_state(self.current_state)
        self.proposed_contact_indices = self.current_contact_indices

        self.step_index = 0
        self.accepted_steps = 0
        self.rejected_steps = 0
        self.trace = [
            _state_trace_entry(
                step=0,
                state=self.current_state,
                contact_indices=self.current_contact_indices,
                metrics=self.current_metrics,
                temperature=self.temperature,
                step_size=self.step_size,
                accepted=None,
            )
        ]

    def try_step(self) -> tuple[HandState, tuple[int, ...]]:
        self.old_state = _copy_state(self.current_state)
        self.old_contact_indices = tuple(self.current_contact_indices)
        self.old_metrics = self.current_metrics

        gradient = finite_difference_gradient(
            self.scene,
            self.current_contact_indices,
            self.current_state,
            self.current_metrics.score,
            self.config,
        )
        self.ema_grad_hand_pose = (
            (1.0 - self.config.optimize.mu) * self.ema_grad_hand_pose
            + self.config.optimize.mu * np.square(gradient)
        )
        rms_denom = np.sqrt(self.ema_grad_hand_pose + 1.0e-8)
        trial_vector = _pack_state(self.current_state) - self.step_size * gradient / rms_denom
        self.proposed_state = _project_state(self.scene, _unpack_state(trial_vector), self.config)
        self.proposed_contact_indices = resample_contact_indices(
            self.current_contact_indices,
            point_count=self.point_count,
            rng=self.rng,
            switch_possibility=self.config.contacts.switch_possibility,
        )
        return self.proposed_state, self.proposed_contact_indices

    def _accept_probability(self, new_energy: float) -> float:
        delta = float(new_energy - self.old_metrics.score)
        if delta <= 0.0:
            return 1.0
        return float(np.exp(-delta / max(self.temperature, 1.0e-8)))

    def assess_step(self, proposed_metrics: GeneMetrics) -> bool:
        self.step_index += 1
        accept_probability = self._accept_probability(proposed_metrics.score)
        accepted = bool(accept_probability >= 1.0 or self.rng.random() < accept_probability)

        if accepted:
            self.current_state = _copy_state(self.proposed_state)
            self.current_contact_indices = tuple(self.proposed_contact_indices)
            self.current_metrics = proposed_metrics
            self.accepted_steps += 1
            if proposed_metrics.score < self.best_metrics.score - 1.0e-12:
                self.best_state = _copy_state(self.proposed_state)
                self.best_contact_indices = tuple(self.proposed_contact_indices)
                self.best_metrics = proposed_metrics
        else:
            self.current_state = _copy_state(self.old_state)
            self.current_contact_indices = tuple(self.old_contact_indices)
            self.current_metrics = self.old_metrics
            _apply_state(self.scene, self.current_state)
            self.rejected_steps += 1

        if self.step_index % self.config.optimize.annealing_period == 0:
            self.temperature = max(
                self.temperature * self.config.optimize.temperature_decay,
                1.0e-8,
            )
        if self.step_index % self.config.optimize.stepsize_period == 0:
            self.step_size *= self.config.optimize.temperature_decay

        if (
            self.step_index % self.config.optimize.trace_stride == 0
            or self.step_index == self.config.optimize.max_steps
        ):
            self.trace.append(
                _state_trace_entry(
                    step=self.step_index,
                    state=self.current_state,
                    contact_indices=self.current_contact_indices,
                    metrics=self.current_metrics,
                    temperature=self.temperature,
                    step_size=self.step_size,
                    accepted=accepted,
                )
            )
        return accepted

    def asset_step(self, proposed_metrics: GeneMetrics) -> bool:
        return self.assess_step(proposed_metrics)


def build_pose_search_results(
    config: OptimizerConfig,
) -> tuple[list[PoseSearchResult], list[dict[str, Any]]]:
    pose_results: list[PoseSearchResult] = []
    candidate_records: list[dict[str, Any]] = []

    for pose_sample in sample_initial_pose_sequence(config):
        sample_index = pose_sample.sample_index
        print(
            f"[pose {sample_index:02d}] init_dist={np.linalg.norm(pose_sample.palm_world_pos - pose_sample.anchor_world_pos):.4f} "
            f"roll={pose_sample.palm_roll_deg:.1f}deg thumb={np.rad2deg(pose_sample.hand_ctrl[0]):.1f}deg"
        )

        scene = build_optimization_scene(config, pose_sample)
        rng = np.random.default_rng(config.run.random_seed + 1009 * sample_index)
        point_count = len(scene.point_records)
        initial_contact_indices = make_random_contact_indices(rng, point_count, config.contacts)
        optimizer = AnnealingRMSPropOptimizer(
            scene=scene,
            config=config,
            rng=rng,
            initial_state=scene.initial_state,
            initial_contact_indices=initial_contact_indices,
            point_count=point_count,
        )

        step_stats: list[dict[str, Any]] = []
        for step in range(1, config.optimize.max_steps + 1):
            optimizer.try_step()
            proposed_metrics, _ = calculate_energy(
                scene,
                optimizer.proposed_contact_indices,
                optimizer.proposed_state,
                config,
                include_contacts=False,
            )
            accepted = optimizer.assess_step(proposed_metrics)
            current_fingers = _contact_fingers(optimizer.current_contact_indices, scene.point_records)

            if (
                step == 1
                or step % config.optimize.log_period == 0
                or step == config.optimize.max_steps
            ):
                print(
                    f"[pose {sample_index:02d}] step={step:04d} energy={optimizer.current_metrics.score:.6f} "
                    f"accepted={int(accepted)} temp={optimizer.temperature:.4f} "
                    f"step_size={optimizer.step_size:.5f} k={len(optimizer.current_contact_indices)} "
                    f"fingers={','.join(current_fingers)}"
                )

            if (
                step % config.optimize.trace_stride == 0
                or step == config.optimize.max_steps
            ):
                step_stats.append(
                    {
                        "step": int(step),
                        "score": float(optimizer.current_metrics.score),
                        "accepted": bool(accepted),
                        "temperature": float(optimizer.temperature),
                        "step_size": float(optimizer.step_size),
                        "contact_count": len(optimizer.current_contact_indices),
                        "fingers": list(current_fingers),
                    }
                )

        final_metrics, final_contacts = calculate_energy(
            scene,
            optimizer.best_contact_indices,
            optimizer.best_state,
            config,
            include_contacts=True,
        )
        result = AnnealingOptimizationResult(
            contact_indices=optimizer.best_contact_indices,
            initial_contact_indices=tuple(initial_contact_indices),
            final_state=optimizer.best_state,
            metrics=final_metrics,
            contacts=final_contacts,
            trace=optimizer.trace,
            optimize_steps=config.optimize.max_steps,
            optimize_stop_reason=f"max_steps({config.optimize.max_steps})",
            fingers=_contact_fingers(optimizer.best_contact_indices, scene.point_records),
            accepted_steps=optimizer.accepted_steps,
            rejected_steps=optimizer.rejected_steps,
            final_temperature=float(optimizer.temperature),
            final_step_size=float(optimizer.step_size),
        )
        pose_results.append(PoseSearchResult(pose_sample=pose_sample, result=result, step_stats=step_stats))

        candidate_records.append(
            {
                "pose_index": sample_index,
                "contact_count": len(result.contact_indices),
                "fingers": list(result.fingers),
                "hand_root_pos": _json_array(result.final_state.root_pos),
                "hand_root_quat": _json_array(result.final_state.root_quat),
                "hand_qpos": _json_array(result.final_state.hand_qpos),
                "hand_ctrl": _json_array(result.final_state.hand_ctrl),
                "initial_hand_root_pos": _json_array(pose_sample.root_pos),
                "initial_hand_root_quat": _json_array(pose_sample.root_quat),
                "initial_hand_qpos": _json_array(pose_sample.hand_qpos),
                "initial_point_indices": [int(value) for value in result.initial_contact_indices],
                "palm_world_pos": _json_array(pose_sample.palm_world_pos),
                "palm_world_normal": _json_array(pose_sample.palm_world_normal),
                "palm_roll_deg": float(pose_sample.palm_roll_deg),
                "anchor_world_pos": _json_array(pose_sample.anchor_world_pos),
                "surface_world_pos": _json_array(pose_sample.surface_world_pos),
                "surface_world_normal": _json_array(pose_sample.surface_world_normal),
                "orientation_score": float(pose_sample.orientation_score),
                "point_indices": [int(value) for value in result.contact_indices],
                "score": float(result.metrics.score),
                "e_dis": float(result.metrics.e_dis),
                "e_tq": float(result.metrics.e_tq),
                "e_align": float(result.metrics.e_align),
                "e_palm": float(result.metrics.e_palm),
                "e_pen": float(result.metrics.e_pen),
                "e_qpos": float(result.metrics.e_qpos),
                "e_force": float(result.metrics.e_force),
                "blocked_path_count": int(result.metrics.selected_path_blocked_count),
                "scene_penetration_depth": float(result.metrics.scene_penetration_depth),
                "optimization_steps": int(result.optimize_steps),
                "optimization_stop_reason": result.optimize_stop_reason,
                "accepted_steps": int(result.accepted_steps),
                "rejected_steps": int(result.rejected_steps),
                "final_temperature": float(result.final_temperature),
                "final_step_size": float(result.final_step_size),
                "optimization_trace": result.trace,
                "step_stats": step_stats,
                "contacts": [
                    {
                        "point_index": probe.point_index,
                        "finger": probe.finger,
                        "segment": probe.segment,
                        "role": probe.role,
                        "body_name": probe.body_name,
                        "hand_local_pos": _json_array(probe.local_pos),
                        "hand_local_normal": _json_array(probe.local_normal),
                        "hand_world_pos": _json_array(probe.world_pos),
                        "hand_world_normal": _json_array(probe.world_normal),
                        "object_geom_name": probe.object_geom_name,
                        "object_body_name": probe.object_body_name,
                        "object_local_pos": _json_array(probe.object_local_pos),
                        "object_world_pos": _json_array(probe.object_world_pos),
                        "object_world_normal": _json_array(probe.object_world_normal),
                        "distance": float(probe.distance),
                        "signed_distance": float(probe.signed_distance),
                        "path_blocked": bool(probe.path_blocked),
                    }
                    for probe in result.contacts
                ],
            }
        )

    return pose_results, candidate_records


def _reference_hand_model(side: str) -> mujoco.MjModel:
    from view_franka_inspire import _load_hand_spec

    return _load_hand_spec(side).compile()


def _reference_pose_transform(side: str) -> tuple[np.ndarray, np.ndarray]:
    from generate_grasp_candidates import _reference_root_to_palm_transform

    return _reference_root_to_palm_transform(side)


def save_candidate_json(
    config: OptimizerConfig,
    pose_results: list[PoseSearchResult],
    candidate_records: list[dict[str, Any]],
) -> tuple[Path, list[dict[str, Any]]]:
    sorted_candidates = sorted(candidate_records, key=lambda item: item["score"])
    limited_candidates = sorted_candidates[: config.run.candidate_limit]
    for rank, candidate in enumerate(limited_candidates, start=1):
        candidate["rank"] = rank

    payload = {
        "config": {
            "scene": {
                "hand": config.scene.hand,
                "hand_points_per_joint": config.scene.hand_points_per_joint,
                "object_body_name": config.scene.object_body_name,
                "object_geom_names": list(config.scene.object_geom_names),
            },
            "hand_pose": {
                "sample_count": config.hand_pose.sample_count,
                "surface_offset": config.hand_pose.surface_offset,
                "surface_candidate_count": config.hand_pose.surface_candidate_count,
                "roll_search_steps": config.hand_pose.roll_search_steps,
                "distance_std_weight": config.hand_pose.distance_std_weight,
                "distance_max_weight": config.hand_pose.distance_max_weight,
                "accessible_normal_min_z": config.hand_pose.accessible_normal_min_z,
                "thumb_contact_weight": config.hand_pose.thumb_contact_weight,
            },
            "hand_init": {
                "flexion_min_rad": config.hand_init.flexion_min_rad,
                "flexion_max_rad": config.hand_init.flexion_max_rad,
                "thumb_pinch_rad": config.hand_init.thumb_pinch_rad,
                "thumb_non_yaw_zero": config.hand_init.thumb_non_yaw_zero,
            },
            "contacts": {
                "contact_count": config.contacts.contact_count,
                "switch_possibility": config.contacts.switch_possibility,
            },
            "run": {
                "random_seed": config.run.random_seed,
                "candidate_limit": config.run.candidate_limit,
            },
            "optimize": {
                "max_steps": config.optimize.max_steps,
                "starting_temperature": config.optimize.starting_temperature,
                "temperature_decay": config.optimize.temperature_decay,
                "annealing_period": config.optimize.annealing_period,
                "step_size": config.optimize.step_size,
                "stepsize_period": config.optimize.stepsize_period,
                "mu": config.optimize.mu,
                "position_epsilon": config.optimize.position_epsilon,
                "quat_epsilon": config.optimize.quat_epsilon,
                "qpos_epsilon": config.optimize.qpos_epsilon,
                "gradient_clip": config.optimize.gradient_clip,
                "root_position_margin": config.optimize.root_position_margin,
                "trace_stride": config.optimize.trace_stride,
                "log_period": config.optimize.log_period,
            },
            "loss": {
                "distance_weight": config.loss.distance_weight,
                "torque_weight": config.loss.torque_weight,
                "penetration_weight": config.loss.penetration_weight,
                "penetration_penalty_value": config.loss.penetration_penalty_value,
                "qpos_weight": config.loss.qpos_weight,
                "qpos_edge_margin": config.loss.qpos_edge_margin,
                "qpos_excluded_joint_names": list(config.loss.qpos_excluded_joint_names),
                "contact_alignment_weight": config.loss.contact_alignment_weight,
                "contact_alignment_min_dot": config.loss.contact_alignment_min_dot,
                "palm_alignment_weight": config.loss.palm_alignment_weight,
                "palm_alignment_min_dot": config.loss.palm_alignment_min_dot,
            },
        },
        "summary": {
            "pose_sample_count": len(pose_results),
            "saved_candidate_count": len(limited_candidates),
            "hand_points_per_joint": config.scene.hand_points_per_joint,
            "hand_point_library_size_per_pose": config.scene.hand_points_per_joint * len(SEGMENT_SPECS),
            "optimizer": "simulated_annealing_rmsprop",
        },
        "pose_results": [
            {
                "pose_index": result.pose_sample.sample_index,
                "initial_hand_root_pos": _json_array(result.pose_sample.root_pos),
                "initial_hand_root_quat": _json_array(result.pose_sample.root_quat),
                "initial_hand_qpos": _json_array(result.pose_sample.hand_qpos),
                "palm_world_pos": _json_array(result.pose_sample.palm_world_pos),
                "palm_world_normal": _json_array(result.pose_sample.palm_world_normal),
                "palm_roll_deg": float(result.pose_sample.palm_roll_deg),
                "initial_point_indices": [int(value) for value in result.result.initial_contact_indices],
                "final_point_indices": [int(value) for value in result.result.contact_indices],
                "final_score": float(result.result.metrics.score),
                "optimization_steps": int(result.result.optimize_steps),
                "optimization_stop_reason": result.result.optimize_stop_reason,
                "accepted_steps": int(result.result.accepted_steps),
                "rejected_steps": int(result.result.rejected_steps),
                "final_temperature": float(result.result.final_temperature),
                "final_step_size": float(result.result.final_step_size),
                "step_stats": result.step_stats,
            }
            for result in pose_results
        ],
        "candidates": limited_candidates,
    }

    output_path = config.output.candidate_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path, limited_candidates


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


def build_best_candidate_snapshot(
    config: OptimizerConfig,
    candidate: dict[str, Any],
    output_path: Path,
) -> None:
    pose_sample = HandPoseSample(
        sample_index=int(candidate["pose_index"]),
        root_pos=np.asarray(candidate["hand_root_pos"], dtype=float),
        root_quat=np.asarray(candidate["hand_root_quat"], dtype=float),
        palm_world_pos=np.asarray(candidate["palm_world_pos"], dtype=float),
        palm_world_normal=np.asarray(candidate["palm_world_normal"], dtype=float),
        palm_roll_deg=float(candidate["palm_roll_deg"]),
        hand_qpos=np.asarray(candidate["hand_qpos"], dtype=float),
        hand_ctrl=np.asarray(candidate["hand_ctrl"], dtype=float),
        anchor_world_pos=np.asarray(candidate.get("anchor_world_pos", candidate["palm_world_pos"]), dtype=float),
        surface_world_pos=np.asarray(candidate.get("surface_world_pos", candidate["palm_world_pos"]), dtype=float),
        surface_world_normal=np.asarray(candidate.get("surface_world_normal", candidate["palm_world_normal"]), dtype=float),
        orientation_score=float(candidate.get("orientation_score", 0.0)),
    )
    spec, _, _ = build_hand_only_scene(config, pose_sample)
    object_body = spec.body(config.scene.object_body_name)
    for contact_index, contact in enumerate(candidate["contacts"]):
        hand_body = spec.body(contact["body_name"])
        role = str(contact["role"])
        _add_marker_geom_to_body(
            hand_body,
            name=f"candidate_hand_contact_{contact_index:02d}",
            pos=np.asarray(contact["hand_local_pos"], dtype=float),
            radius=SELECTED_HAND_POINT_RADIUS,
            rgba=HAND_ROLE_COLORS[role],
        )
        _add_marker_geom_to_body(
            object_body,
            name=f"candidate_object_contact_{contact_index:02d}",
            pos=np.asarray(contact["object_local_pos"], dtype=float),
            radius=SELECTED_OBJECT_POINT_RADIUS,
            rgba=SELECTED_OBJECT_POINT_COLOR,
        )

    model = spec.compile()
    data = mujoco.MjData(model)
    root_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"inspire_{config.scene.hand}_hand_base"))
    palm_site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"inspire_{config.scene.hand}_palm"))
    object_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.scene.object_body_name))
    scene = OptimizationScene(
        model=model,
        data=data,
        root_body_id=root_body_id,
        palm_site_id=palm_site_id,
        object_body_id=object_body_id,
        object_geom_ids=tuple(
            int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name))
            for geom_name in config.scene.object_geom_names
        ),
        point_records=[],
        point_body_ids=(),
        point_local_positions=np.zeros((0, 3), dtype=float),
        point_local_normals=np.zeros((0, 3), dtype=float),
        actuator_specs=describe_hand_actuators(model),
        qpos_lower=model.actuator_ctrlrange[:, 0].copy(),
        qpos_upper=model.actuator_ctrlrange[:, 1].copy(),
        object_center=np.zeros(3, dtype=float),
        object_geom_type=None,
        object_geom_size=None,
        object_geom_pos=None,
        object_geom_rot=None,
        initial_state=HandState(
            root_pos=np.asarray(candidate["hand_root_pos"], dtype=float),
            root_quat=np.asarray(candidate["hand_root_quat"], dtype=float),
            hand_qpos=np.asarray(candidate["hand_qpos"], dtype=float),
            hand_ctrl=np.asarray(candidate["hand_ctrl"], dtype=float),
        ),
    )
    _apply_state(scene, scene.initial_state)
    object_center = data.xpos[object_body_id].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=720, width=960)
    camera = mujoco.MjvCamera()
    camera.lookat = object_center
    camera.distance = 0.72
    camera.azimuth = 148.0
    camera.elevation = -20.0
    renderer.update_scene(data, camera=camera)
    Image.fromarray(renderer.render()).save(output_path)
    renderer.close()


def print_candidate_summary(candidates: list[dict[str, Any]], top_n: int) -> None:
    for candidate in candidates[:top_n]:
        fingers = ",".join(candidate["fingers"])
        labels = ", ".join(
            f"{contact['point_index']}:{contact['finger']}_{contact['role']}"
            for contact in candidate["contacts"]
        )
        print(
            f"{candidate['rank']:02d}. pose={candidate['pose_index']:02d} "
            f"k={candidate['contact_count']} fingers={fingers} "
            f"steps={candidate['optimization_steps']} "
            f"score={candidate['score']:.6f} "
            f"e_dis={candidate['e_dis']:.6f} e_tq={candidate['e_tq']:.6f} "
            f"e_align={candidate.get('e_align', 0.0):.6f} "
            f"e_palm={candidate.get('e_palm', 0.0):.6f} "
            f"e_qpos={candidate['e_qpos']:.6f} e_pen={candidate['e_pen']:.1f} "
            f"blocked={candidate.get('blocked_path_count', 0)} "
            f"contacts=[{labels}]"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate grasp data with simulated annealing over contact points and RMSProp updates for hand pose."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--print-top", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_optimizer_config(args.config)
    if args.output_json is not None:
        output_json = args.output_json if args.output_json.is_absolute() else PROJECT_ROOT / args.output_json
        config = OptimizerConfig(
            scene=config.scene,
            hand_pose=config.hand_pose,
            hand_init=config.hand_init,
            contacts=config.contacts,
            run=config.run,
            optimize=config.optimize,
            loss=config.loss,
            output=OutputConfig(candidate_json=output_json, snapshot=config.output.snapshot),
        )
    if args.snapshot is not None:
        snapshot_path = args.snapshot if args.snapshot.is_absolute() else PROJECT_ROOT / args.snapshot
        config = OptimizerConfig(
            scene=config.scene,
            hand_pose=config.hand_pose,
            hand_init=config.hand_init,
            contacts=config.contacts,
            run=config.run,
            optimize=config.optimize,
            loss=config.loss,
            output=OutputConfig(candidate_json=config.output.candidate_json, snapshot=snapshot_path),
        )

    pose_results, candidate_records = build_pose_search_results(config)
    output_json, saved_candidates = save_candidate_json(config, pose_results, candidate_records)
    total_optimize_steps = sum(result.result.optimize_steps for result in pose_results)
    total_accepted_steps = sum(result.result.accepted_steps for result in pose_results)
    total_rejected_steps = sum(result.result.rejected_steps for result in pose_results)

    print(f"Pose samples       : {len(pose_results)}")
    print(f"Hand points/joint  : {config.scene.hand_points_per_joint}")
    print(f"Annealing steps    : {total_optimize_steps}")
    print(f"Accepted steps     : {total_accepted_steps}")
    print(f"Rejected steps     : {total_rejected_steps}")
    print(f"Candidates saved   : {len(saved_candidates)}")
    print(f"Saved candidates   : {output_json}")
    print_candidate_summary(saved_candidates, top_n=max(args.print_top, 0))

    if config.output.snapshot is not None and saved_candidates:
        build_best_candidate_snapshot(config, saved_candidates[0], config.output.snapshot)
        print(f"Saved snapshot     : {config.output.snapshot}")


if __name__ == "__main__":
    main()
