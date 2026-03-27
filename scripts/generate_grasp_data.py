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
    resolve_random_seed,
    sample_initial_pose_sequence,
)
from view_surface_points import HAND_ROLE_COLORS, SEGMENT_SPECS, compute_finger_surface_point_records
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "grasp_data_search.toml"
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
HAND_COLLISION_GEOMGROUP = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
PATH_BLOCK_RAY_EPS = 1.0e-4
PENETRATION_SURFACE_EPS = 1.0e-3


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
    point_body_index_array: np.ndarray
    point_local_positions: np.ndarray
    point_local_normals: np.ndarray
    penetration_body_index_array: np.ndarray
    penetration_local_positions: np.ndarray
    penetration_area_weights: np.ndarray
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
        hand_penetration_sample_count=int(scene_raw.get("hand_penetration_sample_count", 1000)),
        hand_point_surface_source=str(scene_raw.get("hand_point_surface_source", "visual")),
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
        random_seed=resolve_random_seed(run_raw.get("random_seed", "auto")),
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
    if config.scene.hand_point_surface_source not in {"collision", "visual"}:
        raise ValueError("scene.hand_point_surface_source must be 'collision' or 'visual'.")
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
    if config.scene.hand_penetration_sample_count < 0:
        raise ValueError("scene.hand_penetration_sample_count must be non-negative.")
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


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    matrix = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=float))
    return matrix.reshape(3, 3)


def _geom_surface_area(geom_type: int, geom_size: np.ndarray) -> float:
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        hx, hy, hz = (float(value) for value in geom_size[:3])
        return 8.0 * (hx * hy + hx * hz + hy * hz)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        radius = float(geom_size[0])
        return 4.0 * np.pi * radius * radius
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        radius = float(geom_size[0])
        half_length = float(geom_size[1])
        return 4.0 * np.pi * radius * (half_length + radius)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        radius = float(geom_size[0])
        half_height = float(geom_size[1])
        return 4.0 * np.pi * radius * half_height + 2.0 * np.pi * radius * radius
    return 0.0


def _allocate_surface_sample_counts(surface_areas: np.ndarray, total_sample_count: int) -> np.ndarray:
    counts = np.zeros(len(surface_areas), dtype=int)
    valid_indices = np.flatnonzero(surface_areas > 0.0)
    if total_sample_count <= 0 or valid_indices.size == 0:
        return counts
    if total_sample_count <= valid_indices.size:
        order = valid_indices[np.argsort(surface_areas[valid_indices])[::-1]]
        counts[order[:total_sample_count]] = 1
        return counts

    counts[valid_indices] = 1
    remaining = total_sample_count - int(valid_indices.size)
    weights = surface_areas[valid_indices] / float(np.sum(surface_areas[valid_indices]))
    scaled = remaining * weights
    extra = np.floor(scaled).astype(int)
    counts[valid_indices] += extra
    leftover = total_sample_count - int(np.sum(counts))
    if leftover > 0:
        fractional = scaled - extra
        order = valid_indices[np.argsort(fractional)[::-1]]
        counts[order[:leftover]] += 1
    return counts


def _sample_box_surface_local(
    rng: np.random.Generator,
    half_extents: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    if sample_count <= 0:
        return np.zeros((0, 3), dtype=float)

    hx, hy, hz = (float(value) for value in half_extents[:3])
    face_areas = np.array(
        [
            4.0 * hy * hz,
            4.0 * hy * hz,
            4.0 * hx * hz,
            4.0 * hx * hz,
            4.0 * hx * hy,
            4.0 * hx * hy,
        ],
        dtype=float,
    )
    face_ids = rng.choice(6, size=sample_count, p=face_areas / float(np.sum(face_areas)))
    uv = rng.uniform(-1.0, 1.0, size=(sample_count, 2))
    samples = np.zeros((sample_count, 3), dtype=float)

    mask = face_ids == 0
    samples[mask] = np.column_stack([np.full(np.sum(mask), hx), uv[mask, 0] * hy, uv[mask, 1] * hz])
    mask = face_ids == 1
    samples[mask] = np.column_stack([np.full(np.sum(mask), -hx), uv[mask, 0] * hy, uv[mask, 1] * hz])
    mask = face_ids == 2
    samples[mask] = np.column_stack([uv[mask, 0] * hx, np.full(np.sum(mask), hy), uv[mask, 1] * hz])
    mask = face_ids == 3
    samples[mask] = np.column_stack([uv[mask, 0] * hx, np.full(np.sum(mask), -hy), uv[mask, 1] * hz])
    mask = face_ids == 4
    samples[mask] = np.column_stack([uv[mask, 0] * hx, uv[mask, 1] * hy, np.full(np.sum(mask), hz)])
    mask = face_ids == 5
    samples[mask] = np.column_stack([uv[mask, 0] * hx, uv[mask, 1] * hy, np.full(np.sum(mask), -hz)])
    return samples


def _sample_sphere_surface_local(
    rng: np.random.Generator,
    radius: float,
    sample_count: int,
) -> np.ndarray:
    if sample_count <= 0:
        return np.zeros((0, 3), dtype=float)
    directions = rng.normal(size=(sample_count, 3))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= np.clip(norms, 1.0e-12, None)
    return radius * directions


def _sample_capsule_surface_local(
    rng: np.random.Generator,
    radius: float,
    half_length: float,
    sample_count: int,
) -> np.ndarray:
    if sample_count <= 0:
        return np.zeros((0, 3), dtype=float)

    cylinder_area = 4.0 * np.pi * radius * half_length
    cap_area = 4.0 * np.pi * radius * radius
    total_area = cylinder_area + cap_area
    if total_area <= 0.0:
        return np.zeros((sample_count, 3), dtype=float)

    use_cylinder = rng.random(sample_count) < (cylinder_area / total_area)
    samples = np.zeros((sample_count, 3), dtype=float)

    cylinder_count = int(np.sum(use_cylinder))
    if cylinder_count > 0:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=cylinder_count)
        z = rng.uniform(-half_length, half_length, size=cylinder_count)
        samples[use_cylinder] = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])

    cap_count = sample_count - cylinder_count
    if cap_count > 0:
        directions = rng.normal(size=(cap_count, 3))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions /= np.clip(norms, 1.0e-12, None)
        centers = np.zeros((cap_count, 3), dtype=float)
        centers[:, 2] = np.where(directions[:, 2] >= 0.0, half_length, -half_length)
        samples[~use_cylinder] = centers + radius * directions

    return samples


def _sample_cylinder_surface_local(
    rng: np.random.Generator,
    radius: float,
    half_height: float,
    sample_count: int,
) -> np.ndarray:
    if sample_count <= 0:
        return np.zeros((0, 3), dtype=float)

    side_area = 4.0 * np.pi * radius * half_height
    cap_area = 2.0 * np.pi * radius * radius
    total_area = side_area + cap_area
    if total_area <= 0.0:
        return np.zeros((sample_count, 3), dtype=float)

    use_side = rng.random(sample_count) < (side_area / total_area)
    samples = np.zeros((sample_count, 3), dtype=float)

    side_count = int(np.sum(use_side))
    if side_count > 0:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=side_count)
        z = rng.uniform(-half_height, half_height, size=side_count)
        samples[use_side] = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])

    cap_count = sample_count - side_count
    if cap_count > 0:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=cap_count)
        radial = radius * np.sqrt(rng.uniform(0.0, 1.0, size=cap_count))
        z = np.where(rng.random(cap_count) < 0.5, half_height, -half_height)
        samples[~use_side] = np.column_stack([radial * np.cos(theta), radial * np.sin(theta), z])

    return samples


def _sample_geom_surface_local(
    rng: np.random.Generator,
    geom_type: int,
    geom_size: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return _sample_box_surface_local(rng, geom_size[:3], sample_count)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return _sample_sphere_surface_local(rng, radius=float(geom_size[0]), sample_count=sample_count)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return _sample_capsule_surface_local(
            rng,
            radius=float(geom_size[0]),
            half_length=float(geom_size[1]),
            sample_count=sample_count,
        )
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return _sample_cylinder_surface_local(
            rng,
            radius=float(geom_size[0]),
            half_height=float(geom_size[1]),
            sample_count=sample_count,
        )
    return np.zeros((0, 3), dtype=float)


def _build_hand_penetration_samples(
    model: mujoco.MjModel,
    side: str,
    total_sample_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if total_sample_count <= 0:
        return (
            np.zeros(0, dtype=np.intp),
            np.zeros((0, 3), dtype=float),
            np.zeros(0, dtype=float),
        )

    geom_entries: list[tuple[int, float]] = []
    geom_prefix = f"inspire_collision_hand_{side}_"
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not geom_name.startswith(geom_prefix):
            continue
        geom_type = int(model.geom_type[geom_id])
        geom_area = _geom_surface_area(geom_type, model.geom_size[geom_id])
        if geom_area <= 0.0:
            continue
        geom_entries.append((geom_id, geom_area))

    if not geom_entries:
        return (
            np.zeros(0, dtype=np.intp),
            np.zeros((0, 3), dtype=float),
            np.zeros(0, dtype=float),
        )

    surface_areas = np.asarray([entry[1] for entry in geom_entries], dtype=float)
    geom_sample_counts = _allocate_surface_sample_counts(surface_areas, total_sample_count)
    total_surface_area = float(np.sum(surface_areas))

    body_ids: list[np.ndarray] = []
    local_positions: list[np.ndarray] = []
    area_weights: list[np.ndarray] = []
    for (geom_id, geom_area), sample_count in zip(geom_entries, geom_sample_counts, strict=True):
        if sample_count <= 0:
            continue
        geom_type = int(model.geom_type[geom_id])
        geom_local_positions = _sample_geom_surface_local(
            rng,
            geom_type=geom_type,
            geom_size=model.geom_size[geom_id],
            sample_count=int(sample_count),
        )
        if geom_local_positions.size == 0:
            continue

        geom_rot = _quat_to_matrix(model.geom_quat[geom_id])
        geom_pos = model.geom_pos[geom_id].copy()
        body_local_positions = geom_pos[None, :] + geom_local_positions @ geom_rot.T

        body_ids.append(np.full(len(body_local_positions), int(model.geom_bodyid[geom_id]), dtype=np.intp))
        local_positions.append(body_local_positions)
        normalized_weight = geom_area / max(total_surface_area, 1.0e-12) / float(len(body_local_positions))
        area_weights.append(np.full(len(body_local_positions), normalized_weight, dtype=float))

    if not local_positions:
        return (
            np.zeros(0, dtype=np.intp),
            np.zeros((0, 3), dtype=float),
            np.zeros(0, dtype=float),
        )

    return (
        np.concatenate(body_ids, axis=0),
        np.concatenate(local_positions, axis=0),
        np.concatenate(area_weights, axis=0),
    )


def _body_local_points_to_world(
    scene: OptimizationScene,
    body_index_array: np.ndarray,
    local_positions: np.ndarray,
) -> np.ndarray:
    if local_positions.size == 0:
        return np.zeros((0, 3), dtype=float)
    body_positions = scene.data.xpos[body_index_array]
    body_rotations = scene.data.xmat[body_index_array].reshape(-1, 3, 3)
    return body_positions + np.einsum("nij,nj->ni", body_rotations, local_positions)


def _batch_object_signed_distances(
    scene: OptimizationScene,
    world_positions: np.ndarray,
) -> np.ndarray | None:
    if (
        scene.object_geom_type is None
        or scene.object_geom_size is None
        or scene.object_geom_pos is None
        or scene.object_geom_rot is None
    ):
        return None
    if world_positions.size == 0:
        return np.zeros(0, dtype=float)

    points_local = (world_positions - scene.object_geom_pos[None, :]) @ scene.object_geom_rot
    geom_type = scene.object_geom_type
    geom_size = scene.object_geom_size
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return _batch_cylinder_signed_distances(
            points_local,
            radius=float(geom_size[0]),
            half_height=float(geom_size[1]),
        )
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        return _batch_box_signed_distances(points_local, geom_size[:3])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return _batch_sphere_signed_distances(points_local, radius=float(geom_size[0]))
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return _batch_capsule_signed_distances(
            points_local,
            radius=float(geom_size[0]),
            half_length=float(geom_size[1]),
        )
    return None


def _scene_penetration_metrics_fast(scene: OptimizationScene) -> tuple[float, float]:
    if scene.penetration_local_positions.size == 0:
        return 0.0, 0.0

    world_positions = _body_local_points_to_world(
        scene,
        scene.penetration_body_index_array,
        scene.penetration_local_positions,
    )
    signed_distances = _batch_object_signed_distances(scene, world_positions)
    if signed_distances is None or signed_distances.size == 0:
        return 0.0, 0.0

    penetration_depths = np.maximum(-(signed_distances + PENETRATION_SURFACE_EPS), 0.0)
    if penetration_depths.size == 0:
        return 0.0, 0.0

    max_depth = float(np.max(penetration_depths))
    penalty = float(np.sum(scene.penetration_area_weights * penetration_depths))
    return penalty, max_depth


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
        surface_source=config.scene.hand_point_surface_source,
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
    point_body_index_array = np.asarray(point_body_ids, dtype=np.intp)
    point_local_positions = np.asarray(
        [np.asarray(point_record["local_pos"], dtype=float) for point_record in hand_point_records],
        dtype=float,
    )
    point_local_normals = np.asarray(
        [np.asarray(point_record["local_normal"], dtype=float) for point_record in hand_point_records],
        dtype=float,
    )
    penetration_rng = np.random.default_rng(config.run.random_seed + 20011 * (pose_sample.sample_index + 1))
    penetration_body_index_array, penetration_local_positions, penetration_area_weights = _build_hand_penetration_samples(
        model,
        side=config.scene.hand,
        total_sample_count=config.scene.hand_penetration_sample_count,
        rng=penetration_rng,
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
        point_body_index_array=point_body_index_array,
        point_local_positions=point_local_positions,
        point_local_normals=point_local_normals,
        penetration_body_index_array=penetration_body_index_array,
        penetration_local_positions=penetration_local_positions,
        penetration_area_weights=penetration_area_weights,
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
    q = np.abs(points_local) - half_extents
    outside = np.maximum(q, 0.0)
    outside_distance = np.linalg.norm(outside, axis=-1)
    inside_distance = np.minimum(np.max(q, axis=-1), 0.0)
    return outside_distance + inside_distance


def _batch_sphere_signed_distances(points_local: np.ndarray, radius: float) -> np.ndarray:
    return np.linalg.norm(points_local, axis=-1) - radius


def _batch_capsule_signed_distances(points_local: np.ndarray, radius: float, half_length: float) -> np.ndarray:
    segment_z = np.clip(points_local[..., 2], -half_length, half_length)
    delta = points_local - np.stack(
        [
            np.zeros_like(segment_z),
            np.zeros_like(segment_z),
            segment_z,
        ],
        axis=-1,
    )
    return np.linalg.norm(delta, axis=-1) - radius


def _batch_cylinder_signed_distances(points_local: np.ndarray, radius: float, half_height: float) -> np.ndarray:
    # Equivalent capped-cylinder SDF, written without stacking temporary tensors.
    radial = np.sqrt(points_local[..., 0] * points_local[..., 0] + points_local[..., 1] * points_local[..., 1])
    radial_delta = radial - radius
    height_delta = np.abs(points_local[..., 2]) - half_height
    outside_radial = np.maximum(radial_delta, 0.0)
    outside_height = np.maximum(height_delta, 0.0)
    outside_distance = np.sqrt(outside_radial * outside_radial + outside_height * outside_height)
    inside_distance = np.minimum(np.maximum(radial_delta, height_delta), 0.0)
    return outside_distance + inside_distance


def _distance_only_contact_metrics_fast(
    scene: OptimizationScene,
    contact_indices: tuple[int, ...],
) -> tuple[np.ndarray, bool] | None:
    if len(contact_indices) == 0:
        return np.zeros(0, dtype=float), False

    contact_index_array = np.asarray(contact_indices, dtype=np.intp)
    body_index_array = scene.point_body_index_array[contact_index_array]
    local_positions = scene.point_local_positions[contact_index_array]
    world_positions = _body_local_points_to_world(scene, body_index_array, local_positions)
    signed_distances = _batch_object_signed_distances(scene, world_positions)
    if signed_distances is None:
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
            e_pen, penetration_depth = _scene_penetration_metrics_fast(scene)
            e_dis = float(np.sum(distances))
            score = float(
                config.loss.distance_weight * e_dis
                + config.loss.penetration_weight * e_pen
            )
            metrics = GeneMetrics(
                point_indices=contact_indices,
                score=score,
                e_dis=e_dis,
                e_tq=0.0,
                e_pen=e_pen,
                e_qpos=0.0,
                e_force=0.0,
                scene_penetration_depth=penetration_depth,
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
    e_pen, penetration_depth = _scene_penetration_metrics_fast(scene)
    e_dis = float(np.sum(distances))
    score = float(
        config.loss.distance_weight * e_dis
        + config.loss.penetration_weight * e_pen
    )
    metrics = GeneMetrics(
        point_indices=contact_indices,
        score=score,
        e_dis=e_dis,
        e_tq=0.0,
        e_pen=e_pen,
        e_qpos=0.0,
        e_force=0.0,
        scene_penetration_depth=penetration_depth,
        selected_penetration=bool(selected_penetration),
        selected_path_blocked_count=int(blocked_path_count),
        e_align=0.0,
        e_palm=0.0,
    )
    return metrics, contacts


def _score_projected_states_batch(
    scene: OptimizationScene,
    contact_indices: tuple[int, ...],
    projected_states: list[HandState],
    config: OptimizerConfig,
) -> np.ndarray:
    state_count = len(projected_states)
    if state_count == 0:
        return np.zeros(0, dtype=float)

    if (
        scene.object_geom_type is None
        or scene.object_geom_size is None
        or scene.object_geom_pos is None
        or scene.object_geom_rot is None
    ):
        scores = np.empty(state_count, dtype=float)
        for state_index, trial_state in enumerate(projected_states):
            trial_metrics, _ = calculate_energy(
                scene,
                contact_indices,
                trial_state,
                config,
                include_contacts=False,
            )
            scores[state_index] = float(trial_metrics.score)
        return scores

    contact_index_array = np.asarray(contact_indices, dtype=np.intp)
    contact_count = int(len(contact_index_array))
    contact_world_positions = np.zeros((state_count, contact_count, 3), dtype=float)

    penetration_count = int(scene.penetration_local_positions.shape[0])
    penetration_world_positions = np.zeros((state_count, penetration_count, 3), dtype=float)

    if contact_count > 0:
        contact_body_indices = scene.point_body_index_array[contact_index_array]
        contact_local_positions = scene.point_local_positions[contact_index_array]
    else:
        contact_body_indices = np.zeros(0, dtype=np.intp)
        contact_local_positions = np.zeros((0, 3), dtype=float)

    for state_index, trial_state in enumerate(projected_states):
        _apply_state(scene, trial_state)
        if contact_count > 0:
            contact_world_positions[state_index] = _body_local_points_to_world(
                scene,
                contact_body_indices,
                contact_local_positions,
            )
        if penetration_count > 0:
            penetration_world_positions[state_index] = _body_local_points_to_world(
                scene,
                scene.penetration_body_index_array,
                scene.penetration_local_positions,
            )

    scores = np.zeros(state_count, dtype=float)
    if contact_count > 0:
        contact_signed_distances = _batch_object_signed_distances(scene, contact_world_positions)
        if contact_signed_distances is None:
            return np.array(
                [
                    calculate_energy(scene, contact_indices, trial_state, config, include_contacts=False)[0].score
                    for trial_state in projected_states
                ],
                dtype=float,
            )
        scores += config.loss.distance_weight * np.sum(np.abs(contact_signed_distances), axis=1)

    if penetration_count > 0 and config.loss.penetration_weight != 0.0:
        penetration_signed_distances = _batch_object_signed_distances(scene, penetration_world_positions)
        if penetration_signed_distances is None:
            return np.array(
                [
                    calculate_energy(scene, contact_indices, trial_state, config, include_contacts=False)[0].score
                    for trial_state in projected_states
                ],
                dtype=float,
            )
        penetration_depths = np.maximum(-(penetration_signed_distances + PENETRATION_SURFACE_EPS), 0.0)
        scores += config.loss.penetration_weight * np.sum(
            scene.penetration_area_weights[None, :] * penetration_depths,
            axis=1,
        )

    _apply_state(scene, projected_states[-1])
    return scores


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
    valid_dim_indices: list[int] = []
    actual_deltas: list[float] = []
    projected_trial_states: list[HandState] = []

    for dim_index, epsilon in enumerate(epsilons):
        trial_vector = base_vector.copy()
        trial_vector[dim_index] += epsilon
        trial_state = _project_state(scene, _unpack_state(trial_vector), config)
        projected_vector = _pack_state(trial_state)
        actual_delta = projected_vector[dim_index] - base_vector[dim_index]
        if abs(actual_delta) < 1e-9:
            continue
        valid_dim_indices.append(dim_index)
        actual_deltas.append(float(actual_delta))
        projected_trial_states.append(trial_state)

    trial_scores = _score_projected_states_batch(
        scene,
        contact_indices,
        projected_trial_states,
        config,
    )
    for dim_index, actual_delta, trial_score in zip(
        valid_dim_indices,
        actual_deltas,
        trial_scores,
        strict=True,
    ):
        gradient[dim_index] = (float(trial_score) - base_score) / actual_delta

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
                "hand_penetration_sample_count": config.scene.hand_penetration_sample_count,
                "hand_point_surface_source": config.scene.hand_point_surface_source,
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
            "hand_penetration_sample_count": config.scene.hand_penetration_sample_count,
            "hand_point_surface_source": config.scene.hand_point_surface_source,
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
        point_body_index_array=np.zeros(0, dtype=np.intp),
        point_local_positions=np.zeros((0, 3), dtype=float),
        point_local_normals=np.zeros((0, 3), dtype=float),
        penetration_body_index_array=np.zeros(0, dtype=np.intp),
        penetration_local_positions=np.zeros((0, 3), dtype=float),
        penetration_area_weights=np.zeros(0, dtype=float),
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
