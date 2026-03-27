#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from view_franka_inspire import (
    PICK_CYLINDER_HALF_HEIGHT,
    PICK_CYLINDER_POS,
    PICK_CYLINDER_RADIUS,
    _load_hand_spec,
)
from view_surface_points import (
    HAND_ROLE_COLORS,
    SEGMENT_SPECS,
    compute_finger_surface_point_records,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "grasp_candidate_search.toml"

SELECTED_HAND_POINT_RADIUS = 0.0065
SELECTED_OBJECT_POINT_RADIUS = 0.0065
SELECTED_OBJECT_POINT_COLOR = np.array([0.95, 0.97, 0.99, 1.0], dtype=float)

FLOOR_NAME = "floor"
FLOOR_RGBA = np.array([0.92, 0.93, 0.95, 1.0], dtype=float)
CYLINDER_RGBA = np.array([0.91, 0.58, 0.19, 1.0], dtype=float)


@dataclass(frozen=True)
class SceneConfig:
    hand: str
    hand_points_per_joint: int
    object_body_name: str
    object_geom_names: tuple[str, ...]
    hand_penetration_sample_count: int = 0
    hand_point_surface_source: str = "collision"


@dataclass(frozen=True)
class HandPoseConfig:
    sample_count: int
    surface_offset: float
    surface_candidate_count: int
    roll_search_steps: int
    distance_std_weight: float
    distance_max_weight: float
    accessible_normal_min_z: float
    thumb_contact_weight: float


@dataclass(frozen=True)
class HandInitConfig:
    flexion_min_rad: float
    flexion_max_rad: float
    thumb_pinch_rad: float
    thumb_non_yaw_zero: bool


@dataclass(frozen=True)
class GeneConfig:
    contact_count: int
    mutation_probability: float
    force_at_least_one_mutation: bool


@dataclass(frozen=True)
class SearchConfig:
    random_seed: int
    initial_gene_count: int
    children_per_parent: int
    min_generations: int
    max_generations: int
    plateau_window: int
    plateau_tolerance: float
    max_population_size: int
    candidate_limit: int
    strict_improvement_margin: float


@dataclass(frozen=True)
class LossConfig:
    distance_weight: float
    torque_weight: float
    penetration_weight: float
    penetration_penalty_value: float
    qpos_weight: float
    qpos_edge_margin: float
    qpos_excluded_joint_names: tuple[str, ...]
    contact_alignment_weight: float = 0.0
    contact_alignment_min_dot: float = 0.0
    palm_alignment_weight: float = 0.0
    palm_alignment_min_dot: float = -1.0


@dataclass(frozen=True)
class OutputConfig:
    candidate_json: Path
    snapshot: Path | None


@dataclass(frozen=True)
class OptimizerConfig:
    scene: SceneConfig
    hand_pose: HandPoseConfig
    hand_init: HandInitConfig
    gene: GeneConfig
    search: SearchConfig
    loss: LossConfig
    output: OutputConfig


@dataclass(frozen=True)
class HandPoseSample:
    sample_index: int
    root_pos: np.ndarray
    root_quat: np.ndarray
    palm_world_pos: np.ndarray
    palm_world_normal: np.ndarray
    palm_roll_deg: float
    hand_qpos: np.ndarray
    hand_ctrl: np.ndarray
    anchor_world_pos: np.ndarray
    surface_world_pos: np.ndarray
    surface_world_normal: np.ndarray
    orientation_score: float


@dataclass(frozen=True)
class HandActuatorSpec:
    actuator_index: int
    actuator_name: str
    joint_name: str
    qpos_index: int
    ctrl_min: float
    ctrl_max: float
    finger: str
    role: str


@dataclass(frozen=True)
class SurfaceQueryResult:
    geom_name: str
    body_name: str
    body_id: int
    world_pos: np.ndarray
    world_normal: np.ndarray
    signed_distance: float


@dataclass(frozen=True)
class HandPointProbe:
    point_index: int
    finger: str
    segment: str
    role: str
    body_name: str
    local_pos: np.ndarray
    local_normal: np.ndarray
    world_pos: np.ndarray
    world_normal: np.ndarray
    object_geom_name: str
    object_body_name: str
    object_world_pos: np.ndarray
    object_world_normal: np.ndarray
    object_local_pos: np.ndarray
    distance: float
    signed_distance: float
    torque_vector: np.ndarray
    force_vector: np.ndarray
    path_blocked: bool = False


@dataclass(frozen=True)
class GeneMetrics:
    point_indices: tuple[int, ...]
    score: float
    e_dis: float
    e_tq: float
    e_pen: float
    e_qpos: float
    e_force: float
    scene_penetration_depth: float
    selected_penetration: bool
    selected_path_blocked_count: int = 0
    e_align: float = 0.0
    e_palm: float = 0.0


@dataclass(frozen=True)
class PoseSearchResult:
    pose_sample: HandPoseSample
    probes: list[HandPointProbe]
    object_center: np.ndarray
    final_population: list[tuple[int, ...]]
    metrics_cache: dict[tuple[int, ...], GeneMetrics]
    provenance: dict[tuple[int, ...], dict[str, Any]]
    generation_stats: list[dict[str, Any]]
    scene_penetration_depth: float
    e_qpos: float
    finger_qpos_penalties: dict[str, float]
    stop_reason: str


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec.copy()
    return vec / norm


def _smoothstep_unit(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _path_from_config(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def _angle_to_rad(raw: dict[str, Any], rad_key: str, deg_key: str) -> float:
    if rad_key in raw:
        return float(raw[rad_key])
    if deg_key in raw:
        return float(np.deg2rad(float(raw[deg_key])))
    raise KeyError(f"Expected '{rad_key}' or '{deg_key}' in config.")


def parse_hand_init_config(raw: dict[str, Any]) -> HandInitConfig:
    return HandInitConfig(
        flexion_min_rad=_angle_to_rad(raw, "flexion_min_rad", "flexion_min_deg"),
        flexion_max_rad=_angle_to_rad(raw, "flexion_max_rad", "flexion_max_deg"),
        thumb_pinch_rad=_angle_to_rad(raw, "thumb_pinch_rad", "thumb_pinch_deg"),
        thumb_non_yaw_zero=bool(raw.get("thumb_non_yaw_zero", False)),
    )


def parse_qpos_excluded_joint_names(raw: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(name) for name in raw.get("qpos_excluded_joint_names", ()))


def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    return _normalize(np.array([w, x, y, z], dtype=float))


def _make_orthonormal_basis_from_x(x_axis: np.ndarray, roll_rad: float) -> np.ndarray:
    x_axis = _normalize(x_axis)
    helper = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(np.dot(helper, x_axis)) > 0.95:
        helper = np.array([0.0, 1.0, 0.0], dtype=float)

    y_axis = helper - np.dot(helper, x_axis) * x_axis
    y_axis = _normalize(y_axis)
    z_axis = _normalize(np.cross(x_axis, y_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))

    cos_r = float(np.cos(roll_rad))
    sin_r = float(np.sin(roll_rad))
    rolled_y = cos_r * y_axis + sin_r * z_axis
    rolled_z = -sin_r * y_axis + cos_r * z_axis
    return np.column_stack([x_axis, rolled_y, rolled_z])


def load_optimizer_config(config_path: Path) -> OptimizerConfig:
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    scene_raw = raw["scene"]
    hand_pose_raw = raw["hand_pose"]
    hand_init_raw = raw["hand_init"]
    gene_raw = raw["gene"]
    search_raw = raw["search"]
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
    gene = GeneConfig(
        contact_count=int(gene_raw["contact_count"]),
        mutation_probability=float(gene_raw["mutation_probability"]),
        force_at_least_one_mutation=bool(gene_raw["force_at_least_one_mutation"]),
    )
    search = SearchConfig(
        random_seed=int(search_raw["random_seed"]),
        initial_gene_count=int(search_raw["initial_gene_count"]),
        children_per_parent=int(search_raw["children_per_parent"]),
        min_generations=int(search_raw["min_generations"]),
        max_generations=int(search_raw["max_generations"]),
        plateau_window=int(search_raw["plateau_window"]),
        plateau_tolerance=float(search_raw["plateau_tolerance"]),
        max_population_size=int(search_raw["max_population_size"]),
        candidate_limit=int(search_raw["candidate_limit"]),
        strict_improvement_margin=float(search_raw["strict_improvement_margin"]),
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
        gene=gene,
        search=search,
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
    if len(config.scene.object_geom_names) == 0:
        raise ValueError("scene.object_geom_names must contain at least one geom name.")
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
    if config.hand_init.flexion_min_rad < 0.0 or config.hand_init.flexion_max_rad < 0.0:
        raise ValueError("hand_init flexion bounds must be non-negative.")
    if config.hand_init.flexion_min_rad > config.hand_init.flexion_max_rad:
        raise ValueError("hand_init.flexion_min_rad must be <= hand_init.flexion_max_rad.")
    if config.gene.contact_count <= 0:
        raise ValueError("gene.contact_count must be positive.")
    if not 0.0 <= config.gene.mutation_probability <= 1.0:
        raise ValueError("gene.mutation_probability must be in [0, 1].")
    if config.search.initial_gene_count <= 0:
        raise ValueError("search.initial_gene_count must be positive.")
    if config.search.children_per_parent <= 0:
        raise ValueError("search.children_per_parent must be positive.")
    if config.search.min_generations < 0:
        raise ValueError("search.min_generations must be non-negative.")
    if config.search.max_generations <= 0:
        raise ValueError("search.max_generations must be positive.")
    if config.search.min_generations > config.search.max_generations:
        raise ValueError("search.min_generations must be <= search.max_generations.")
    if config.search.plateau_window <= 0:
        raise ValueError("search.plateau_window must be positive.")
    if config.search.plateau_tolerance < 0.0:
        raise ValueError("search.plateau_tolerance must be non-negative.")
    if config.search.max_population_size <= 0:
        raise ValueError("search.max_population_size must be positive for hand-only optimization.")
    if config.search.candidate_limit <= 0:
        raise ValueError("search.candidate_limit must be positive.")
    if config.loss.qpos_weight < 0.0:
        raise ValueError("loss.qpos_weight must be non-negative.")
    if not 0.0 < config.loss.qpos_edge_margin < 0.5:
        raise ValueError("loss.qpos_edge_margin must be in (0, 0.5).")
    if config.output.candidate_json is None:
        raise ValueError("output.candidate_json is required.")


def _box_signed_distance_and_closest(point_local: np.ndarray, half_extents: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    clamped = np.clip(point_local, -half_extents, half_extents)
    outside_delta = point_local - clamped
    outside_dist = np.linalg.norm(outside_delta)

    if outside_dist > 1e-8:
        normal = outside_delta / outside_dist
        return clamped, normal, float(outside_dist)

    interior_clearance = half_extents - np.abs(point_local)
    axis = int(np.argmin(interior_clearance))
    sign = 1.0 if point_local[axis] >= 0.0 else -1.0
    closest = point_local.copy()
    closest[axis] = sign * half_extents[axis]
    normal = np.zeros(3, dtype=float)
    normal[axis] = sign
    return closest, normal, -float(interior_clearance[axis])


def _sphere_signed_distance_and_closest(point_local: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, float]:
    distance_to_center = np.linalg.norm(point_local)
    normal = np.array([1.0, 0.0, 0.0], dtype=float) if distance_to_center < 1e-8 else point_local / distance_to_center
    closest = radius * normal
    return closest, normal, float(distance_to_center - radius)


def _capsule_signed_distance_and_closest(point_local: np.ndarray, radius: float, half_length: float) -> tuple[np.ndarray, np.ndarray, float]:
    segment_point = np.array([0.0, 0.0, np.clip(point_local[2], -half_length, half_length)], dtype=float)
    delta = point_local - segment_point
    radial_norm = np.linalg.norm(delta)
    normal = np.array([1.0, 0.0, 0.0], dtype=float) if radial_norm < 1e-8 else delta / radial_norm
    closest = segment_point + radius * normal
    return closest, normal, float(radial_norm - radius)


def _cylinder_signed_distance_and_closest(point_local: np.ndarray, radius: float, half_height: float) -> tuple[np.ndarray, np.ndarray, float]:
    radial = np.linalg.norm(point_local[:2])
    radial_dir = np.array([1.0, 0.0], dtype=float) if radial < 1e-8 else point_local[:2] / radial

    side_z = float(np.clip(point_local[2], -half_height, half_height))
    side_point = np.array([radius * radial_dir[0], radius * radial_dir[1], side_z], dtype=float)
    side_normal = np.array([radial_dir[0], radial_dir[1], 0.0], dtype=float)

    cap_xy_radius = min(radial, radius)
    cap_xy = cap_xy_radius * radial_dir
    top_point = np.array([cap_xy[0], cap_xy[1], half_height], dtype=float)
    top_normal = np.array([0.0, 0.0, 1.0], dtype=float)

    candidates = (
        (side_point, side_normal),
        (top_point, top_normal),
    )
    closest_local, normal_local = min(
        candidates,
        key=lambda item: float(np.linalg.norm(point_local - item[0])),
    )

    radial_delta = radial - radius
    top_delta = point_local[2] - half_height
    bottom_delta = -half_height - point_local[2]

    if radial_delta <= 0.0 and top_delta <= 0.0 and bottom_delta <= 0.0:
        signed_distance = -float(min(radius - radial, half_height - point_local[2], half_height + point_local[2]))
    else:
        outside = np.array([max(radial_delta, 0.0), max(top_delta, 0.0), max(bottom_delta, 0.0)], dtype=float)
        signed_distance = float(np.linalg.norm(outside))

    return closest_local, normal_local, signed_distance


def closest_surface_on_geom(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    world_point: np.ndarray,
) -> SurfaceQueryResult:
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
    body_id = int(model.geom_bodyid[geom_id])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"

    geom_center = data.geom_xpos[geom_id].copy()
    geom_rot = data.geom_xmat[geom_id].reshape(3, 3)
    point_local = geom_rot.T @ (world_point - geom_center)
    geom_type = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id].copy()

    if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        closest_local, normal_local, signed_distance = _cylinder_signed_distance_and_closest(
            point_local,
            radius=float(size[0]),
            half_height=float(size[1]),
        )
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        closest_local, normal_local, signed_distance = _box_signed_distance_and_closest(point_local, size[:3])
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        closest_local, normal_local, signed_distance = _sphere_signed_distance_and_closest(
            point_local,
            radius=float(size[0]),
        )
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        closest_local, normal_local, signed_distance = _capsule_signed_distance_and_closest(
            point_local,
            radius=float(size[0]),
            half_length=float(size[1]),
        )
    else:
        try:
            geom_type_name = mujoco.mjtGeom(geom_type).name
        except Exception:
            geom_type_name = str(geom_type)
        raise NotImplementedError(
            f"Closest-surface query is not implemented for geom type {geom_type_name} ({geom_name})."
        )

    return SurfaceQueryResult(
        geom_name=geom_name,
        body_name=body_name,
        body_id=body_id,
        world_pos=geom_center + geom_rot @ closest_local,
        world_normal=_normalize(geom_rot @ normal_local),
        signed_distance=float(signed_distance),
    )


def closest_surface_on_object(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_ids: tuple[int, ...],
    world_point: np.ndarray,
) -> SurfaceQueryResult:
    results = [closest_surface_on_geom(model, data, geom_id, world_point) for geom_id in geom_ids]
    return min(results, key=lambda result: float(np.linalg.norm(world_point - result.world_pos)))


def _add_floor(spec: mujoco.MjSpec) -> None:
    floor = spec.worldbody.add_geom()
    floor.name = FLOOR_NAME
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = np.array([0.0, 0.0, 0.05], dtype=float)
    floor.pos = np.array([0.0, 0.0, 0.0], dtype=float)
    floor.rgba = FLOOR_RGBA.copy()
    floor.friction = np.array([1.0, 0.05, 0.01], dtype=float)


def _add_fixed_cylinder(spec: mujoco.MjSpec, body_name: str, geom_name: str) -> None:
    body = spec.worldbody.add_body()
    body.name = body_name
    body.pos = PICK_CYLINDER_POS.copy()

    geom = body.add_geom()
    geom.name = geom_name
    geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    geom.size = np.array([PICK_CYLINDER_RADIUS, PICK_CYLINDER_HALF_HEIGHT, 0.0], dtype=float)
    geom.condim = 4
    geom.friction = np.array([1.1, 0.05, 0.01], dtype=float)
    geom.rgba = CYLINDER_RGBA.copy()

    anchor_site = body.add_site()
    anchor_site.name = "grasp_anchor"
    anchor_site.type = mujoco.mjtGeom.mjGEOM_SPHERE
    anchor_site.pos = np.zeros(3, dtype=float)
    anchor_site.size = np.array([0.006, 0.0, 0.0], dtype=float)
    anchor_site.rgba = np.array([1.0, 0.2, 0.2, 1.0], dtype=float)


def _initialize_static_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_qpos: np.ndarray | None = None,
    hand_ctrl: np.ndarray | None = None,
) -> None:
    qpos = model.qpos0.copy()
    if hand_qpos is not None:
        qpos[:] = np.asarray(hand_qpos, dtype=float)
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    if model.nu > 0:
        ctrl = np.zeros(model.nu, dtype=float) if hand_ctrl is None else np.asarray(hand_ctrl, dtype=float)
        data.ctrl[:] = np.clip(ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    mujoco.mj_forward(model, data)


def _reference_root_to_palm_transform(side: str) -> tuple[np.ndarray, np.ndarray]:
    hand_spec = _load_hand_spec(side)
    root_body = hand_spec.body(f"{side}_hand_base")
    root_body.pos = np.zeros(3, dtype=float)
    root_body.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    model = hand_spec.compile()
    data = mujoco.MjData(model)
    _initialize_static_state(model, data)

    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_base")
    palm_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_palm")
    root_pos = data.xpos[root_id].copy()
    root_rot = data.xmat[root_id].reshape(3, 3)
    palm_pos = data.site_xpos[palm_site_id].copy()
    palm_rot = data.site_xmat[palm_site_id].reshape(3, 3)
    return root_rot.T @ (palm_pos - root_pos), root_rot.T @ palm_rot


def describe_hand_actuators(model: mujoco.MjModel) -> list[HandActuatorSpec]:
    actuator_specs: list[HandActuatorSpec] = []
    for actuator_index in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_index) or f"actuator_{actuator_index}"
        joint_id = int(model.actuator_trnid[actuator_index, 0])
        if joint_id < 0:
            continue
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
        qpos_index = int(model.jnt_qposadr[joint_id])
        ctrl_min, ctrl_max = (float(value) for value in model.actuator_ctrlrange[actuator_index])

        name_parts = actuator_name.split("_")
        prefix_offset = 1 if len(name_parts) > 1 and name_parts[0] == "inspire" else 0
        if len(name_parts) - prefix_offset < 3:
            finger = "unknown"
            role = actuator_name
        else:
            finger = name_parts[prefix_offset + 1]
            role = "_".join(name_parts[prefix_offset + 2 : -1]) or actuator_name

        actuator_specs.append(
            HandActuatorSpec(
                actuator_index=actuator_index,
                actuator_name=actuator_name,
                joint_name=joint_name,
                qpos_index=qpos_index,
                ctrl_min=ctrl_min,
                ctrl_max=ctrl_max,
                finger=finger,
                role=role,
            )
        )
    return actuator_specs


def _target_from_radians(target_rad: float, ctrl_min: float, ctrl_max: float) -> float:
    magnitude = float(abs(target_rad))
    if target_rad < 0.0 and ctrl_min < 0.0:
        raw_target = -magnitude
    elif target_rad > 0.0 and ctrl_max > 0.0:
        raw_target = magnitude
    elif ctrl_max <= 0.0:
        raw_target = -magnitude
    elif ctrl_min >= 0.0:
        raw_target = magnitude
    else:
        raw_target = 0.0
    return float(np.clip(raw_target, ctrl_min, ctrl_max))


def sample_hand_joint_targets(
    rng: np.random.Generator,
    actuator_specs: list[HandActuatorSpec],
    config: HandInitConfig,
) -> tuple[np.ndarray, np.ndarray]:
    hand_ctrl = np.zeros(len(actuator_specs), dtype=float)

    for spec in actuator_specs:
        if spec.finger == "thumb" and "yaw" in spec.role:
            target_rad = config.thumb_pinch_rad
        elif spec.finger == "thumb" and config.thumb_non_yaw_zero:
            target_rad = 0.0
        else:
            target_rad = float(rng.uniform(config.flexion_min_rad, config.flexion_max_rad))
            if spec.ctrl_max <= 0.0:
                target_rad *= -1.0
        target = _target_from_radians(target_rad, spec.ctrl_min, spec.ctrl_max)
        hand_ctrl[spec.actuator_index] = target

    hand_qpos = hand_ctrl.copy()
    return hand_qpos, hand_ctrl


def _axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _normalize(np.asarray(axis, dtype=float))
    if np.linalg.norm(axis) < 1e-8 or abs(angle_rad) < 1e-12:
        return np.eye(3, dtype=float)
    x, y, z = axis
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def _rotation_align_vectors(source_vec: np.ndarray, target_vec: np.ndarray) -> np.ndarray:
    source = _normalize(np.asarray(source_vec, dtype=float))
    target = _normalize(np.asarray(target_vec, dtype=float))
    if np.linalg.norm(source) < 1e-8 or np.linalg.norm(target) < 1e-8:
        return np.eye(3, dtype=float)

    dot_value = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot_value > 1.0 - 1.0e-8:
        return np.eye(3, dtype=float)
    if dot_value < -1.0 + 1.0e-8:
        fallback = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(source[0]) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=float)
        axis = _normalize(np.cross(source, fallback))
        return _axis_angle_rotation(axis, np.pi)

    axis = _normalize(np.cross(source, target))
    angle = float(np.arccos(dot_value))
    return _axis_angle_rotation(axis, angle)


def _build_reference_object_scene(
    config: OptimizerConfig,
) -> tuple[mujoco.MjModel, mujoco.MjData, int, tuple[int, ...]]:
    dummy_qpos = np.zeros(12, dtype=float)
    dummy_pose = HandPoseSample(
        sample_index=-1,
        root_pos=np.array([1.2, -1.2, 0.6], dtype=float),
        root_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        palm_world_pos=np.zeros(3, dtype=float),
        palm_world_normal=np.array([1.0, 0.0, 0.0], dtype=float),
        palm_roll_deg=0.0,
        hand_qpos=dummy_qpos.copy(),
        hand_ctrl=dummy_qpos.copy(),
        anchor_world_pos=np.zeros(3, dtype=float),
        surface_world_pos=np.zeros(3, dtype=float),
        surface_world_normal=np.array([1.0, 0.0, 0.0], dtype=float),
        orientation_score=0.0,
    )
    _, model, data = build_hand_only_scene(config, dummy_pose)
    object_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.scene.object_body_name))
    if object_body_id < 0:
        raise ValueError(f"Object body '{config.scene.object_body_name}' was not found.")
    object_geom_ids = tuple(
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name))
        for geom_name in config.scene.object_geom_names
    )
    if any(geom_id < 0 for geom_id in object_geom_ids):
        raise ValueError("One or more object geom names were not found in the reference scene.")
    return model, data, object_body_id, object_geom_ids


def _object_anchor_world_pos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_body_id: int,
) -> np.ndarray:
    for site_name in ("grasp_anchor", "anchor_marker"):
        site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name))
        if site_id >= 0 and int(model.site_bodyid[site_id]) == object_body_id:
            return data.site_xpos[site_id].copy()
    return data.xipos[object_body_id].copy()


def _sample_box_surface_local(
    rng: np.random.Generator,
    half_extents: np.ndarray,
    normal_min_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    hx, hy, hz = (float(value) for value in half_extents)
    faces = [
        (np.array([1.0, 0.0, 0.0], dtype=float), 4.0 * hy * hz),
        (np.array([-1.0, 0.0, 0.0], dtype=float), 4.0 * hy * hz),
        (np.array([0.0, 1.0, 0.0], dtype=float), 4.0 * hx * hz),
        (np.array([0.0, -1.0, 0.0], dtype=float), 4.0 * hx * hz),
        (np.array([0.0, 0.0, 1.0], dtype=float), 4.0 * hx * hy),
    ]
    weights = np.array([area for _, area in faces], dtype=float)
    normal = faces[int(rng.choice(len(faces), p=weights / np.sum(weights)))][0]
    if normal[2] < normal_min_z:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)

    if abs(normal[0]) > 0.5:
        point_local = np.array(
            [normal[0] * hx, float(rng.uniform(-hy, hy)), float(rng.uniform(-hz, hz))],
            dtype=float,
        )
    elif abs(normal[1]) > 0.5:
        point_local = np.array(
            [float(rng.uniform(-hx, hx)), normal[1] * hy, float(rng.uniform(-hz, hz))],
            dtype=float,
        )
    else:
        point_local = np.array(
            [float(rng.uniform(-hx, hx)), float(rng.uniform(-hy, hy)), hz],
            dtype=float,
        )
    return point_local, normal


def _sample_cylinder_surface_local(
    rng: np.random.Generator,
    radius: float,
    half_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    side_area = 4.0 * np.pi * radius * half_height
    top_area = np.pi * radius * radius
    if float(rng.uniform(0.0, side_area + top_area)) < side_area:
        theta = float(rng.uniform(-np.pi, np.pi))
        z = float(rng.uniform(-half_height, half_height))
        point_local = np.array([radius * np.cos(theta), radius * np.sin(theta), z], dtype=float)
        normal_local = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)
    else:
        theta = float(rng.uniform(-np.pi, np.pi))
        radial = float(np.sqrt(rng.uniform(0.0, 1.0)) * radius)
        point_local = np.array([radial * np.cos(theta), radial * np.sin(theta), half_height], dtype=float)
        normal_local = np.array([0.0, 0.0, 1.0], dtype=float)
    return point_local, normal_local


def _sample_sphere_surface_local(
    rng: np.random.Generator,
    radius: float,
    normal_min_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(256):
        direction = _normalize(rng.normal(size=3))
        if direction[2] >= normal_min_z:
            return radius * direction, direction
    direction = np.array([0.0, 0.0, 1.0], dtype=float)
    return radius * direction, direction


def _sample_surface_point_on_geom(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    rng: np.random.Generator,
    normal_min_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    geom_center = data.geom_xpos[geom_id].copy()
    geom_rot = data.geom_xmat[geom_id].reshape(3, 3)
    geom_type = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id].copy()

    if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        point_local, normal_local = _sample_cylinder_surface_local(
            rng,
            radius=float(size[0]),
            half_height=float(size[1]),
        )
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        point_local, normal_local = _sample_box_surface_local(
            rng,
            half_extents=size[:3],
            normal_min_z=normal_min_z,
        )
    elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        point_local, normal_local = _sample_sphere_surface_local(
            rng,
            radius=float(size[0]),
            normal_min_z=normal_min_z,
        )
    else:
        radius = float(max(size[0], size[1], size[2], 1.0e-3))
        for _ in range(256):
            direction = _normalize(rng.normal(size=3))
            if direction[2] < normal_min_z:
                continue
            query = closest_surface_on_geom(model, data, geom_id, geom_center + 3.0 * radius * direction)
            if query.world_normal[2] >= normal_min_z:
                return query.world_pos, query.world_normal
        query = closest_surface_on_geom(model, data, geom_id, geom_center + np.array([0.0, 0.0, 3.0 * radius]))
        return query.world_pos, query.world_normal

    return geom_center + geom_rot @ point_local, _normalize(geom_rot @ normal_local)


def _sample_surface_point_on_object(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_ids: tuple[int, ...],
    rng: np.random.Generator,
    normal_min_z: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(geom_ids) == 1:
        return _sample_surface_point_on_geom(model, data, geom_ids[0], rng, normal_min_z)
    geom_index = int(rng.integers(len(geom_ids)))
    return _sample_surface_point_on_geom(model, data, geom_ids[geom_index], rng, normal_min_z)


def _reference_contact_points_in_root_frame(
    side: str,
    hand_qpos: np.ndarray,
    hand_ctrl: np.ndarray,
    point_count_per_segment: int,
    thumb_contact_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    spec = mujoco.MjSpec()
    world_site = spec.worldbody.add_site()
    world_site.name = "hand_mount_site"
    world_site.pos = np.zeros(3, dtype=float)
    world_site.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    hand_spec = _load_hand_spec(side)
    root_body = hand_spec.body(f"{side}_hand_base")
    root_body.pos = np.zeros(3, dtype=float)
    root_body.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    spec.attach(hand_spec, prefix="inspire_", site=world_site)

    model = spec.compile()
    data = mujoco.MjData(model)
    _initialize_static_state(model, data, hand_qpos=hand_qpos, hand_ctrl=hand_ctrl)
    hand_point_records, _ = compute_finger_surface_point_records(
        model,
        data,
        side=side,
        total_point_count=point_count_per_segment * len(SEGMENT_SPECS),
        point_count_per_segment=point_count_per_segment,
    )
    root_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"inspire_{side}_hand_base"))
    root_pos = data.xpos[root_id].copy()
    root_rot = data.xmat[root_id].reshape(3, 3)
    local_points = np.stack(
        [root_rot.T @ (np.asarray(record["world_pos"], dtype=float) - root_pos) for record in hand_point_records],
        axis=0,
    )
    point_weights = np.array(
        [thumb_contact_weight if str(record["finger"]) == "thumb" else 1.0 for record in hand_point_records],
        dtype=float,
    )
    return local_points, point_weights


def _roll_alignment_score(
    contact_offsets_from_palm: np.ndarray,
    point_weights: np.ndarray,
    anchor_from_palm_world: np.ndarray,
    rotation: np.ndarray,
    std_weight: float,
    max_weight: float,
) -> float:
    world_points = (rotation @ contact_offsets_from_palm.T).T
    distances = np.linalg.norm(anchor_from_palm_world[None, :] - world_points, axis=1)
    weights = np.asarray(point_weights, dtype=float)
    weights = weights / np.sum(weights)
    mean_distance = float(np.sum(weights * distances))
    variance = float(np.sum(weights * (distances - mean_distance) ** 2))
    return float(
        mean_distance
        + std_weight * np.sqrt(max(variance, 0.0))
        + max_weight * np.max(distances)
    )


def sample_hand_pose(
    sample_index: int,
    rng: np.random.Generator,
    config: HandPoseConfig,
    object_model: mujoco.MjModel,
    object_data: mujoco.MjData,
    object_body_id: int,
    object_geom_ids: tuple[int, ...],
    root_to_palm_pos: np.ndarray,
    root_to_palm_rot: np.ndarray,
    prior_palm_positions: list[np.ndarray],
    point_count_per_segment: int,
    side: str,
    hand_qpos: np.ndarray,
    hand_ctrl: np.ndarray,
) -> HandPoseSample:
    anchor_world_pos = _object_anchor_world_pos(object_model, object_data, object_body_id)
    contact_root_points, point_weights = _reference_contact_points_in_root_frame(
        side=side,
        hand_qpos=hand_qpos,
        hand_ctrl=hand_ctrl,
        point_count_per_segment=point_count_per_segment,
        thumb_contact_weight=config.thumb_contact_weight,
    )
    contact_offsets_from_palm = contact_root_points - root_to_palm_pos[None, :]
    local_grasp_axis = _normalize(np.sum(contact_offsets_from_palm * point_weights[:, None], axis=0))

    candidate_count = max(config.surface_candidate_count, 1)
    surface_candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
    for _ in range(candidate_count):
        surface_world_pos, surface_world_normal = _sample_surface_point_on_object(
            object_model,
            object_data,
            object_geom_ids,
            rng,
            normal_min_z=config.accessible_normal_min_z,
        )
        palm_world_pos = surface_world_pos + config.surface_offset * surface_world_normal
        if prior_palm_positions:
            diversity = min(float(np.linalg.norm(palm_world_pos - prior_pos)) for prior_pos in prior_palm_positions)
        else:
            diversity = 0.0
        surface_candidates.append((surface_world_pos, surface_world_normal, palm_world_pos, diversity))

    if prior_palm_positions:
        surface_world_pos, surface_world_normal, palm_world_pos, _ = max(
            surface_candidates,
            key=lambda item: item[3],
        )
    else:
        surface_world_pos, surface_world_normal, palm_world_pos, _ = surface_candidates[
            int(rng.integers(len(surface_candidates)))
        ]

    anchor_from_palm_world = anchor_world_pos - palm_world_pos
    reach_dir = _normalize(anchor_from_palm_world)
    base_rotation = _rotation_align_vectors(local_grasp_axis, reach_dir)

    best_rotation = base_rotation
    best_roll_deg = 0.0
    best_score = _roll_alignment_score(
        contact_offsets_from_palm=contact_offsets_from_palm,
        point_weights=point_weights,
        anchor_from_palm_world=anchor_from_palm_world,
        rotation=base_rotation,
        std_weight=config.distance_std_weight,
        max_weight=config.distance_max_weight,
    )
    for roll_index in range(config.roll_search_steps):
        angle_rad = (2.0 * np.pi * roll_index) / config.roll_search_steps
        rotation = _axis_angle_rotation(reach_dir, angle_rad) @ base_rotation
        score = _roll_alignment_score(
            contact_offsets_from_palm=contact_offsets_from_palm,
            point_weights=point_weights,
            anchor_from_palm_world=anchor_from_palm_world,
            rotation=rotation,
            std_weight=config.distance_std_weight,
            max_weight=config.distance_max_weight,
        )
        if score < best_score:
            best_score = score
            best_rotation = rotation
            best_roll_deg = float(np.rad2deg(angle_rad))

    root_world_rot = best_rotation
    root_world_pos = palm_world_pos - root_world_rot @ root_to_palm_pos
    root_world_quat = _rotation_matrix_to_quaternion(root_world_rot)
    palm_world_rot = root_world_rot @ root_to_palm_rot
    palm_world_normal = _normalize(palm_world_rot[:, 0])

    return HandPoseSample(
        sample_index=sample_index,
        root_pos=root_world_pos,
        root_quat=root_world_quat,
        palm_world_pos=palm_world_pos,
        palm_world_normal=palm_world_normal,
        palm_roll_deg=best_roll_deg,
        hand_qpos=hand_qpos.copy(),
        hand_ctrl=hand_ctrl.copy(),
        anchor_world_pos=anchor_world_pos.copy(),
        surface_world_pos=surface_world_pos.copy(),
        surface_world_normal=surface_world_normal.copy(),
        orientation_score=float(best_score),
    )


def sample_initial_pose_sequence(config: OptimizerConfig) -> list[HandPoseSample]:
    rng = np.random.default_rng(config.search.random_seed)
    root_to_palm_pos, root_to_palm_rot = _reference_root_to_palm_transform(config.scene.hand)
    hand_reference_model = _load_hand_spec(config.scene.hand).compile()
    actuator_specs = describe_hand_actuators(hand_reference_model)
    object_model, object_data, object_body_id, object_geom_ids = _build_reference_object_scene(config)

    prior_palm_positions: list[np.ndarray] = []
    pose_samples: list[HandPoseSample] = []
    for sample_index in range(config.hand_pose.sample_count):
        hand_qpos, hand_ctrl = sample_hand_joint_targets(
            rng=rng,
            actuator_specs=actuator_specs,
            config=config.hand_init,
        )
        pose_sample = sample_hand_pose(
            sample_index=sample_index,
            rng=rng,
            config=config.hand_pose,
            object_model=object_model,
            object_data=object_data,
            object_body_id=object_body_id,
            object_geom_ids=object_geom_ids,
            root_to_palm_pos=root_to_palm_pos,
            root_to_palm_rot=root_to_palm_rot,
            prior_palm_positions=prior_palm_positions,
            point_count_per_segment=config.scene.hand_points_per_joint,
            side=config.scene.hand,
            hand_qpos=hand_qpos,
            hand_ctrl=hand_ctrl,
        )
        pose_samples.append(pose_sample)
        prior_palm_positions.append(pose_sample.palm_world_pos.copy())
    return pose_samples


def build_hand_only_scene(
    config: OptimizerConfig,
    pose_sample: HandPoseSample,
) -> tuple[mujoco.MjSpec, mujoco.MjModel, mujoco.MjData]:
    spec = mujoco.MjSpec()
    spec.modelname = f"grasp_search_{config.scene.hand}_pose_{pose_sample.sample_index:03d}"
    spec.stat.center = np.array([0.58, 0.02, 0.30], dtype=float)
    spec.stat.extent = 0.95
    spec.visual.global_.offwidth = 1600
    spec.visual.global_.offheight = 1200

    light = spec.worldbody.add_light()
    light.name = "key_light"
    light.pos = np.array([0.3, -0.6, 1.3], dtype=float)
    light.dir = np.array([0.0, 0.2, -1.0], dtype=float)
    light.diffuse = np.array([0.95, 0.95, 0.95], dtype=float)
    light.specular = np.array([0.25, 0.25, 0.25], dtype=float)
    light.castshadow = True

    _add_floor(spec)
    _add_fixed_cylinder(
        spec,
        body_name=config.scene.object_body_name,
        geom_name=config.scene.object_geom_names[0],
    )

    hand_spec = _load_hand_spec(config.scene.hand)
    root_body = hand_spec.body(f"{config.scene.hand}_hand_base")
    root_body.pos = pose_sample.root_pos.copy()
    root_body.quat = pose_sample.root_quat.copy()

    world_site = spec.worldbody.add_site()
    world_site.name = "hand_mount_site"
    world_site.pos = np.zeros(3, dtype=float)
    world_site.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    spec.attach(hand_spec, prefix="inspire_", site=world_site)

    model = spec.compile()
    data = mujoco.MjData(model)
    _initialize_static_state(model, data, hand_qpos=pose_sample.hand_qpos, hand_ctrl=pose_sample.hand_ctrl)
    return spec, model, data


def _is_hand_geom_name(geom_name: str) -> bool:
    return geom_name.startswith("inspire_collision_hand_")


def scene_penetration_depth(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_geom_names: tuple[str, ...],
) -> float:
    object_geom_set = set(object_geom_names)
    max_depth = 0.0
    for contact in data.contact[: data.ncon]:
        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or ""
        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ""
        pair = {geom1_name, geom2_name}
        involves_hand = _is_hand_geom_name(geom1_name) or _is_hand_geom_name(geom2_name)
        if not involves_hand:
            continue
        if FLOOR_NAME not in pair and pair.isdisjoint(object_geom_set):
            continue
        if contact.dist < 0.0:
            max_depth = max(max_depth, -float(contact.dist))
    return max_depth


def build_hand_point_probes(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    hand_point_records: list[dict[str, Any]],
    config: OptimizerConfig,
) -> tuple[list[HandPointProbe], np.ndarray, float]:
    geom_ids = tuple(
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name))
        for geom_name in config.scene.object_geom_names
    )
    if any(geom_id < 0 for geom_id in geom_ids):
        missing = [
            geom_name
            for geom_name, geom_id in zip(config.scene.object_geom_names, geom_ids, strict=True)
            if geom_id < 0
        ]
        raise ValueError(f"Object geom(s) not found: {', '.join(missing)}")

    object_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.scene.object_body_name))
    if object_body_id < 0:
        raise ValueError(f"Object body '{config.scene.object_body_name}' was not found.")

    object_center = data.xipos[object_body_id].copy()
    object_body_pos = data.xpos[object_body_id].copy()
    object_body_rot = data.xmat[object_body_id].reshape(3, 3)
    pose_penetration_depth = scene_penetration_depth(model, data, config.scene.object_geom_names)

    probes = []
    for point_index, point_record in enumerate(hand_point_records):
        world_point = np.asarray(point_record["world_pos"], dtype=float)
        query = closest_surface_on_object(model, data, geom_ids, world_point)
        force_vector = -query.world_normal
        torque_vector = np.cross(query.world_pos - object_center, force_vector)
        object_local_pos = object_body_rot.T @ (query.world_pos - object_body_pos)

        probes.append(
            HandPointProbe(
                point_index=point_index,
                finger=str(point_record["finger"]),
                segment=str(point_record["segment"]),
                role=str(point_record["role"]),
                body_name=str(point_record["body_name"]),
                local_pos=np.asarray(point_record["local_pos"], dtype=float),
                local_normal=np.asarray(point_record["local_normal"], dtype=float),
                world_pos=world_point,
                world_normal=np.asarray(point_record["world_normal"], dtype=float),
                object_geom_name=query.geom_name,
                object_body_name=query.body_name,
                object_world_pos=query.world_pos,
                object_world_normal=query.world_normal,
                object_local_pos=object_local_pos,
                distance=float(np.linalg.norm(world_point - query.world_pos)),
                signed_distance=query.signed_distance,
                torque_vector=torque_vector,
                force_vector=force_vector,
            )
        )

    return probes, object_center, pose_penetration_depth


def _joint_edge_penalty(qpos_value: float, ctrl_min: float, ctrl_max: float, edge_margin: float) -> float:
    ctrl_span = float(ctrl_max - ctrl_min)
    if ctrl_span <= 1e-8:
        return 0.0

    normalized = float(np.clip((qpos_value - ctrl_min) / ctrl_span, 0.0, 1.0))
    edge_distance = min(normalized, 1.0 - normalized)
    if edge_distance >= edge_margin:
        return 0.0

    alpha = 1.0 - edge_distance / edge_margin
    return _smoothstep_unit(alpha)


def compute_finger_qpos_penalties(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_specs: list[HandActuatorSpec],
    loss: LossConfig,
) -> tuple[float, dict[str, float]]:
    per_finger_penalties: dict[str, list[float]] = defaultdict(list)
    ordered_fingers = ("thumb", "index", "middle", "ring", "pinky")
    excluded_joint_names = tuple(loss.qpos_excluded_joint_names)

    for spec in actuator_specs:
        if any(spec.joint_name == name or spec.joint_name.endswith(name) for name in excluded_joint_names):
            continue
        qpos_value = float(data.qpos[spec.qpos_index])
        per_joint_penalty = _joint_edge_penalty(
            qpos_value=qpos_value,
            ctrl_min=spec.ctrl_min,
            ctrl_max=spec.ctrl_max,
            edge_margin=loss.qpos_edge_margin,
        )
        per_finger_penalties[spec.finger].append(per_joint_penalty)

    finger_penalties = {
        finger: float(max(per_finger_penalties.get(finger, [0.0])))
        for finger in ordered_fingers
    }
    e_qpos = float(sum(finger_penalties.values()))
    return e_qpos, finger_penalties


def evaluate_gene(
    point_indices: tuple[int, ...],
    probes: list[HandPointProbe],
    loss: LossConfig,
    pose_penetration_depth: float,
    e_qpos: float,
) -> GeneMetrics:
    selected = [probes[index] for index in point_indices]
    distances = np.array([probe.distance for probe in selected], dtype=float)
    e_dis = float(np.linalg.norm(distances))
    e_tq = float(np.linalg.norm(np.sum([probe.torque_vector for probe in selected], axis=0)))
    e_force = float(np.linalg.norm(np.sum([probe.force_vector for probe in selected], axis=0)))
    selected_penetration = any(probe.signed_distance < 0.0 for probe in selected)
    pose_penetrating = pose_penetration_depth > 0.0
    e_pen = loss.penetration_penalty_value if (selected_penetration or pose_penetrating) else 0.0
    score = (
        loss.distance_weight * e_dis
        + loss.torque_weight * e_tq
        + loss.penetration_weight * e_pen
        + loss.qpos_weight * e_qpos
    )
    return GeneMetrics(
        point_indices=point_indices,
        score=float(score),
        e_dis=e_dis,
        e_tq=e_tq,
        e_pen=float(e_pen),
        e_qpos=float(e_qpos),
        e_force=e_force,
        scene_penetration_depth=float(pose_penetration_depth),
        selected_penetration=selected_penetration,
    )


def make_random_gene(
    rng: np.random.Generator,
    point_count: int,
    contact_count: int,
) -> tuple[int, ...]:
    return tuple(sorted(int(index) for index in rng.choice(point_count, size=contact_count, replace=False)))


def mutate_gene(
    parent_indices: tuple[int, ...],
    point_count: int,
    mutation_probability: float,
    force_at_least_one_mutation: bool,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    child = list(parent_indices)
    mutated = False

    for slot in range(len(child)):
        if rng.random() >= mutation_probability:
            continue

        current_value = child[slot]
        forbidden = {value for idx, value in enumerate(child) if idx != slot}
        candidates = [index for index in range(point_count) if index not in forbidden and index != current_value]
        if not candidates:
            continue
        child[slot] = int(rng.choice(candidates))
        mutated = True

    if not mutated and force_at_least_one_mutation:
        slot = int(rng.integers(len(child)))
        current_value = child[slot]
        forbidden = {value for idx, value in enumerate(child) if idx != slot}
        candidates = [index for index in range(point_count) if index not in forbidden and index != current_value]
        if candidates:
            child[slot] = int(rng.choice(candidates))

    return tuple(sorted(child))


def _dedupe_population_by_score(population: list[tuple[int, ...]], metrics_cache: dict[tuple[int, ...], GeneMetrics]) -> list[tuple[int, ...]]:
    return sorted(set(population), key=lambda gene: metrics_cache[gene].score)


def run_genetic_search(
    probes: list[HandPointProbe],
    config: OptimizerConfig,
    pose_penetration_depth: float,
    pose_sample_index: int,
    e_qpos: float,
) -> tuple[
    list[tuple[int, ...]],
    dict[tuple[int, ...], GeneMetrics],
    dict[tuple[int, ...], dict[str, Any]],
    list[dict[str, Any]],
    str,
]:
    point_count = len(probes)
    if point_count < config.gene.contact_count:
        raise ValueError(
            f"Requested {config.gene.contact_count} gene contacts, but only {point_count} hand points are available."
        )

    rng_seed = (
        config.search.random_seed
        + 1009 * pose_sample_index
        + int(round(pose_penetration_depth * 1e6))
        + point_count
    )
    rng = np.random.default_rng(rng_seed)
    metrics_cache: dict[tuple[int, ...], GeneMetrics] = {}
    provenance: dict[tuple[int, ...], dict[str, Any]] = {}

    def cached_metrics(gene: tuple[int, ...]) -> GeneMetrics:
        if gene not in metrics_cache:
            metrics_cache[gene] = evaluate_gene(
                gene,
                probes,
                config.loss,
                pose_penetration_depth,
                e_qpos=e_qpos,
            )
        return metrics_cache[gene]

    initial_population_set: set[tuple[int, ...]] = set()
    max_attempts = max(10 * config.search.initial_gene_count, 200)
    for _ in range(max_attempts):
        if len(initial_population_set) >= config.search.initial_gene_count:
            break
        initial_population_set.add(
            make_random_gene(rng, point_count=point_count, contact_count=config.gene.contact_count)
        )
    if len(initial_population_set) == 0:
        raise RuntimeError("Failed to generate any initial genes.")

    population = sorted(initial_population_set, key=lambda gene: cached_metrics(gene).score)
    population = population[: config.search.max_population_size]
    for gene in population:
        provenance[gene] = {"generation": 0, "parent": None}

    best_score = cached_metrics(population[0]).score
    generation_stats = [
        {
            "generation": 0,
            "population_size": len(population),
            "accepted_children": 0,
            "survivor_size": len(population),
            "best_score": best_score,
            "best_delta": 0.0,
        }
    ]
    print(
        f"[pose {pose_sample_index:02d}] gen=000 best={best_score:.6f} "
        f"accepted=0 survivors={len(population)}"
    )

    final_population = population
    best_history = [best_score]
    stop_reason = "max_generations"
    for generation in range(1, config.search.max_generations + 1):
        accepted_children: list[tuple[int, ...]] = []
        for parent_gene in population:
            parent_metrics = cached_metrics(parent_gene)
            for _ in range(config.search.children_per_parent):
                child_gene = mutate_gene(
                    parent_gene,
                    point_count=point_count,
                    mutation_probability=config.gene.mutation_probability,
                    force_at_least_one_mutation=config.gene.force_at_least_one_mutation,
                    rng=rng,
                )
                if child_gene == parent_gene:
                    continue
                child_metrics = cached_metrics(child_gene)
                if child_metrics.score + config.search.strict_improvement_margin < parent_metrics.score:
                    accepted_children.append(child_gene)
                    provenance.setdefault(
                        child_gene,
                        {"generation": generation, "parent": list(parent_gene)},
                    )

        accepted_children = _dedupe_population_by_score(accepted_children, metrics_cache)
        survivors = _dedupe_population_by_score(population + accepted_children, metrics_cache)
        survivors = survivors[: config.search.max_population_size]
        previous_best = cached_metrics(population[0]).score
        current_best = cached_metrics(survivors[0]).score
        best_delta = previous_best - current_best
        generation_stats.append(
            {
                "generation": generation,
                "population_size": len(population),
                "accepted_children": len(accepted_children),
                "survivor_size": len(survivors),
                "best_score": current_best,
                "best_delta": best_delta,
            }
        )
        print(
            f"[pose {pose_sample_index:02d}] gen={generation:03d} best={current_best:.6f} "
            f"delta={best_delta:.6e} accepted={len(accepted_children)} survivors={len(survivors)}"
        )

        if not accepted_children:
            final_population = survivors
            stop_reason = "no_accepted_children"
            break

        population = survivors
        final_population = population
        best_history.append(current_best)
        if (
            generation >= config.search.min_generations
            and len(best_history) > config.search.plateau_window
        ):
            window_delta = best_history[-config.search.plateau_window - 1] - best_history[-1]
            if window_delta < config.search.plateau_tolerance:
                stop_reason = (
                    f"plateau(window={config.search.plateau_window}, "
                    f"tol={config.search.plateau_tolerance:.2e})"
                )
                break

    generation_stats[-1]["stop_reason"] = stop_reason
    print(
        f"[pose {pose_sample_index:02d}] stop={stop_reason} "
        f"generations={generation_stats[-1]['generation']} "
        f"best={cached_metrics(final_population[0]).score:.6f}"
    )
    return final_population, metrics_cache, provenance, generation_stats, stop_reason


def _json_array(value: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(value, dtype=float).tolist()]


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
    pose_sample: HandPoseSample,
    candidate: dict[str, Any],
    output_path: Path,
) -> None:
    spec, model, data = build_hand_only_scene(config, pose_sample)
    object_body = spec.body(config.scene.object_body_name)
    for contact_index, contact in enumerate(candidate["contacts"]):
        hand_body = spec.body(contact["body_name"])
        role = str(contact["role"])
        hand_local_pos = np.asarray(contact["hand_local_pos"], dtype=float)
        object_local_pos = np.asarray(contact["object_local_pos"], dtype=float)

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
    _initialize_static_state(model, data, hand_qpos=pose_sample.hand_qpos, hand_ctrl=pose_sample.hand_ctrl)
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.scene.object_body_name)
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


def format_contact_label(probe: HandPointProbe) -> str:
    return f"{probe.point_index}:{probe.finger}_{probe.role}"


def print_candidate_summary(candidates: list[dict[str, Any]], top_n: int) -> None:
    for candidate in candidates[:top_n]:
        labels = ", ".join(
            f"{contact['point_index']}:{contact['finger']}_{contact['role']}"
            for contact in candidate["contacts"]
        )
        print(
            f"{candidate['rank']:02d}. pose={candidate['pose_index']:02d} "
            f"score={candidate['score']:.6f} "
            f"e_dis={candidate['e_dis']:.6f} e_tq={candidate['e_tq']:.6f} "
            f"e_qpos={candidate['e_qpos']:.6f} e_pen={candidate['e_pen']:.1f} "
            f"contacts=[{labels}]"
        )


def search_pose_candidates(
    config: OptimizerConfig,
) -> tuple[list[PoseSearchResult], list[dict[str, Any]]]:
    pose_results: list[PoseSearchResult] = []
    candidate_records: list[dict[str, Any]] = []

    point_count_total = config.scene.hand_points_per_joint * len(SEGMENT_SPECS)
    for pose_sample in sample_initial_pose_sequence(config):
        sample_index = pose_sample.sample_index
        print(
            f"[pose {sample_index:02d}] surface_offset={np.linalg.norm(pose_sample.palm_world_pos - pose_sample.surface_world_pos):.4f} "
            f"roll={pose_sample.palm_roll_deg:.1f}deg "
            f"thumb_pinch={np.rad2deg(pose_sample.hand_ctrl[0]):.1f}deg"
        )
        _, model, data = build_hand_only_scene(config, pose_sample)
        scene_actuator_specs = describe_hand_actuators(model)
        e_qpos, finger_qpos_penalties = compute_finger_qpos_penalties(
            model,
            data,
            scene_actuator_specs,
            config.loss,
        )
        hand_point_records, _ = compute_finger_surface_point_records(
            model,
            data,
            side=config.scene.hand,
            total_point_count=point_count_total,
            point_count_per_segment=config.scene.hand_points_per_joint,
        )
        probes, object_center_world, pose_penetration_depth = build_hand_point_probes(
            model,
            data,
            hand_point_records,
            config,
        )
        final_population, metrics_cache, provenance, generation_stats, stop_reason = run_genetic_search(
            probes,
            config,
            pose_penetration_depth=pose_penetration_depth,
            pose_sample_index=sample_index,
            e_qpos=e_qpos,
        )
        pose_results.append(
            PoseSearchResult(
                pose_sample=pose_sample,
                probes=probes,
                object_center=object_center_world,
                final_population=final_population,
                metrics_cache=metrics_cache,
                provenance=provenance,
                generation_stats=generation_stats,
                scene_penetration_depth=pose_penetration_depth,
                e_qpos=e_qpos,
                finger_qpos_penalties=finger_qpos_penalties,
                stop_reason=stop_reason,
            )
        )

        sorted_population = sorted(final_population, key=lambda gene: metrics_cache[gene].score)
        for gene in sorted_population:
            metrics = metrics_cache[gene]
            candidate_records.append(
                {
                    "pose_index": pose_sample.sample_index,
                    "hand_root_pos": _json_array(pose_sample.root_pos),
                    "hand_root_quat": _json_array(pose_sample.root_quat),
                    "palm_world_pos": _json_array(pose_sample.palm_world_pos),
                    "palm_world_normal": _json_array(pose_sample.palm_world_normal),
                    "palm_roll_deg": float(pose_sample.palm_roll_deg),
                    "anchor_world_pos": _json_array(pose_sample.anchor_world_pos),
                    "surface_world_pos": _json_array(pose_sample.surface_world_pos),
                    "surface_world_normal": _json_array(pose_sample.surface_world_normal),
                    "orientation_score": float(pose_sample.orientation_score),
                    "hand_qpos": _json_array(pose_sample.hand_qpos),
                    "hand_ctrl": _json_array(pose_sample.hand_ctrl),
                    "scene_penetration_depth": float(pose_penetration_depth),
                    "point_indices": list(gene),
                    "score": float(metrics.score),
                    "e_dis": float(metrics.e_dis),
                    "e_tq": float(metrics.e_tq),
                    "e_pen": float(metrics.e_pen),
                    "e_qpos": float(metrics.e_qpos),
                    "e_force": float(metrics.e_force),
                    "finger_qpos_penalties": {
                        finger: float(value) for finger, value in finger_qpos_penalties.items()
                    },
                    "generation": int(provenance[gene]["generation"]),
                    "parent": provenance[gene]["parent"],
                    "stop_reason": stop_reason,
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
                        }
                        for probe in (probes[index] for index in gene)
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
    limited_candidates = sorted_candidates[: config.search.candidate_limit]
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
            "gene": {
                "contact_count": config.gene.contact_count,
                "mutation_probability": config.gene.mutation_probability,
                "force_at_least_one_mutation": config.gene.force_at_least_one_mutation,
            },
            "search": {
                "random_seed": config.search.random_seed,
                "initial_gene_count": config.search.initial_gene_count,
                "children_per_parent": config.search.children_per_parent,
                "min_generations": config.search.min_generations,
                "max_generations": config.search.max_generations,
                "plateau_window": config.search.plateau_window,
                "plateau_tolerance": config.search.plateau_tolerance,
                "max_population_size": config.search.max_population_size,
                "candidate_limit": config.search.candidate_limit,
                "strict_improvement_margin": config.search.strict_improvement_margin,
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
            "joint_count": len(SEGMENT_SPECS),
            "hand_points_per_joint": config.scene.hand_points_per_joint,
            "hand_point_library_size_per_pose": config.scene.hand_points_per_joint * len(SEGMENT_SPECS),
            "saved_candidate_count": len(limited_candidates),
        },
        "pose_results": [
            {
                "pose_index": result.pose_sample.sample_index,
                "hand_root_pos": _json_array(result.pose_sample.root_pos),
                "hand_root_quat": _json_array(result.pose_sample.root_quat),
                "palm_world_pos": _json_array(result.pose_sample.palm_world_pos),
                "palm_world_normal": _json_array(result.pose_sample.palm_world_normal),
                "palm_roll_deg": float(result.pose_sample.palm_roll_deg),
                "hand_qpos": _json_array(result.pose_sample.hand_qpos),
                "hand_ctrl": _json_array(result.pose_sample.hand_ctrl),
                "scene_penetration_depth": float(result.scene_penetration_depth),
                "e_qpos": float(result.e_qpos),
                "finger_qpos_penalties": {
                    finger: float(value) for finger, value in result.finger_qpos_penalties.items()
                },
                "final_population_size": len(result.final_population),
                "best_score": float(
                    min((result.metrics_cache[gene].score for gene in result.final_population), default=np.inf)
                ),
                "stop_reason": result.stop_reason,
                "generation_stats": result.generation_stats,
            }
            for result in pose_results
        ],
        "candidates": limited_candidates,
    }

    output_path = config.output.candidate_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path, limited_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate grasp contact candidates in a hand-only environment."
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
            gene=config.gene,
            search=config.search,
            loss=config.loss,
            output=OutputConfig(candidate_json=output_json, snapshot=config.output.snapshot),
        )
    if args.snapshot is not None:
        snapshot_path = args.snapshot if args.snapshot.is_absolute() else PROJECT_ROOT / args.snapshot
        config = OptimizerConfig(
            scene=config.scene,
            hand_pose=config.hand_pose,
            hand_init=config.hand_init,
            gene=config.gene,
            search=config.search,
            loss=config.loss,
            output=OutputConfig(candidate_json=config.output.candidate_json, snapshot=snapshot_path),
        )

    pose_results, candidate_records = search_pose_candidates(config)
    output_json, saved_candidates = save_candidate_json(config, pose_results, candidate_records)
    total_generations = sum(result.generation_stats[-1]["generation"] for result in pose_results)

    print(f"Pose samples       : {len(pose_results)}")
    print(f"Hand points/joint  : {config.scene.hand_points_per_joint}")
    print(f"Total generations  : {total_generations}")
    print(f"Candidates saved   : {len(saved_candidates)}")
    print(f"Saved candidates   : {output_json}")
    print_candidate_summary(saved_candidates, top_n=max(args.print_top, 0))

    if config.output.snapshot is not None and len(saved_candidates) > 0:
        best_candidate = saved_candidates[0]
        best_pose_index = int(best_candidate["pose_index"])
        pose_result = next(result for result in pose_results if result.pose_sample.sample_index == best_pose_index)
        build_best_candidate_snapshot(
            config=config,
            pose_sample=pose_result.pose_sample,
            candidate=best_candidate,
            output_path=config.output.snapshot,
        )
        print(f"Saved snapshot     : {config.output.snapshot}")


if __name__ == "__main__":
    main()
