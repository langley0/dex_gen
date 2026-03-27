#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

from view_franka_inspire import _configure_free_camera, build_franka_inspire_spec, initialize_state


HAND_SURFACE_POINT_COUNT = 100
HAND_POINT_RADIUS = 0.0035
POINT_NORMAL_OFFSET = 0.002
FLEXION_SURFACE_HALF_ANGLE = np.deg2rad(55.0)
PALM_VIEW_NORMAL_OFFSET = 0.22
PALM_VIEW_FINGER_OFFSET = 0.10
JOINT_AXIS_CLEARANCE = 0.005

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SEGMENT_SPECS = (
    ("thumb", "1"),
    ("thumb", "0"),
    ("index", "1"),
    ("index", "0"),
    ("middle", "1"),
    ("middle", "0"),
    ("ring", "1"),
    ("ring", "0"),
    ("pinky", "1"),
    ("pinky", "0"),
)

HAND_ROLE_COLORS = {
    "proximal": np.array([0.98, 0.54, 0.16, 1.0], dtype=float),
    "intermediate": np.array([0.24, 0.82, 0.39, 1.0], dtype=float),
    "distal": np.array([0.22, 0.66, 0.95, 1.0], dtype=float),
}


def _segment_role(finger: str, segment: str) -> str:
    if segment == "1":
        return "proximal"
    return "distal" if finger == "thumb" else "intermediate"


def _segment_label(finger: str, segment: str) -> str:
    return f"{finger}_{_segment_role(finger, segment)}"


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec.copy()
    return vec / norm


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    matrix = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=float))
    return matrix.reshape(3, 3)


def _capsule_strip_area(radius: float, half_length: float, half_angle: float) -> float:
    return radius * (2.0 * half_angle) * (2.0 * half_length)


def _merge_axis_intervals(
    intervals: list[tuple[float, float]],
    lower_bound: float,
    upper_bound: float,
) -> list[tuple[float, float]]:
    clipped = []
    for start, end in intervals:
        lo = max(lower_bound, start)
        hi = min(upper_bound, end)
        if hi > lo:
            clipped.append((lo, hi))

    if not clipped:
        return []

    clipped.sort()
    merged = [clipped[0]]
    for start, end in clipped[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _axis_position_allowed(axis_pos: float, excluded_intervals: list[tuple[float, float]]) -> bool:
    return all(not (start <= axis_pos <= end) for start, end in excluded_intervals)


def _valid_axis_positions(
    half_length: float,
    n_axis: int,
    excluded_intervals: list[tuple[float, float]],
) -> np.ndarray:
    full_length = 2.0 * half_length
    full_axis_positions = -half_length + (np.arange(n_axis) + 0.5) * full_length / n_axis
    return np.array(
        [axis_pos for axis_pos in full_axis_positions if _axis_position_allowed(axis_pos, excluded_intervals)],
        dtype=float,
    )


def _segment_target_counts(
    model: mujoco.MjModel,
    side: str,
    total_point_count: int,
    point_count_per_segment: int | None = None,
) -> dict[tuple[str, str], int]:
    if point_count_per_segment is not None:
        if point_count_per_segment <= 0:
            raise ValueError("point_count_per_segment must be positive.")
        return {
            segment_spec: int(point_count_per_segment)
            for segment_spec in SEGMENT_SPECS
        }

    areas = []
    for finger, segment in SEGMENT_SPECS:
        geom_name = f"inspire_collision_hand_{side}_{finger}_{segment}"
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        radius = model.geom_size[geom_id, 0]
        half_length = model.geom_size[geom_id, 1]
        areas.append(_capsule_strip_area(radius, half_length, FLEXION_SURFACE_HALF_ANGLE))

    minimum_per_segment = 4
    remaining = max(0, total_point_count - minimum_per_segment * len(SEGMENT_SPECS))
    areas = np.array(areas, dtype=float)
    weights = areas / np.sum(areas)
    raw = remaining * weights
    counts = np.full(len(SEGMENT_SPECS), minimum_per_segment, dtype=int) + np.floor(raw).astype(int)
    remainder = remaining - int(np.sum(np.floor(raw)))
    if remainder > 0:
        order = np.argsort(-(raw - np.floor(raw)))
        counts[order[:remainder]] += 1

    return {
        segment_spec: int(count)
        for segment_spec, count in zip(SEGMENT_SPECS, counts, strict=True)
    }


def _build_capsule_strip_candidate_world_points(
    center: np.ndarray,
    axis: np.ndarray,
    palm_side: np.ndarray,
    tangent: np.ndarray,
    radius: float,
    axis_positions: np.ndarray,
    n_angle: int,
) -> np.ndarray:
    if len(axis_positions) == 0 or n_angle <= 0:
        return np.empty((0, 3), dtype=float)

    angle_step = (2.0 * FLEXION_SURFACE_HALF_ANGLE) / n_angle
    base_angle_indices = np.arange(n_angle, dtype=float)
    points = []

    for axis_index, axis_pos in enumerate(axis_positions):
        shifted_indices = (base_angle_indices + 0.5 + 0.5 * (axis_index % 2)) % n_angle
        angle_positions = -FLEXION_SURFACE_HALF_ANGLE + shifted_indices * angle_step
        normals = (
            np.cos(angle_positions)[:, None] * palm_side[None, :]
            + np.sin(angle_positions)[:, None] * tangent[None, :]
        )
        points.append(center[None, :] + axis_pos * axis[None, :] + radius * normals)

    return np.concatenate(points, axis=0)


def _farthest_point_sample(points: np.ndarray, target_count: int) -> np.ndarray:
    if target_count <= 0 or len(points) == 0:
        return np.empty((0, 3), dtype=float)
    if len(points) <= target_count:
        return points

    selected_indices = np.empty(target_count, dtype=int)
    centroid = np.mean(points, axis=0)
    start_index = int(np.argmin(np.sum((points - centroid) ** 2, axis=1)))
    selected_indices[0] = start_index

    min_distance_sq = np.sum((points - points[start_index]) ** 2, axis=1)
    min_distance_sq[start_index] = -np.inf

    for selected_count in range(1, target_count):
        next_index = int(np.argmax(min_distance_sq))
        selected_indices[selected_count] = next_index
        distance_sq = np.sum((points - points[next_index]) ** 2, axis=1)
        min_distance_sq = np.minimum(min_distance_sq, distance_sq)
        min_distance_sq[selected_indices[: selected_count + 1]] = -np.inf

    return points[selected_indices]


def _farthest_point_sample_indices(points: np.ndarray, target_count: int) -> np.ndarray:
    if target_count <= 0 or len(points) == 0:
        return np.zeros(0, dtype=int)
    if len(points) <= target_count:
        return np.arange(len(points), dtype=int)

    selected_indices = np.empty(target_count, dtype=int)
    centroid = np.mean(points, axis=0)
    start_index = int(np.argmin(np.sum((points - centroid) ** 2, axis=1)))
    selected_indices[0] = start_index

    min_distance_sq = np.sum((points - points[start_index]) ** 2, axis=1)
    min_distance_sq[start_index] = -np.inf

    for selected_count in range(1, target_count):
        next_index = int(np.argmax(min_distance_sq))
        selected_indices[selected_count] = next_index
        distance_sq = np.sum((points - points[next_index]) ** 2, axis=1)
        min_distance_sq = np.minimum(min_distance_sq, distance_sq)
        min_distance_sq[selected_indices[: selected_count + 1]] = -np.inf

    return selected_indices


def _unique_mesh_geom_ids_for_body(model: mujoco.MjModel, body_id: int) -> list[int]:
    geom_ids: list[int] = []
    seen_mesh_ids: set[int] = set()
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) != int(body_id):
            continue
        if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id < 0 or mesh_id in seen_mesh_ids:
            continue
        seen_mesh_ids.add(mesh_id)
        geom_ids.append(geom_id)
    return geom_ids


def _mesh_triangle_data_body_local(
    model: mujoco.MjModel,
    geom_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mesh_id = int(model.geom_dataid[geom_id])
    vertadr = int(model.mesh_vertadr[mesh_id])
    vertnum = int(model.mesh_vertnum[mesh_id])
    faceadr = int(model.mesh_faceadr[mesh_id])
    facenum = int(model.mesh_facenum[mesh_id])

    vertices = model.mesh_vert[vertadr : vertadr + vertnum].copy()
    vertices *= model.geom_size[geom_id][None, :]
    geom_rot = _quat_to_matrix(model.geom_quat[geom_id])
    geom_pos = model.geom_pos[geom_id].copy()
    vertices_body_local = geom_pos[None, :] + vertices @ geom_rot.T

    faces = model.mesh_face[faceadr : faceadr + facenum]
    triangles = vertices_body_local[faces]
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    triangle_normals = np.cross(edge_1, edge_2)
    triangle_norms = np.linalg.norm(triangle_normals, axis=1)
    valid = triangle_norms > 1.0e-12
    if not np.any(valid):
        return (
            np.zeros((0, 3, 3), dtype=float),
            np.zeros((0, 3), dtype=float),
            np.zeros(0, dtype=float),
        )

    triangles = triangles[valid]
    triangle_normals = triangle_normals[valid] / triangle_norms[valid, None]
    triangle_areas = 0.5 * triangle_norms[valid]
    return triangles, triangle_normals, triangle_areas


def _sample_triangle_points_world(
    rng: np.random.Generator,
    triangles_world: np.ndarray,
    triangle_normals_world: np.ndarray,
    triangle_areas: np.ndarray,
    point_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if point_count <= 0 or len(triangles_world) == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)

    probabilities = triangle_areas / float(np.sum(triangle_areas))
    triangle_indices = rng.choice(len(triangles_world), size=point_count, p=probabilities)
    selected_triangles = triangles_world[triangle_indices]
    selected_normals = triangle_normals_world[triangle_indices]

    uv = rng.uniform(size=(point_count, 2))
    sqrt_u = np.sqrt(uv[:, 0])
    barycentric = np.stack(
        [
            1.0 - sqrt_u,
            sqrt_u * (1.0 - uv[:, 1]),
            sqrt_u * uv[:, 1],
        ],
        axis=1,
    )
    world_points = np.einsum("ni,nij->nj", barycentric, selected_triangles)
    return world_points, selected_normals


def _sample_visual_segment_surface_world_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    center: np.ndarray,
    axis: np.ndarray,
    palm_side: np.ndarray,
    point_count: int,
    excluded_axis_intervals: list[tuple[float, float]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    mesh_geom_ids = _unique_mesh_geom_ids_for_body(model, body_id)
    if not mesh_geom_ids:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)

    body_pos = data.xpos[body_id]
    body_rot = data.xmat[body_id].reshape(3, 3)
    min_normal_dot = float(np.cos(FLEXION_SURFACE_HALF_ANGLE))

    triangles_world_parts: list[np.ndarray] = []
    normals_world_parts: list[np.ndarray] = []
    areas_parts: list[np.ndarray] = []
    for geom_id in mesh_geom_ids:
        triangles_local, triangle_normals_local, triangle_areas = _mesh_triangle_data_body_local(model, geom_id)
        if len(triangles_local) == 0:
            continue
        triangles_world = body_pos[None, None, :] + np.einsum("ij,ntj->nti", body_rot, triangles_local)
        normals_world = np.einsum("ij,nj->ni", body_rot, triangle_normals_local)
        centroids_world = np.mean(triangles_world, axis=1)
        axis_positions = np.einsum("nj,j->n", centroids_world - center[None, :], axis)
        normal_dots = np.einsum("nj,j->n", normals_world, palm_side)
        valid_mask = normal_dots >= min_normal_dot
        if excluded_axis_intervals:
            valid_mask &= np.array(
                [_axis_position_allowed(float(axis_pos), excluded_axis_intervals) for axis_pos in axis_positions],
                dtype=bool,
            )
        if not np.any(valid_mask):
            continue
        triangles_world_parts.append(triangles_world[valid_mask])
        normals_world_parts.append(normals_world[valid_mask])
        areas_parts.append(triangle_areas[valid_mask])

    if not triangles_world_parts:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)

    triangles_world = np.concatenate(triangles_world_parts, axis=0)
    normals_world = np.concatenate(normals_world_parts, axis=0)
    triangle_areas = np.concatenate(areas_parts, axis=0)
    candidate_count = max(point_count * 24, point_count + 32, 256)
    candidate_points, candidate_normals = _sample_triangle_points_world(
        rng,
        triangles_world=triangles_world,
        triangle_normals_world=normals_world,
        triangle_areas=triangle_areas,
        point_count=candidate_count,
    )
    if len(candidate_points) < point_count:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)

    sampled_indices = _farthest_point_sample_indices(candidate_points, point_count)
    return candidate_points[sampled_indices], candidate_normals[sampled_indices]


def _segment_surface_direction(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    finger: str,
    segment: str,
    geom_id: int,
    palm_point: np.ndarray,
    palm_view_axis: np.ndarray,
    thumb_reference_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = data.geom_xpos[geom_id]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    geom_axis = _normalize(rotation[:, 2])
    half_length = model.geom_size[geom_id, 1]
    body_id = model.geom_bodyid[geom_id]

    endpoint_a = center - geom_axis * half_length
    endpoint_b = center + geom_axis * half_length
    if segment == "1":
        distal_targets = [
            data.xpos[child_body_id].copy()
            for child_body_id in range(model.nbody)
            if model.body_parentid[child_body_id] == body_id
        ]
        distal_target = np.mean(np.stack(distal_targets, axis=0), axis=0)
    else:
        tip_site_name = f"inspire_{side}_{finger}_tip"
        tip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tip_site_name)
        distal_target = data.site_xpos[tip_site_id].copy()

    if np.linalg.norm(endpoint_a - distal_target) <= np.linalg.norm(endpoint_b - distal_target):
        distal_endpoint = endpoint_a
        proximal_endpoint = endpoint_b
    else:
        distal_endpoint = endpoint_b
        proximal_endpoint = endpoint_a

    axis = _normalize(distal_endpoint - proximal_endpoint)
    body_rotation = data.xmat[body_id].reshape(3, 3)
    joint_id = model.body_jntadr[body_id]
    joint_axis = _normalize(body_rotation @ model.jnt_axis[joint_id])

    # The contact band should lie on the side swept by flexion around the hinge axis.
    palm_side = np.cross(joint_axis, axis)
    if np.linalg.norm(palm_side) < 1e-8:
        palm_side = rotation[:, 1]

    if finger == "thumb":
        reference = thumb_reference_point - center
    else:
        reference = palm_view_axis.copy()
    reference = reference - axis * np.dot(reference, axis)
    if np.linalg.norm(reference) < 1e-8:
        to_palm = palm_point - center
        reference = to_palm - axis * np.dot(to_palm, axis)
    if np.linalg.norm(reference) < 1e-8:
        reference = rotation[:, 1]

    palm_side = _normalize(palm_side)
    reference = _normalize(reference)
    if np.dot(palm_side, reference) < 0.0:
        palm_side = -palm_side

    tangent = np.cross(axis, palm_side)
    if np.linalg.norm(tangent) < 1e-8:
        tangent = rotation[:, 0]
    tangent = _normalize(tangent)
    return center, axis, palm_side, tangent


def _sample_capsule_palm_strip_world_points(
    center: np.ndarray,
    axis: np.ndarray,
    palm_side: np.ndarray,
    tangent: np.ndarray,
    radius: float,
    half_length: float,
    point_count: int,
    excluded_axis_intervals: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    excluded_axis_intervals = _merge_axis_intervals(
        excluded_axis_intervals or [],
        lower_bound=-half_length,
        upper_bound=half_length,
    )
    usable_length = 2.0 * half_length - sum(end - start for start, end in excluded_axis_intervals)
    usable_length = max(usable_length, 1e-4)
    strip_area = radius * (2.0 * FLEXION_SURFACE_HALF_ANGLE) * usable_length
    candidate_target = max(point_count * 12, point_count + 16, 96)
    spacing = np.sqrt(strip_area / max(candidate_target, 1))
    arc_length = radius * (2.0 * FLEXION_SURFACE_HALF_ANGLE)
    n_axis = max(2, int(np.ceil(usable_length / spacing)))
    n_angle = max(3, int(np.ceil(arc_length / spacing)))

    axis_positions = _valid_axis_positions(half_length, n_axis, excluded_axis_intervals)
    candidate_points = _build_capsule_strip_candidate_world_points(
        center=center,
        axis=axis,
        palm_side=palm_side,
        tangent=tangent,
        radius=radius,
        axis_positions=axis_positions,
        n_angle=n_angle,
    )

    while len(candidate_points) < max(point_count, candidate_target):
        if len(axis_positions) == 0:
            n_axis += 1
        else:
            axis_spacing = usable_length / max(len(axis_positions), 1)
            angle_spacing = arc_length / max(n_angle, 1)
            if axis_spacing >= angle_spacing:
                n_axis += 1
            else:
                n_angle += 1

        if n_axis > 1024 or n_angle > 1024:
            break

        axis_positions = _valid_axis_positions(half_length, n_axis, excluded_axis_intervals)
        candidate_points = _build_capsule_strip_candidate_world_points(
            center=center,
            axis=axis,
            palm_side=palm_side,
            tangent=tangent,
            radius=radius,
            axis_positions=axis_positions,
            n_angle=n_angle,
        )

    if len(candidate_points) < point_count:
        raise ValueError("No valid finger-surface samples remain after joint filtering.")

    return _farthest_point_sample(candidate_points, point_count)


def _palm_surface_point(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    site_name = f"inspire_{side}_palm"
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xpos[site_id].copy()


def _palm_view_axis(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    site_name = f"inspire_{side}_palm"
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return data.site_xmat[site_id].reshape(3, 3)[:, 0].copy()


def _average_fingertip_point(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    fingertip_points = []
    for finger in FINGERS:
        site_name = f"inspire_{side}_{finger}_tip"
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        fingertip_points.append(data.site_xpos[site_id].copy())
    return np.mean(np.stack(fingertip_points, axis=0), axis=0)


def _other_fingertip_centroid(model: mujoco.MjModel, data: mujoco.MjData, side: str) -> np.ndarray:
    fingertip_points = []
    for finger in ("index", "middle", "ring", "pinky"):
        site_name = f"inspire_{side}_{finger}_tip"
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        fingertip_points.append(data.site_xpos[site_id].copy())
    return np.mean(np.stack(fingertip_points, axis=0), axis=0)


def _segment_joint_clearance_intervals(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
    center: np.ndarray,
    axis: np.ndarray,
    half_length: float,
) -> list[tuple[float, float]]:
    joint_axis_positions = [float(np.dot(data.xpos[body_id] - center, axis))]
    for child_body_id in range(model.nbody):
        if model.body_parentid[child_body_id] == body_id:
            joint_axis_positions.append(float(np.dot(data.xpos[child_body_id] - center, axis)))

    intervals = [
        (axis_pos - JOINT_AXIS_CLEARANCE, axis_pos + JOINT_AXIS_CLEARANCE)
        for axis_pos in joint_axis_positions
    ]
    return _merge_axis_intervals(intervals, lower_bound=-half_length, upper_bound=half_length)


def _segment_normal_alignment_report(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
) -> list[str]:
    palm_point = _palm_surface_point(model, data, side)
    palm_view_axis = _palm_view_axis(model, data, side)
    thumb_reference_point = _other_fingertip_centroid(model, data, side)
    report = []

    for finger in FINGERS:
        normals = {}
        for segment in ("1", "0"):
            geom_name = f"inspire_collision_hand_{side}_{finger}_{segment}"
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            _, _, palm_side, _ = _segment_surface_direction(
                model,
                data,
                side=side,
                finger=finger,
                segment=segment,
                geom_id=geom_id,
                palm_point=palm_point,
                palm_view_axis=palm_view_axis,
                thumb_reference_point=thumb_reference_point,
            )
            normals[segment] = _normalize(palm_side)

        dot = float(np.dot(normals["1"], normals["0"]))
        angle_deg = float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))
        report.append(f"{finger}:{angle_deg:.1f}deg")

    return report


def compute_finger_surface_point_records(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    total_point_count: int,
    point_count_per_segment: int | None = None,
    surface_source: str = "collision",
) -> tuple[list[dict[str, object]], np.ndarray]:
    if surface_source not in {"collision", "visual"}:
        raise ValueError("surface_source must be 'collision' or 'visual'.")

    palm_point = _palm_surface_point(model, data, side)
    palm_view_axis = _palm_view_axis(model, data, side)
    thumb_reference_point = _other_fingertip_centroid(model, data, side)
    target_counts = _segment_target_counts(
        model,
        side,
        total_point_count,
        point_count_per_segment=point_count_per_segment,
    )
    point_records: list[dict[str, object]] = []
    segment_frames: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    side_seed = 1000 if side == "right" else 2000

    for finger, segment in SEGMENT_SPECS:
        geom_name = f"inspire_collision_hand_{side}_{finger}_{segment}"
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        segment_frames[(finger, segment)] = _segment_surface_direction(
            model,
            data,
            side=side,
            finger=finger,
            segment=segment,
            geom_id=geom_id,
            palm_point=palm_point,
            palm_view_axis=palm_view_axis,
            thumb_reference_point=thumb_reference_point,
        )

    for finger, segment in SEGMENT_SPECS:
        geom_name = f"inspire_collision_hand_{side}_{finger}_{segment}"
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        center, axis, palm_side, tangent = segment_frames[(finger, segment)]
        radius = model.geom_size[geom_id, 0]
        half_length = model.geom_size[geom_id, 1]
        body_id = model.geom_bodyid[geom_id]
        excluded_axis_intervals = _segment_joint_clearance_intervals(
            model,
            data,
            body_id=body_id,
            center=center,
            axis=axis,
            half_length=half_length,
        )
        target_count = target_counts[(finger, segment)]
        world_points = np.zeros((0, 3), dtype=float)
        world_normals = np.zeros((0, 3), dtype=float)
        if surface_source == "visual":
            rng = np.random.default_rng(side_seed + 17 * SEGMENT_SPECS.index((finger, segment)))
            world_points, world_normals = _sample_visual_segment_surface_world_points(
                model,
                data,
                body_id=body_id,
                center=center,
                axis=axis,
                palm_side=palm_side,
                point_count=target_count,
                excluded_axis_intervals=excluded_axis_intervals,
                rng=rng,
            )

        if len(world_points) == 0:
            world_points = _sample_capsule_palm_strip_world_points(
                center=center,
                axis=axis,
                palm_side=palm_side,
                tangent=tangent,
                radius=radius,
                half_length=half_length,
                point_count=target_count,
                excluded_axis_intervals=excluded_axis_intervals,
            )
            world_normals = np.stack(
                [
                    _normalize(world_point - center - axis * np.dot(world_point - center, axis))
                    for world_point in world_points
                ],
                axis=0,
            )

        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        body_pos = data.xpos[body_id]
        body_rot = data.xmat[body_id].reshape(3, 3)

        for world_point, world_normal in zip(world_points, world_normals, strict=True):
            final_world_point = world_point + POINT_NORMAL_OFFSET * world_normal
            point_records.append(
                {
                    "finger": finger,
                    "segment": segment,
                    "role": _segment_role(finger, segment),
                    "body_name": body_name,
                    "local_pos": body_rot.T @ (final_world_point - body_pos),
                    "local_normal": body_rot.T @ world_normal,
                    "surface_world_pos": world_point,
                    "world_pos": final_world_point,
                    "world_normal": world_normal,
                }
            )

    return point_records, palm_point


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


def _hide_hand_visual_geoms(model: mujoco.MjModel) -> None:
    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not body_name.startswith("inspire_"):
            continue
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if int(model.geom_group[geom_id]) in (0, 1):
            model.geom_group[geom_id] = 4


def build_surface_point_model(
    side: str,
    hand_point_count: int,
    surface_source: str = "collision",
) -> tuple[
    mujoco.MjModel,
    mujoco.MjData,
    list[dict[str, object]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    spec = build_franka_inspire_spec(side)
    base_model = spec.compile()
    base_data = mujoco.MjData(base_model)
    initialize_state(base_model, base_data)

    hand_point_records, palm_point = compute_finger_surface_point_records(
        base_model,
        base_data,
        side=side,
        total_point_count=hand_point_count,
        surface_source=surface_source,
    )
    palm_view_axis = _palm_view_axis(base_model, base_data, side)
    fingertip_point = _average_fingertip_point(base_model, base_data, side)

    for point_index, point_record in enumerate(hand_point_records):
        body = spec.body(point_record["body_name"])
        _add_marker_geom_to_body(
            body,
            name=f"surface_point_hand_{point_index:04d}",
            pos=point_record["local_pos"],
            radius=HAND_POINT_RADIUS,
            rgba=HAND_ROLE_COLORS[point_record["role"]],
        )

    model = spec.compile()
    data = mujoco.MjData(model)
    _hide_hand_visual_geoms(model)
    initialize_state(model, data)
    return model, data, hand_point_records, palm_point, palm_view_axis, fingertip_point


def _configure_palm_camera(
    cam: mujoco.MjvCamera,
    hand_point_records: list[dict[str, object]],
    palm_point: np.ndarray,
    palm_view_axis: np.ndarray,
    fingertip_point: np.ndarray,
) -> None:
    hand_world_points = np.stack([record["world_pos"] for record in hand_point_records], axis=0)
    lookat = np.mean(hand_world_points, axis=0)
    finger_direction = _normalize(fingertip_point - palm_point)
    if np.linalg.norm(finger_direction) < 1e-8:
        finger_direction = _normalize(lookat - palm_point)
    if np.linalg.norm(finger_direction) < 1e-8:
        finger_direction = np.array([0.0, 0.0, -1.0], dtype=float)

    camera_pos = (
        palm_point
        + palm_view_axis * PALM_VIEW_NORMAL_OFFSET
        + finger_direction * PALM_VIEW_FINGER_OFFSET
    )
    view_direction = _normalize(lookat - camera_pos)
    distance = np.linalg.norm(lookat - camera_pos)

    cam.lookat = lookat
    cam.distance = distance
    cam.azimuth = np.degrees(np.arctan2(view_direction[1], view_direction[0]))
    cam.elevation = np.degrees(np.arcsin(np.clip(view_direction[2], -1.0, 1.0)))


def _surface_point_scene_option() -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = np.array([1, 0, 1, 1, 0, 0], dtype=np.uint8)
    return opt


def save_snapshot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output_path: Path,
    hand_point_records: list[dict[str, object]],
    palm_point: np.ndarray,
    palm_view_axis: np.ndarray,
    fingertip_point: np.ndarray,
    view: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=720, width=960)
    camera = mujoco.MjvCamera()
    scene_option = _surface_point_scene_option()
    if view == "palm":
        _configure_palm_camera(camera, hand_point_records, palm_point, palm_view_axis, fingertip_point)
    else:
        _configure_free_camera(camera)
    renderer.update_scene(data, camera=camera, scene_option=scene_option)
    Image.fromarray(renderer.render()).save(output_path)
    renderer.close()


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        _configure_free_camera(viewer.cam)
        viewer.opt.geomgroup[:] = _surface_point_scene_option().geomgroup
        while viewer.is_running():
            viewer.sync()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize finger flexion-surface points on the Inspire hand."
    )
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument(
        "--hand-point-count",
        type=int,
        default=HAND_SURFACE_POINT_COUNT,
        help="Approximate total number of finger-surface points across ten finger segments.",
    )
    parser.add_argument(
        "--surface-source",
        choices=("collision", "visual"),
        default="visual",
        help="Sample hand surface points from collision primitives or visual meshes.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Render one offscreen image and exit instead of launching the viewer.",
    )
    parser.add_argument(
        "--snapshot-view",
        choices=("palm", "scene"),
        default="palm",
        help="Snapshot camera preset. palm: look from the palm side toward the sampled finger surfaces.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, data, hand_point_records, palm_point, palm_view_axis, fingertip_point = build_surface_point_model(
        side=args.hand,
        hand_point_count=args.hand_point_count,
        surface_source=args.surface_source,
    )

    segment_labels = [_segment_label(record["finger"], record["segment"]) for record in hand_point_records]
    unique_segments = []
    for label in segment_labels:
        if label not in unique_segments:
            unique_segments.append(label)

    print(f"Finger surface points: {len(hand_point_records)}")
    print("Finger segments     :", ", ".join(unique_segments))
    print("Hand point colors   : proximal=orange, intermediate=green, distal=blue")
    print(f"Normal offset       : {POINT_NORMAL_OFFSET:.3f} m")
    print(f"Joint clearance     : {JOINT_AXIS_CLEARANCE:.3f} m")
    print("Normal alignment    :", ", ".join(_segment_normal_alignment_report(model, data, args.hand)))

    if args.snapshot is not None:
        save_snapshot(
            model,
            data,
            output_path=args.snapshot,
            hand_point_records=hand_point_records,
            palm_point=palm_point,
            palm_view_axis=palm_view_axis,
            fingertip_point=fingertip_point,
            view=args.snapshot_view,
        )
        print(f"Saved snapshot to: {args.snapshot}")
        return

    print("Close the viewer window to exit.")
    run_viewer(model, data)


if __name__ == "__main__":
    main()
