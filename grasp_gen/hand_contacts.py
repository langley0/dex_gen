from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mujoco
import numpy as np

from .hand import CONTACT_CLOUD_SCALE, SEGS


HALF_ANGLE = np.deg2rad(55.0)
JOINT_CLEARANCE = 5.0e-3
SURFACE_COUNT_FACTOR = 4
SEGMENT_POINT_FACTOR = 8
RNG_SEED_BASE = 1729


class _HandLike(Protocol):
    side: str
    model: mujoco.MjModel
    data: mujoco.MjData
    root_body_id: int
    palm_site_id: int
    tip_site_ids: dict[str, int]
    segment_geom_ids: dict[tuple[str, str], int]

    def apply_state(
        self,
        qpos: np.ndarray | None = None,
        ctrl: np.ndarray | None = None,
        root_pos: np.ndarray | None = None,
        root_quat: np.ndarray | None = None,
    ) -> None: ...


@dataclass(frozen=True)
class ContactConfig:
    n_per_seg: int = 10
    thumb_weight: float = 1.0
    palm_clearance: float = 8.0e-3
    target_spacing: float = 5.0e-3
    cloud_scale: float = CONTACT_CLOUD_SCALE


@dataclass(frozen=True)
class SurfaceRecord:
    finger: str
    segment: str
    role: str
    body_id: int
    body_name: str
    geom_id: int
    geom_name: str
    local_pos: np.ndarray
    local_normal: np.ndarray
    world_pos: np.ndarray
    world_normal: np.ndarray
    root_pos: np.ndarray


@dataclass(frozen=True)
class ContactRecord:
    finger: str
    segment: str
    role: str
    body_id: int
    body_name: str
    local_pos: np.ndarray
    local_normal: np.ndarray
    world_pos: np.ndarray
    world_normal: np.ndarray
    root_pos: np.ndarray
    weight: float


@dataclass(frozen=True)
class SegmentFrame:
    center: np.ndarray
    axis: np.ndarray
    face: np.ndarray
    tangent: np.ndarray
    body_id: int
    radius: float
    half: float


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    return vector.copy() if norm < 1.0e-8 else vector / norm


def _merge(intervals: list[tuple[float, float]], lo: float, hi: float) -> list[tuple[float, float]]:
    clipped = sorted((max(lo, a), min(hi, b)) for a, b in intervals if min(hi, b) > max(lo, a))
    if not clipped:
        return []
    merged = [clipped[0]]
    for a, b in clipped[1:]:
        x, y = merged[-1]
        if a <= y:
            merged[-1] = (x, max(y, b))
        else:
            merged.append((a, b))
    return merged


def _fps_indices(points: np.ndarray, n: int) -> np.ndarray:
    if n <= 0 or len(points) == 0:
        return np.zeros((0,), dtype=np.intp)
    if len(points) <= n:
        return np.arange(len(points), dtype=np.intp)

    keep = np.empty(n, dtype=np.intp)
    center = np.mean(points, axis=0)
    keep[0] = int(np.argmin(np.sum((points - center) ** 2, axis=1)))
    dist2 = np.sum((points - points[keep[0]]) ** 2, axis=1)
    dist2[keep[0]] = -np.inf

    for i in range(1, n):
        keep[i] = int(np.argmax(dist2))
        dist2 = np.minimum(dist2, np.sum((points - points[keep[i]]) ** 2, axis=1))
        dist2[keep[: i + 1]] = -np.inf
    return keep


def _tip_mean(hand: _HandLike, fingers: tuple[str, ...]) -> np.ndarray:
    return np.mean(np.stack([hand.data.site_xpos[hand.tip_site_ids[finger]].copy() for finger in fingers], axis=0), axis=0)


def _joint_x(hand: _HandLike, body_id: int, center: np.ndarray, axis: np.ndarray) -> float:
    joint_count = int(hand.model.body_jntnum[body_id])
    if joint_count <= 0:
        return float(np.dot(hand.data.xpos[body_id] - center, axis))
    joint_id = int(hand.model.body_jntadr[body_id])
    return float(np.dot(hand.data.xanchor[joint_id] - center, axis))


def _blocked(hand: _HandLike, frame: SegmentFrame, segment: str, cfg: ContactConfig) -> list[tuple[float, float]]:
    xs = [_joint_x(hand, frame.body_id, frame.center, frame.axis)]
    for child_id in range(hand.model.nbody):
        if hand.model.body_parentid[child_id] == frame.body_id:
            xs.append(_joint_x(hand, child_id, frame.center, frame.axis))
    intervals = [(x - JOINT_CLEARANCE, x + JOINT_CLEARANCE) for x in xs]
    if segment == "1":
        intervals.append((-frame.half, xs[0] + cfg.palm_clearance))
    return _merge(intervals, -frame.half, frame.half)


def _frame(hand: _HandLike, finger: str, segment: str, palm_axis: np.ndarray, thumb_ref: np.ndarray) -> SegmentFrame:
    geom_id = hand.segment_geom_ids[(finger, segment)]
    center = hand.data.geom_xpos[geom_id].copy()
    rotation = hand.data.geom_xmat[geom_id].reshape(3, 3)
    axis0 = _unit(rotation[:, 2])
    half = float(hand.model.geom_size[geom_id, 1])
    body_id = int(hand.model.geom_bodyid[geom_id])
    radius = float(hand.model.geom_size[geom_id, 0])

    endpoint_a = center - half * axis0
    endpoint_b = center + half * axis0
    if segment == "1":
        target = np.mean(
            np.stack(
                [hand.data.xpos[i].copy() for i in range(hand.model.nbody) if hand.model.body_parentid[i] == body_id],
                axis=0,
            ),
            axis=0,
        )
    else:
        target = hand.data.site_xpos[hand.tip_site_ids[finger]].copy()
    axis = _unit(endpoint_a - endpoint_b if np.linalg.norm(endpoint_a - target) < np.linalg.norm(endpoint_b - target) else endpoint_b - endpoint_a)

    joint_id = int(hand.model.body_jntadr[body_id])
    joint_axis = _unit(hand.data.xmat[body_id].reshape(3, 3) @ hand.model.jnt_axis[joint_id])
    face = np.cross(joint_axis, axis)
    if np.linalg.norm(face) < 1.0e-8:
        face = rotation[:, 1]

    ref = thumb_ref - center if finger == "thumb" else palm_axis.copy()
    ref = ref - axis * np.dot(ref, axis)
    if np.linalg.norm(ref) < 1.0e-8:
        ref = hand.data.site_xpos[hand.palm_site_id] - center
        ref = ref - axis * np.dot(ref, axis)
    face = _unit(face)
    ref = _unit(ref)
    if np.dot(face, ref) < 0.0:
        face = -face

    tangent = _unit(np.cross(axis, face))
    if np.linalg.norm(tangent) < 1.0e-8:
        tangent = _unit(rotation[:, 0])

    return SegmentFrame(
        center=center,
        axis=axis,
        face=face,
        tangent=tangent,
        body_id=body_id,
        radius=radius,
        half=half,
    )


def _role(finger: str, segment: str) -> str:
    if finger == "palm":
        return "palm"
    if segment == "1":
        return "proximal"
    return "distal" if finger == "thumb" else "intermediate"


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


def _surface_count(area: float, spacing: float, *, min_count: int = 1) -> int:
    if area <= 0.0 or spacing <= 0.0:
        return int(max(min_count, 1))
    return max(int(np.ceil(area / (spacing * spacing))), int(min_count), 1)


def _sample_box_surface_local(
    rng: np.random.Generator,
    half_extents: np.ndarray,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_count <= 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)

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
    positions = np.zeros((sample_count, 3), dtype=float)
    normals = np.zeros((sample_count, 3), dtype=float)

    mask = face_ids == 0
    positions[mask] = np.column_stack([np.full(np.sum(mask), hx), uv[mask, 0] * hy, uv[mask, 1] * hz])
    normals[mask] = np.array([1.0, 0.0, 0.0], dtype=float)
    mask = face_ids == 1
    positions[mask] = np.column_stack([np.full(np.sum(mask), -hx), uv[mask, 0] * hy, uv[mask, 1] * hz])
    normals[mask] = np.array([-1.0, 0.0, 0.0], dtype=float)
    mask = face_ids == 2
    positions[mask] = np.column_stack([uv[mask, 0] * hx, np.full(np.sum(mask), hy), uv[mask, 1] * hz])
    normals[mask] = np.array([0.0, 1.0, 0.0], dtype=float)
    mask = face_ids == 3
    positions[mask] = np.column_stack([uv[mask, 0] * hx, np.full(np.sum(mask), -hy), uv[mask, 1] * hz])
    normals[mask] = np.array([0.0, -1.0, 0.0], dtype=float)
    mask = face_ids == 4
    positions[mask] = np.column_stack([uv[mask, 0] * hx, uv[mask, 1] * hy, np.full(np.sum(mask), hz)])
    normals[mask] = np.array([0.0, 0.0, 1.0], dtype=float)
    mask = face_ids == 5
    positions[mask] = np.column_stack([uv[mask, 0] * hx, uv[mask, 1] * hy, np.full(np.sum(mask), -hz)])
    normals[mask] = np.array([0.0, 0.0, -1.0], dtype=float)
    return positions, normals


def _sample_sphere_surface_local(
    rng: np.random.Generator,
    radius: float,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_count <= 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)
    normals = rng.normal(size=(sample_count, 3))
    normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12, None)
    return radius * normals, normals


def _sample_capsule_surface_local(
    rng: np.random.Generator,
    radius: float,
    half_length: float,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_count <= 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)

    cylinder_area = 4.0 * np.pi * radius * half_length
    cap_area = 4.0 * np.pi * radius * radius
    total_area = cylinder_area + cap_area
    if total_area <= 0.0:
        return np.zeros((sample_count, 3), dtype=float), np.zeros((sample_count, 3), dtype=float)

    use_cylinder = rng.random(sample_count) < (cylinder_area / total_area)
    positions = np.zeros((sample_count, 3), dtype=float)
    normals = np.zeros((sample_count, 3), dtype=float)

    cylinder_count = int(np.sum(use_cylinder))
    if cylinder_count > 0:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=cylinder_count)
        z = rng.uniform(-half_length, half_length, size=cylinder_count)
        positions[use_cylinder] = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])
        normals[use_cylinder] = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(cylinder_count, dtype=float)])

    cap_count = sample_count - cylinder_count
    if cap_count > 0:
        cap_normals = rng.normal(size=(cap_count, 3))
        cap_normals /= np.clip(np.linalg.norm(cap_normals, axis=1, keepdims=True), 1.0e-12, None)
        cap_centers = np.zeros((cap_count, 3), dtype=float)
        cap_centers[:, 2] = np.where(cap_normals[:, 2] >= 0.0, half_length, -half_length)
        positions[~use_cylinder] = cap_centers + radius * cap_normals
        normals[~use_cylinder] = cap_normals

    return positions, normals


def _sample_cylinder_surface_local(
    rng: np.random.Generator,
    radius: float,
    half_height: float,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_count <= 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)

    side_area = 4.0 * np.pi * radius * half_height
    cap_area = 2.0 * np.pi * radius * radius
    total_area = side_area + cap_area
    if total_area <= 0.0:
        return np.zeros((sample_count, 3), dtype=float), np.zeros((sample_count, 3), dtype=float)

    use_side = rng.random(sample_count) < (side_area / total_area)
    positions = np.zeros((sample_count, 3), dtype=float)
    normals = np.zeros((sample_count, 3), dtype=float)

    side_count = int(np.sum(use_side))
    if side_count > 0:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=side_count)
        z = rng.uniform(-half_height, half_height, size=side_count)
        positions[use_side] = np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z])
        normals[use_side] = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(side_count, dtype=float)])

    cap_count = sample_count - side_count
    if cap_count > 0:
        theta = rng.uniform(0.0, 2.0 * np.pi, size=cap_count)
        radial = radius * np.sqrt(rng.uniform(0.0, 1.0, size=cap_count))
        z = np.where(rng.random(cap_count) < 0.5, half_height, -half_height)
        positions[~use_side] = np.column_stack([radial * np.cos(theta), radial * np.sin(theta), z])
        normals[~use_side] = np.column_stack(
            [
                np.zeros(cap_count, dtype=float),
                np.zeros(cap_count, dtype=float),
                np.where(z > 0.0, 1.0, -1.0),
            ]
        )

    return positions, normals


def _sample_geom_surface_local(
    rng: np.random.Generator,
    geom_type: int,
    geom_size: np.ndarray,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
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
    return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)


def _collision_geom_ids(hand: _HandLike) -> list[int]:
    prefix = f"collision_hand_{hand.side}_"
    geom_ids: list[int] = []
    for geom_id in range(hand.model.ngeom):
        geom_name = mujoco.mj_id2name(hand.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if geom_name.startswith(prefix):
            geom_ids.append(geom_id)
    return geom_ids


def _geom_labels(hand: _HandLike, geom_id: int) -> tuple[str, str, str]:
    prefix = f"collision_hand_{hand.side}_"
    geom_name = mujoco.mj_id2name(hand.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
    suffix = geom_name[len(prefix) :] if geom_name.startswith(prefix) else geom_name
    parts = suffix.split("_")
    if len(parts) >= 2:
        finger = parts[0]
        segment = parts[1]
    else:
        finger = "unknown"
        segment = "0"
    return finger, segment, _role(finger, segment)


def _world_to_root(root_pos: np.ndarray, root_rot: np.ndarray, world_positions: np.ndarray) -> np.ndarray:
    return (world_positions - root_pos[None, :]) @ root_rot


def _root_to_world(root_pos: np.ndarray, root_rot: np.ndarray, root_positions: np.ndarray) -> np.ndarray:
    return root_pos[None, :] + root_positions @ root_rot.T


def build_surface_cloud(
    hand: _HandLike,
    *,
    qpos: np.ndarray | None = None,
    ctrl: np.ndarray | None = None,
    cfg: ContactConfig | None = None,
) -> list[SurfaceRecord]:
    cfg = ContactConfig() if cfg is None else cfg
    if cfg.target_spacing <= 0.0:
        raise ValueError("cfg.target_spacing must be positive")
    if cfg.cloud_scale <= 0.0:
        raise ValueError("cfg.cloud_scale must be positive")

    hand.apply_state(qpos=qpos, ctrl=ctrl)
    root_pos = hand.data.xpos[hand.root_body_id].copy()
    root_rot = hand.data.xmat[hand.root_body_id].reshape(3, 3)

    records: list[SurfaceRecord] = []
    for geom_id in _collision_geom_ids(hand):
        geom_type = int(hand.model.geom_type[geom_id])
        geom_size = hand.model.geom_size[geom_id]
        geom_area = _geom_surface_area(geom_type, geom_size)
        geom_name = mujoco.mj_id2name(hand.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
        finger, segment, role = _geom_labels(hand, geom_id)
        min_count = cfg.n_per_seg * SEGMENT_POINT_FACTOR if (finger, segment) in SEGS else 1
        target_count = _surface_count(geom_area, float(cfg.target_spacing), min_count=min_count)
        candidate_count = max(target_count * SURFACE_COUNT_FACTOR, target_count)

        rng = np.random.default_rng(RNG_SEED_BASE + int(geom_id))
        geom_local_positions, geom_local_normals = _sample_geom_surface_local(rng, geom_type, geom_size, candidate_count)
        if geom_local_positions.size == 0:
            continue

        keep = _fps_indices(geom_local_positions, target_count)
        geom_local_positions = geom_local_positions[keep]
        geom_local_normals = geom_local_normals[keep]

        body_id = int(hand.model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(hand.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        body_pos = hand.data.xpos[body_id].copy()
        body_rot = hand.data.xmat[body_id].reshape(3, 3)
        geom_rot = np.zeros(9, dtype=float)
        mujoco.mju_quat2Mat(geom_rot, hand.model.geom_quat[geom_id])
        geom_rot = geom_rot.reshape(3, 3)
        geom_pos = hand.model.geom_pos[geom_id].copy()

        body_local_positions = geom_pos[None, :] + geom_local_positions @ geom_rot.T
        body_local_normals = geom_local_normals @ geom_rot.T
        world_positions = body_pos[None, :] + body_local_positions @ body_rot.T
        world_normals = body_local_normals @ body_rot.T
        root_positions = _world_to_root(root_pos, root_rot, world_positions)
        scaled_world_positions = _root_to_world(root_pos, root_rot, float(cfg.cloud_scale) * root_positions)
        scaled_body_local_positions = (scaled_world_positions - body_pos[None, :]) @ body_rot

        for local_pos, local_normal, world_pos, world_normal, root_point in zip(
            scaled_body_local_positions,
            body_local_normals,
            scaled_world_positions,
            world_normals,
            float(cfg.cloud_scale) * root_positions,
            strict=True,
        ):
            records.append(
                SurfaceRecord(
                    finger=finger,
                    segment=segment,
                    role=role,
                    body_id=body_id,
                    body_name=body_name,
                    geom_id=geom_id,
                    geom_name=geom_name,
                    local_pos=np.asarray(local_pos, dtype=float),
                    local_normal=_unit(local_normal),
                    world_pos=np.asarray(world_pos, dtype=float),
                    world_normal=_unit(world_normal),
                    root_pos=np.asarray(root_point, dtype=float),
                )
            )
    return records


def _contact_mask(records: list[SurfaceRecord], frame: SegmentFrame, blocked: list[tuple[float, float]]) -> np.ndarray:
    if not records:
        return np.zeros((0,), dtype=bool)
    points = np.asarray([record.world_pos for record in records], dtype=float)
    delta = points - frame.center[None, :]
    axial = delta @ frame.axis
    radial = delta - axial[:, None] * frame.axis[None, :]
    radial_norm = np.linalg.norm(radial, axis=1)
    radial_unit = radial / np.clip(radial_norm[:, None], 1.0e-12, None)

    mask = np.abs(axial) <= (frame.half + 1.0e-6)
    for lo, hi in blocked:
        mask &= (axial < lo) | (axial > hi)
    mask &= radial_norm > 1.0e-6
    mask &= (radial_unit @ frame.face) >= np.cos(HALF_ANGLE)
    return mask


def sample_contacts(
    hand: _HandLike,
    surface_records: list[SurfaceRecord],
    *,
    cfg: ContactConfig | None = None,
) -> list[ContactRecord]:
    cfg = ContactConfig() if cfg is None else cfg
    if cfg.n_per_seg <= 0:
        raise ValueError("cfg.n_per_seg must be positive")

    palm_axis = hand.data.site_xmat[hand.palm_site_id].reshape(3, 3)[:, 0].copy()
    thumb_ref = _tip_mean(hand, ("index", "middle", "ring", "pinky"))
    per_segment: dict[tuple[str, str], list[SurfaceRecord]] = {key: [] for key in SEGS}
    for record in surface_records:
        key = (record.finger, record.segment)
        if key in per_segment:
            per_segment[key].append(record)

    contacts: list[ContactRecord] = []
    for finger, segment in SEGS:
        frame = _frame(hand, finger, segment, palm_axis, thumb_ref)
        blocked = _blocked(hand, frame, segment, cfg)
        candidates = per_segment[(finger, segment)]
        mask = _contact_mask(candidates, frame, blocked)
        filtered = [record for record, keep in zip(candidates, mask, strict=True) if keep]

        if len(filtered) < cfg.n_per_seg:
            relaxed = [record for record in candidates if np.abs(np.dot(record.world_pos - frame.center, frame.axis)) <= (frame.half + 1.0e-6)]
            filtered = relaxed if len(relaxed) >= cfg.n_per_seg else candidates
        if len(filtered) < cfg.n_per_seg:
            raise ValueError(f"Not enough dense cloud points to sample {cfg.n_per_seg} contacts for {finger}_{segment}.")

        filtered_points = np.asarray([record.world_pos for record in filtered], dtype=float)
        keep = _fps_indices(filtered_points, int(cfg.n_per_seg))
        for index in keep.tolist():
            record = filtered[index]
            contacts.append(
                ContactRecord(
                    finger=record.finger,
                    segment=record.segment,
                    role=record.role,
                    body_id=record.body_id,
                    body_name=record.body_name,
                    local_pos=record.local_pos.copy(),
                    local_normal=record.local_normal.copy(),
                    world_pos=record.world_pos.copy(),
                    world_normal=record.world_normal.copy(),
                    root_pos=record.root_pos.copy(),
                    weight=cfg.thumb_weight if record.finger == "thumb" else 1.0,
                )
            )
    return contacts


def build_contacts(
    hand: _HandLike,
    *,
    qpos: np.ndarray | None = None,
    ctrl: np.ndarray | None = None,
    cfg: ContactConfig | None = None,
) -> list[ContactRecord]:
    surface_records = build_surface_cloud(hand, qpos=qpos, ctrl=ctrl, cfg=cfg)
    return sample_contacts(hand, surface_records, cfg=cfg)


def contact_points_root(
    hand: _HandLike,
    *,
    qpos: np.ndarray | None = None,
    ctrl: np.ndarray | None = None,
    cfg: ContactConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    records = build_contacts(hand, qpos=qpos, ctrl=ctrl, cfg=cfg)
    return (
        np.asarray([record.root_pos for record in records], dtype=float),
        np.asarray([record.weight for record in records], dtype=float),
    )
