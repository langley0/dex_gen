from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .hand import Hand, SEGS


HALF_ANGLE = np.deg2rad(55.0)
POINT_OFFSET = 2.0e-3
JOINT_CLEARANCE = 5.0e-3


@dataclass(frozen=True)
class ContactConfig:
    n_per_seg: int = 4
    thumb_weight: float = 1.0
    palm_clearance: float = 8.0e-3


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


def _fps(points: np.ndarray, n: int) -> np.ndarray:
    if n <= 0 or len(points) == 0:
        return np.zeros((0, 3), dtype=float)
    if len(points) <= n:
        return points

    keep = np.empty(n, dtype=int)
    center = np.mean(points, axis=0)
    keep[0] = int(np.argmin(np.sum((points - center) ** 2, axis=1)))
    dist2 = np.sum((points - points[keep[0]]) ** 2, axis=1)
    dist2[keep[0]] = -np.inf

    for i in range(1, n):
        keep[i] = int(np.argmax(dist2))
        dist2 = np.minimum(dist2, np.sum((points - points[keep[i]]) ** 2, axis=1))
        dist2[keep[: i + 1]] = -np.inf
    return points[keep]


def _strip(frame: SegmentFrame, n: int, blocked: list[tuple[float, float]]) -> np.ndarray:
    nx = max(8, 3 * n)
    na = max(8, 4 * n)
    xs = np.linspace(-frame.half, frame.half, nx)
    mask = np.ones(nx, dtype=bool)
    for lo, hi in blocked:
        mask &= (xs < lo) | (xs > hi)
    if np.any(mask):
        xs = xs[mask]
    if len(xs) == 0:
        xs = np.linspace(-frame.half, frame.half, max(2, nx // 2))

    step = (2.0 * HALF_ANGLE) / na
    angles = -HALF_ANGLE + (np.arange(na, dtype=float) + 0.5) * step
    points = []
    for i, x in enumerate(xs):
        angle = angles + 0.5 * step * (i % 2)
        normal = np.cos(angle)[:, None] * frame.face[None, :] + np.sin(angle)[:, None] * frame.tangent[None, :]
        points.append(frame.center[None, :] + x * frame.axis[None, :] + frame.radius * normal)
    return _fps(np.concatenate(points, axis=0), n)


def _role(finger: str, segment: str) -> str:
    if segment == "1":
        return "proximal"
    return "distal" if finger == "thumb" else "intermediate"


def _tip_mean(hand: Hand, fingers: tuple[str, ...]) -> np.ndarray:
    return np.mean(np.stack([hand.data.site_xpos[hand.tip_site_ids[finger]].copy() for finger in fingers], axis=0), axis=0)


def _joint_x(hand: Hand, body_id: int, center: np.ndarray, axis: np.ndarray) -> float:
    joint_count = int(hand.model.body_jntnum[body_id])
    if joint_count <= 0:
        return float(np.dot(hand.data.xpos[body_id] - center, axis))
    joint_id = int(hand.model.body_jntadr[body_id])
    return float(np.dot(hand.data.xanchor[joint_id] - center, axis))


def _blocked(hand: Hand, frame: SegmentFrame, segment: str, cfg: ContactConfig) -> list[tuple[float, float]]:
    xs = [_joint_x(hand, frame.body_id, frame.center, frame.axis)]
    for child_id in range(hand.model.nbody):
        if hand.model.body_parentid[child_id] == frame.body_id:
            xs.append(_joint_x(hand, child_id, frame.center, frame.axis))
    intervals = [(x - JOINT_CLEARANCE, x + JOINT_CLEARANCE) for x in xs]
    if segment == "1":
        intervals.append((-frame.half, xs[0] + cfg.palm_clearance))
    return _merge(intervals, -frame.half, frame.half)


def _frame(hand: Hand, finger: str, segment: str, palm_axis: np.ndarray, thumb_ref: np.ndarray) -> SegmentFrame:
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


def build_contacts(
    hand: Hand,
    *,
    qpos: np.ndarray | None = None,
    ctrl: np.ndarray | None = None,
    cfg: ContactConfig | None = None,
) -> list[ContactRecord]:
    cfg = ContactConfig() if cfg is None else cfg
    if cfg.n_per_seg <= 0:
        raise ValueError("cfg.n_per_seg must be positive")

    hand.apply_state(qpos=qpos, ctrl=ctrl)
    root_pos = hand.data.xpos[hand.root_body_id].copy()
    root_rot = hand.data.xmat[hand.root_body_id].reshape(3, 3)
    palm_axis = hand.data.site_xmat[hand.palm_site_id].reshape(3, 3)[:, 0].copy()
    thumb_ref = _tip_mean(hand, ("index", "middle", "ring", "pinky"))

    records: list[ContactRecord] = []
    for finger, segment in SEGS:
        frame = _frame(hand, finger, segment, palm_axis, thumb_ref)
        body_pos = hand.data.xpos[frame.body_id].copy()
        body_rot = hand.data.xmat[frame.body_id].reshape(3, 3)
        body_name = mujoco.mj_id2name(hand.model, mujoco.mjtObj.mjOBJ_BODY, frame.body_id) or f"body_{frame.body_id}"
        blocked = _blocked(hand, frame, segment, cfg)
        for point in _strip(frame, cfg.n_per_seg, blocked):
            world_normal = _unit(point - frame.center - frame.axis * np.dot(point - frame.center, frame.axis))
            world_pos = point + POINT_OFFSET * world_normal
            records.append(
                ContactRecord(
                    finger=finger,
                    segment=segment,
                    role=_role(finger, segment),
                    body_id=frame.body_id,
                    body_name=body_name,
                    local_pos=body_rot.T @ (world_pos - body_pos),
                    local_normal=body_rot.T @ world_normal,
                    world_pos=world_pos,
                    world_normal=world_normal,
                    root_pos=root_rot.T @ (world_pos - root_pos),
                    weight=cfg.thumb_weight if finger == "thumb" else 1.0,
                )
            )
    return records


def contact_points_root(
    hand: Hand,
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
