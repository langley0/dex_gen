from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hand import Hand
from .hand_contacts import ContactConfig, contact_points_root


@dataclass(frozen=True)
class JointInitConfig:
    flex_min: float = 0.1
    flex_max: float = 0.25
    thumb_pinch: float = float(np.pi / 2.0)
    thumb_zero: bool = True


@dataclass(frozen=True)
class InitPoseConfig:
    contact_n_per_seg: int = 10
    thumb_weight: float = 4.0
    palm_offset: float = 0.30
    reach: tuple[float, float, float] = (0.0, 1.0, 0.0)
    roll_steps: int = 72


@dataclass(frozen=True)
class HandPose:
    root_pos: np.ndarray
    root_quat: np.ndarray
    qpos: np.ndarray
    ctrl: np.ndarray
    palm_pos: np.ndarray
    palm_normal: np.ndarray
    anchor: np.ndarray
    score: float


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    return vector.copy() if norm < 1.0e-8 else vector / norm


def _unit_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float).copy()
    norm = np.linalg.norm(quat)
    if norm < 1.0e-8:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        quat /= norm
    if quat[0] < 0.0:
        quat *= -1.0
    return quat


def _quat_from_rot(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s,
            ],
            dtype=float,
        )
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        quat = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / s,
                0.25 * s,
                (rotation[0, 1] + rotation[1, 0]) / s,
                (rotation[0, 2] + rotation[2, 0]) / s,
            ],
            dtype=float,
        )
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        quat = np.array(
            [
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[0, 1] + rotation[1, 0]) / s,
                0.25 * s,
                (rotation[1, 2] + rotation[2, 1]) / s,
            ],
            dtype=float,
        )
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        quat = np.array(
            [
                (rotation[1, 0] - rotation[0, 1]) / s,
                (rotation[0, 2] + rotation[2, 0]) / s,
                (rotation[1, 2] + rotation[2, 1]) / s,
                0.25 * s,
            ],
            dtype=float,
        )
    return _unit_quat(quat)


def _rad_ctrl(rad: float, lo: float, hi: float) -> float:
    mag = float(abs(rad))
    if rad < 0.0 and lo < 0.0:
        raw = -mag
    elif rad > 0.0 and hi > 0.0:
        raw = mag
    elif hi <= 0.0:
        raw = -mag
    elif lo >= 0.0:
        raw = mag
    else:
        raw = 0.0
    return float(np.clip(raw, lo, hi))


def _axis_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _unit(axis)
    if np.linalg.norm(axis) < 1.0e-8 or abs(angle) < 1.0e-12:
        return np.eye(3, dtype=float)
    x, y, z = axis
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    d = 1.0 - c
    return np.array(
        [
            [c + x * x * d, x * y * d - z * s, x * z * d + y * s],
            [y * x * d + z * s, c + y * y * d, y * z * d - x * s],
            [z * x * d - y * s, z * y * d + x * s, c + z * z * d],
        ],
        dtype=float,
    )


def _align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = _unit(src)
    dst = _unit(dst)
    if np.linalg.norm(src) < 1.0e-8 or np.linalg.norm(dst) < 1.0e-8:
        return np.eye(3, dtype=float)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    if dot > 1.0 - 1.0e-8:
        return np.eye(3, dtype=float)
    if dot < -1.0 + 1.0e-8:
        aux = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(src[0]) > 0.9:
            aux = np.array([0.0, 1.0, 0.0], dtype=float)
        return _axis_rot(_unit(np.cross(src, aux)), np.pi)
    return _axis_rot(_unit(np.cross(src, dst)), float(np.arccos(dot)))


def _score(offsets: np.ndarray, weight: np.ndarray, anchor_from_palm: np.ndarray, rotation: np.ndarray) -> float:
    world_points = (rotation @ offsets.T).T
    distances = np.linalg.norm(anchor_from_palm[None, :] - world_points, axis=1)
    weight = np.asarray(weight, dtype=float)
    weight = weight / np.sum(weight)
    mean = float(np.sum(weight * distances))
    var = float(np.sum(weight * (distances - mean) ** 2))
    return float(mean + 0.75 * np.sqrt(max(var, 0.0)) + 0.25 * np.max(distances))


def sample_joint_targets(
    hand: Hand,
    *,
    cfg: JointInitConfig | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    cfg = JointInitConfig() if cfg is None else cfg
    rng = np.random.default_rng(seed)
    ctrl = np.zeros(hand.model.nu, dtype=float)
    for actuator in hand.actuators:
        if actuator.finger == "thumb" and "yaw" in actuator.role:
            rad = cfg.thumb_pinch
        elif actuator.finger == "thumb" and cfg.thumb_zero:
            rad = 0.0
        else:
            rad = float(rng.uniform(cfg.flex_min, cfg.flex_max))
            if actuator.hi <= 0.0:
                rad *= -1.0
        ctrl[actuator.idx] = _rad_ctrl(rad, actuator.lo, actuator.hi)
    return ctrl.copy(), ctrl


def make_home_pose(hand: Hand) -> HandPose:
    qpos = hand.model.qpos0.copy()
    ctrl = np.zeros(hand.model.nu, dtype=float)
    hand.apply_state(qpos=qpos, ctrl=ctrl, root_pos=hand.root_home_pos, root_quat=hand.root_home_quat)
    return HandPose(
        root_pos=hand.root_home_pos.copy(),
        root_quat=hand.root_home_quat.copy(),
        qpos=qpos,
        ctrl=ctrl,
        palm_pos=hand.data.site_xpos[hand.palm_site_id].copy(),
        palm_normal=hand.data.site_xmat[hand.palm_site_id].reshape(3, 3)[:, 0].copy(),
        anchor=np.zeros(3, dtype=float),
        score=0.0,
    )


def make_init_pose(
    hand: Hand,
    *,
    anchor: np.ndarray | None = None,
    cfg: InitPoseConfig | None = None,
    joint_cfg: JointInitConfig | None = None,
    qpos: np.ndarray | None = None,
    ctrl: np.ndarray | None = None,
    seed: int = 0,
) -> HandPose:
    cfg = InitPoseConfig() if cfg is None else cfg
    anchor = np.zeros(3, dtype=float) if anchor is None else np.asarray(anchor, dtype=float)
    reach = _unit(np.asarray(cfg.reach, dtype=float))
    if np.linalg.norm(reach) < 1.0e-8:
        raise ValueError("cfg.reach must be non-zero")

    if qpos is None or ctrl is None:
        q0, u0 = sample_joint_targets(hand, cfg=joint_cfg, seed=seed)
        qpos = q0 if qpos is None else qpos
        ctrl = u0 if ctrl is None else ctrl

    hand.apply_state(qpos=qpos, ctrl=ctrl, root_pos=hand.root_home_pos, root_quat=hand.root_home_quat)
    contact_cfg = ContactConfig(n_per_seg=cfg.contact_n_per_seg, thumb_weight=cfg.thumb_weight)
    points, weight = contact_points_root(hand, qpos=qpos, ctrl=ctrl, cfg=contact_cfg)
    root_to_palm_pos, root_to_palm_rot = hand.root_to_palm()
    offsets = points - root_to_palm_pos[None, :]
    local_axis = _unit(np.sum(offsets * weight[:, None], axis=0))

    palm_pos = anchor - cfg.palm_offset * reach
    anchor_from_palm = anchor - palm_pos
    base_rot = _align(local_axis, reach)
    best_rot = base_rot
    best_score = _score(offsets, weight, anchor_from_palm, base_rot)

    steps = max(int(cfg.roll_steps), 1)
    for i in range(steps):
        angle = (2.0 * np.pi * i) / steps
        rot = _axis_rot(reach, angle) @ base_rot
        score = _score(offsets, weight, anchor_from_palm, rot)
        if score < best_score:
            best_rot = rot
            best_score = score

    root_pos = palm_pos - best_rot @ root_to_palm_pos
    root_quat = _quat_from_rot(best_rot)
    palm_normal = _unit((best_rot @ root_to_palm_rot)[:, 0])
    return HandPose(
        root_pos=root_pos,
        root_quat=root_quat,
        qpos=np.asarray(qpos, dtype=float).copy(),
        ctrl=np.asarray(ctrl, dtype=float).copy(),
        palm_pos=palm_pos,
        palm_normal=palm_normal,
        anchor=anchor.copy(),
        score=float(best_score),
    )


def apply_pose(hand: Hand, pose: HandPose) -> None:
    hand.apply_state(qpos=pose.qpos, ctrl=pose.ctrl, root_pos=pose.root_pos, root_quat=pose.root_quat)
