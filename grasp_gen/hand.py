from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
import numpy as np


if TYPE_CHECKING:
    from .hand_contacts import ContactConfig, ContactRecord


ROOT = Path(__file__).resolve().parent.parent
HAND_XML = {
    "right": ROOT / "assets" / "inspire" / "right.xml",
    "left": ROOT / "assets" / "inspire" / "left.xml",
}
TARGET = np.zeros(3, dtype=float)
ORTHO_EPS = 1.0e-8
CONTACT_CLOUD_SCALE = 1.00935
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SEGS = (
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


@dataclass(frozen=True)
class ActuatorSpec:
    idx: int
    name: str
    joint_name: str
    qpos_index: int
    finger: str
    role: str
    lo: float
    hi: float


@dataclass(frozen=True)
class JointConfig:
    flex_min: float = 0.1
    flex_max: float = 0.25
    thumb_pinch: float = float(np.pi / 2.0)
    thumb_zero: bool = True


@dataclass(frozen=True)
class InitConfig:
    n_per_seg: int = 10
    thumb_weight: float = 4.0
    palm_clearance: float = 8.0e-3
    contact_spacing: float = 5.0e-3
    contact_cloud_scale: float = CONTACT_CLOUD_SCALE
    palm_offset: float = 0.30
    roll_steps: int = 72
    std_weight: float = 0.75
    max_weight: float = 0.25
    pool_factor: int = 32
    pool_min: int = 1024


@dataclass(frozen=True)
class Pose:
    root_pos: np.ndarray
    root_quat: np.ndarray
    qpos: np.ndarray
    ctrl: np.ndarray
    palm_pos: np.ndarray
    palm_normal: np.ndarray
    reach_dir: np.ndarray
    roll_deg: float
    score: float


@dataclass(frozen=True)
class PoseBatch:
    root_pos: np.ndarray
    root_quat: np.ndarray
    root_ortho6d: np.ndarray
    qpos: np.ndarray
    ctrl: np.ndarray
    palm_pos: np.ndarray
    palm_normal: np.ndarray
    reach_dir: np.ndarray
    roll_deg: np.ndarray
    score: np.ndarray

    def __len__(self) -> int:
        return int(self.root_pos.shape[0])

    def __getitem__(self, index: int) -> Pose:
        return self.pose(index)

    def pose(self, index: int) -> Pose:
        idx = int(index)
        return Pose(
            root_pos=self.root_pos[idx].copy(),
            root_quat=self.root_quat[idx].copy(),
            qpos=self.qpos[idx].copy(),
            ctrl=self.ctrl[idx].copy(),
            palm_pos=self.palm_pos[idx].copy(),
            palm_normal=self.palm_normal[idx].copy(),
            reach_dir=self.reach_dir[idx].copy(),
            roll_deg=float(self.roll_deg[idx]),
            score=float(self.score[idx]),
        )

    def state_vectors(self) -> np.ndarray:
        return np.concatenate([self.root_pos, self.root_ortho6d, self.qpos], axis=1)


class _Kinematics(NamedTuple):
    parent_indices: np.ndarray
    body_local_pos: jax.Array
    body_local_rot: jax.Array
    joint_qpos_indices: np.ndarray
    joint_axes: jax.Array
    palm_body_index: int
    palm_site_pos: jax.Array
    palm_site_rot: jax.Array


class _ContactBatch(NamedTuple):
    body_indices: jax.Array
    local_positions: jax.Array
    weights: jax.Array
    dense_body_indices: jax.Array
    dense_local_positions: jax.Array


class _JointBatch(NamedTuple):
    qpos_indices: np.ndarray
    ctrl_lo: jax.Array
    ctrl_hi: jax.Array
    thumb_yaw: jax.Array
    thumb_lock: jax.Array


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


def _quat_to_matrix_np(quat: np.ndarray) -> np.ndarray:
    matrix = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=float))
    return matrix.reshape(3, 3)


def _matrix_to_quat_np(rotation: np.ndarray) -> np.ndarray:
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
    quat = _unit_quat(quat)
    return np.asarray(quat, dtype=np.float32)


def _matrix_to_rpy_deg_np(rotation: np.ndarray) -> np.ndarray:
    sy = float(np.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0]))
    singular = sy < 1.0e-8
    if not singular:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.rad2deg(np.array([roll, pitch, yaw], dtype=float))


def _build_actuators(model: mujoco.MjModel) -> list[ActuatorSpec]:
    actuators: list[ActuatorSpec] = []
    for actuator_index in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_index) or f"act_{actuator_index}"
        joint_id = int(model.actuator_trnid[actuator_index, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
        qpos_index = int(model.jnt_qposadr[joint_id])
        parts = name.split("_")
        prefix = 1 if len(parts) > 1 and parts[0] == "inspire" else 0
        finger = parts[prefix + 1] if len(parts) - prefix >= 3 else "unknown"
        role = "_".join(parts[prefix + 2 : -1]) or name
        actuators.append(
            ActuatorSpec(
                idx=actuator_index,
                name=name,
                joint_name=joint_name,
                qpos_index=qpos_index,
                finger=finger,
                role=role,
                lo=float(model.actuator_ctrlrange[actuator_index, 0]),
                hi=float(model.actuator_ctrlrange[actuator_index, 1]),
            )
        )
    return actuators


def _make_spec(
    side: str,
    root_pos: np.ndarray | None = None,
    root_quat: np.ndarray | None = None,
) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(HAND_XML[side]))

    for name in (
        f"{side}_pos_x_position",
        f"{side}_pos_y_position",
        f"{side}_pos_z_position",
        f"{side}_rot_x_position",
        f"{side}_rot_y_position",
        f"{side}_rot_z_position",
    ):
        actuator = spec.actuator(name)
        if actuator is not None:
            spec.delete(actuator)

    for name in (
        f"{side}_pos_x",
        f"{side}_pos_y",
        f"{side}_pos_z",
        f"{side}_rot_x",
        f"{side}_rot_y",
        f"{side}_rot_z",
    ):
        joint = spec.joint(name)
        if joint is not None:
            spec.delete(joint)

    root = spec.body(f"{side}_hand_base")
    if root is None:
        raise ValueError(f"Hand root body '{side}_hand_base' was not found.")

    root.pos = np.zeros(3, dtype=float) if root_pos is None else np.asarray(root_pos, dtype=float)
    root.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float) if root_quat is None else _unit_quat(root_quat)
    return spec


def _safe_normalize(vector: jax.Array, fallback: jax.Array) -> jax.Array:
    vector = jnp.asarray(vector, dtype=jnp.float32)
    fallback = jnp.broadcast_to(jnp.asarray(fallback, dtype=vector.dtype), vector.shape)
    norm = jnp.linalg.norm(vector, axis=-1, keepdims=True)
    return jnp.where(norm > ORTHO_EPS, vector / jnp.maximum(norm, ORTHO_EPS), fallback)


def _axis_angle_rotation(axis: jax.Array, angle: jax.Array | float) -> jax.Array:
    axis = _safe_normalize(axis, jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float32))
    angle = jnp.asarray(angle, dtype=jnp.float32)
    x, y, z = axis
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    d = 1.0 - c
    return jnp.stack(
        [
            jnp.stack([c + x * x * d, x * y * d - z * s, x * z * d + y * s], axis=-1),
            jnp.stack([y * x * d + z * s, c + y * y * d, y * z * d - x * s], axis=-1),
            jnp.stack([z * x * d - y * s, z * y * d + x * s, c + z * z * d], axis=-1),
        ],
        axis=-2,
    )


def _align_vectors(src: jax.Array, dst: jax.Array) -> jax.Array:
    identity = jnp.eye(3, dtype=jnp.float32)
    src = _safe_normalize(src, jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float32))
    dst = _safe_normalize(dst, jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float32))
    dot = jnp.clip(jnp.dot(src, dst), -1.0, 1.0)
    helper = jnp.where(
        jnp.abs(src[0]) < 0.9,
        jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32),
    )
    opposite_axis = _safe_normalize(jnp.cross(src, helper), jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float32))
    general_axis = _safe_normalize(jnp.cross(src, dst), opposite_axis)
    general_rot = _axis_angle_rotation(general_axis, jnp.arccos(dot))
    opposite_rot = _axis_angle_rotation(opposite_axis, np.pi)
    return jnp.where(dot > 1.0 - ORTHO_EPS, identity, jnp.where(dot < -1.0 + ORTHO_EPS, opposite_rot, general_rot))


def _rad_to_ctrl(rad: jax.Array, lo: jax.Array, hi: jax.Array) -> jax.Array:
    mag = jnp.abs(rad)
    raw = jnp.where(
        (rad < 0.0) & (lo < 0.0),
        -mag,
        jnp.where(
            (rad > 0.0) & (hi > 0.0),
            mag,
            jnp.where(hi <= 0.0, -mag, jnp.where(lo >= 0.0, mag, jnp.zeros_like(mag))),
        ),
    )
    return jnp.clip(raw, lo, hi)


def _hand_body_ids(hand: Hand) -> list[int]:
    body_ids: list[int] = []
    for body_id in range(1, hand.model.nbody):
        current_id = body_id
        while current_id > 0 and current_id != hand.root_body_id:
            current_id = int(hand.model.body_parentid[current_id])
        if current_id == hand.root_body_id:
            body_ids.append(body_id)
    if hand.root_body_id not in body_ids:
        body_ids.insert(0, hand.root_body_id)
    return body_ids


def _prepare_kinematics(hand: Hand) -> tuple[_Kinematics, dict[int, int]]:
    body_ids = _hand_body_ids(hand)
    id_map = {old_id: new_id for new_id, old_id in enumerate(body_ids)}

    parent_indices: list[int] = []
    body_local_pos: list[np.ndarray] = []
    body_local_rot: list[np.ndarray] = []
    joint_qpos_indices: list[int] = []
    joint_axes: list[np.ndarray] = []

    for new_body_id, old_body_id in enumerate(body_ids):
        if new_body_id == 0:
            parent_indices.append(-1)
            body_local_pos.append(np.zeros(3, dtype=np.float32))
            body_local_rot.append(np.eye(3, dtype=np.float32))
            joint_qpos_indices.append(-1)
            joint_axes.append(np.zeros(3, dtype=np.float32))
            continue

        parent_id = int(hand.model.body_parentid[old_body_id])
        if parent_id not in id_map:
            raise ValueError("Encountered a hand body whose parent is outside the compact hand tree.")
        parent_indices.append(id_map[parent_id])
        body_local_pos.append(np.asarray(hand.model.body_pos[old_body_id], dtype=np.float32))
        body_local_rot.append(_quat_to_matrix_np(hand.model.body_quat[old_body_id]).astype(np.float32))

        joint_count = int(hand.model.body_jntnum[old_body_id])
        if joint_count == 0:
            joint_qpos_indices.append(-1)
            joint_axes.append(np.zeros(3, dtype=np.float32))
            continue
        if joint_count != 1:
            raise NotImplementedError("Hand batch kinematics supports at most one joint per body.")

        joint_id = int(hand.model.body_jntadr[old_body_id])
        if np.linalg.norm(hand.model.jnt_pos[joint_id]) > 1.0e-8:
            raise NotImplementedError("Hand batch kinematics requires zero joint offsets.")
        joint_qpos_indices.append(int(hand.model.jnt_qposadr[joint_id]))
        joint_axes.append(np.asarray(hand.model.jnt_axis[joint_id], dtype=np.float32))

    palm_body_id = int(hand.model.site_bodyid[hand.palm_site_id])
    if palm_body_id not in id_map:
        raise ValueError("Palm site body is outside the compact hand tree.")

    return (
        _Kinematics(
            parent_indices=np.asarray(parent_indices, dtype=np.int32),
            body_local_pos=jnp.asarray(np.stack(body_local_pos, axis=0), dtype=jnp.float32),
            body_local_rot=jnp.asarray(np.stack(body_local_rot, axis=0), dtype=jnp.float32),
            joint_qpos_indices=np.asarray(joint_qpos_indices, dtype=np.int32),
            joint_axes=jnp.asarray(np.stack(joint_axes, axis=0), dtype=jnp.float32),
            palm_body_index=id_map[palm_body_id],
            palm_site_pos=jnp.asarray(hand.model.site_pos[hand.palm_site_id], dtype=jnp.float32),
            palm_site_rot=jnp.asarray(_quat_to_matrix_np(hand.model.site_quat[hand.palm_site_id]), dtype=jnp.float32),
        ),
        id_map,
    )


def _prepare_joint_batch(hand: Hand) -> _JointBatch:
    qpos_indices: list[int] = []
    thumb_yaw: list[bool] = []
    thumb_lock: list[bool] = []
    ctrl_lo: list[float] = []
    ctrl_hi: list[float] = []

    for actuator in hand.actuators:
        joint_id = int(hand.model.actuator_trnid[actuator.idx, 0])
        qpos_indices.append(int(hand.model.jnt_qposadr[joint_id]))
        thumb_yaw.append(bool(actuator.finger == "thumb" and "yaw" in actuator.role))
        thumb_lock.append(bool(actuator.finger == "thumb" and "yaw" not in actuator.role))
        ctrl_lo.append(float(actuator.lo))
        ctrl_hi.append(float(actuator.hi))

    return _JointBatch(
        qpos_indices=np.asarray(qpos_indices, dtype=np.int32),
        ctrl_lo=jnp.asarray(ctrl_lo, dtype=jnp.float32),
        ctrl_hi=jnp.asarray(ctrl_hi, dtype=jnp.float32),
        thumb_yaw=jnp.asarray(thumb_yaw, dtype=bool),
        thumb_lock=jnp.asarray(thumb_lock, dtype=bool),
    )


def _sample_joint_targets(
    joints: _JointBatch,
    rng_key: jax.Array,
    *,
    batch_size: int,
    cfg: JointConfig,
) -> tuple[jax.Array, jax.Array]:
    base = jax.random.uniform(
        rng_key,
        shape=(batch_size, joints.ctrl_lo.shape[0]),
        minval=jnp.asarray(cfg.flex_min, dtype=jnp.float32),
        maxval=jnp.asarray(cfg.flex_max, dtype=jnp.float32),
        dtype=jnp.float32,
    )
    base = jnp.where(joints.ctrl_hi[None, :] <= 0.0, -base, base)
    rad = jnp.where(joints.thumb_yaw[None, :], jnp.asarray(cfg.thumb_pinch, dtype=jnp.float32), base)
    rad = jnp.where(jnp.asarray(cfg.thumb_zero, dtype=bool) & joints.thumb_lock[None, :], 0.0, rad)

    ctrl = _rad_to_ctrl(rad, joints.ctrl_lo[None, :], joints.ctrl_hi[None, :])
    qpos = jnp.zeros((batch_size, joints.ctrl_lo.shape[0]), dtype=jnp.float32)
    qpos = qpos.at[:, joints.qpos_indices].set(ctrl)
    return qpos, ctrl


def _forward_kinematics(kin: _Kinematics, hand_qpos: jax.Array) -> tuple[jax.Array, jax.Array]:
    batch_size = hand_qpos.shape[0]
    identity = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (batch_size, 3, 3))
    body_positions = [jnp.zeros((batch_size, 3), dtype=jnp.float32)]
    body_rotations = [identity]

    for body_index in range(1, len(kin.parent_indices)):
        parent_index = int(kin.parent_indices[body_index])
        parent_pos = body_positions[parent_index]
        parent_rot = body_rotations[parent_index]
        local_pos = kin.body_local_pos[body_index]
        local_rot = kin.body_local_rot[body_index]
        qpos_index = int(kin.joint_qpos_indices[body_index])

        if qpos_index >= 0:
            joint_rot = _axis_angle_rotation(kin.joint_axes[body_index], hand_qpos[:, qpos_index])
            body_rot = jnp.einsum("bij,jk,bkl->bil", parent_rot, local_rot, joint_rot)
        else:
            body_rot = jnp.einsum("bij,jk->bik", parent_rot, local_rot)
        body_pos = parent_pos + jnp.einsum("bij,j->bi", parent_rot, local_pos)
        body_positions.append(body_pos)
        body_rotations.append(body_rot)

    return jnp.stack(body_positions, axis=1), jnp.stack(body_rotations, axis=1)


def _world_points(
    body_positions: jax.Array,
    body_rotations: jax.Array,
    body_indices: jax.Array,
    local_positions: jax.Array,
) -> jax.Array:
    world_pos = body_positions[:, body_indices, :]
    world_rot = body_rotations[:, body_indices, :, :]
    return world_pos + jnp.einsum("bpij,pj->bpi", world_rot, local_positions)


def _site_world(
    body_positions: jax.Array,
    body_rotations: jax.Array,
    body_index: int,
    local_pos: jax.Array,
    local_rot: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    body_pos = body_positions[:, body_index, :]
    body_rot = body_rotations[:, body_index, :, :]
    site_pos = body_pos + jnp.einsum("bij,j->bi", body_rot, local_pos)
    site_rot = jnp.einsum("bij,jk->bik", body_rot, local_rot)
    return site_pos, site_rot


def _sample_dirs(rng_key: jax.Array, *, pool_size: int) -> jax.Array:
    dirs = jax.random.normal(rng_key, shape=(pool_size, 3), dtype=jnp.float32)
    return _safe_normalize(dirs, jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32))


def _farthest_indices(points: jax.Array, *, count: int, start_index: jax.Array) -> jax.Array:
    chosen = jnp.zeros((count,), dtype=jnp.int32)
    chosen = chosen.at[0].set(start_index)
    dist2 = jnp.sum((points - points[start_index]) ** 2, axis=1)
    dist2 = dist2.at[start_index].set(-jnp.inf)

    def step_body(step: int, carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        current, current_dist2 = carry
        next_index = jnp.argmax(current_dist2)
        current = current.at[step].set(next_index)
        next_dist2 = jnp.sum((points - points[next_index]) ** 2, axis=1)
        current_dist2 = jnp.minimum(current_dist2, next_dist2)
        current_dist2 = current_dist2.at[next_index].set(-jnp.inf)
        return current, current_dist2

    chosen, _ = jax.lax.fori_loop(1, count, step_body, (chosen, dist2))
    return chosen


def _run_batch(
    kin: _Kinematics,
    contacts: _ContactBatch,
    joints: _JointBatch,
    rng_key: jax.Array,
    *,
    batch_size: int,
    pool_size: int,
    cfg: InitConfig,
    joint_cfg: JointConfig,
) -> tuple[jax.Array, ...]:
    @partial(jax.jit, static_argnames=("batch_size", "pool_size", "roll_steps"))
    def run(
        rng_key: jax.Array,
        *,
        batch_size: int,
        pool_size: int,
        roll_steps: int,
    ) -> tuple[jax.Array, ...]:
        joint_key, dir_key, start_key = jax.random.split(rng_key, 3)
        hand_qpos, hand_ctrl = _sample_joint_targets(joints, joint_key, batch_size=batch_size, cfg=joint_cfg)

        body_positions, body_rotations = _forward_kinematics(kin, hand_qpos)
        contact_points = _world_points(body_positions, body_rotations, contacts.body_indices, contacts.local_positions)
        root_to_palm_pos, root_to_palm_rot = _site_world(
            body_positions,
            body_rotations,
            kin.palm_body_index,
            kin.palm_site_pos,
            kin.palm_site_rot,
        )

        offsets = contact_points - root_to_palm_pos[:, None, :]
        weights = contacts.weights / jnp.sum(contacts.weights)
        local_axis = _safe_normalize(
            jnp.sum(offsets * weights[None, :, None], axis=1),
            jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32),
        )

        dirs = _sample_dirs(dir_key, pool_size=pool_size)
        start_index = jax.random.randint(start_key, shape=(), minval=0, maxval=pool_size)
        chosen = _farthest_indices(dirs, count=batch_size, start_index=start_index)
        reach_dir = dirs[chosen]
        palm_pos = -jnp.asarray(cfg.palm_offset, dtype=jnp.float32) * reach_dir

        base_rot = jax.vmap(_align_vectors)(local_axis, reach_dir)
        angle_step = (2.0 * np.pi) / max(roll_steps, 1)
        angles = jnp.arange(roll_steps, dtype=jnp.float32) * jnp.asarray(angle_step, dtype=jnp.float32)
        roll_rot = jax.vmap(lambda axis: jax.vmap(lambda angle: _axis_angle_rotation(axis, angle))(angles))(reach_dir)
        rotations = jnp.einsum("brij,bjk->brik", roll_rot, base_rot)

        anchor_from_palm = -palm_pos
        world_offsets = jnp.einsum("brij,bpj->brpi", rotations, offsets)
        distances = jnp.linalg.norm(anchor_from_palm[:, None, None, :] - world_offsets, axis=-1)
        mean_distance = jnp.sum(weights[None, None, :] * distances, axis=-1)
        variance = jnp.sum(weights[None, None, :] * jnp.square(distances - mean_distance[:, :, None]), axis=-1)
        scores = (
            mean_distance
            + jnp.asarray(cfg.std_weight, dtype=jnp.float32) * jnp.sqrt(jnp.maximum(variance, 0.0))
            + jnp.asarray(cfg.max_weight, dtype=jnp.float32) * jnp.max(distances, axis=-1)
        )

        best_roll = jnp.argmin(scores, axis=1)
        batch_index = jnp.arange(batch_size, dtype=jnp.int32)
        best_rot = rotations[batch_index, best_roll]
        best_score = scores[batch_index, best_roll]
        root_pos = palm_pos - jnp.einsum("bij,bj->bi", best_rot, root_to_palm_pos)
        palm_axes = jnp.einsum("bij,bjk->bik", best_rot, root_to_palm_rot)
        palm_normal = _safe_normalize(palm_axes[:, :, 0], reach_dir)
        root_ortho6d = jnp.concatenate([best_rot[:, :, 0], best_rot[:, :, 1]], axis=1)
        roll_deg = best_roll.astype(jnp.float32) * (360.0 / max(roll_steps, 1))
        return (
            root_pos,
            root_ortho6d,
            hand_qpos,
            hand_ctrl,
            palm_pos,
            palm_normal,
            reach_dir,
            roll_deg,
            best_score,
            best_rot,
        )

    return run(rng_key, batch_size=batch_size, pool_size=pool_size, roll_steps=max(int(cfg.roll_steps), 1))


class Hand:
    def __init__(self, side: str):
        if side not in HAND_XML:
            raise ValueError(f"Unsupported hand side: {side}")

        self.side = side
        self.target = TARGET.copy()
        spec = _make_spec(side)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self.root_body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand_base"))
        self.palm_site_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_palm"))
        self.tip_site_ids = {
            finger: int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_{finger}_tip"))
            for finger in FINGERS
        }
        self.segment_geom_ids = {
            key: int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"collision_hand_{side}_{key[0]}_{key[1]}"))
            for key in SEGS
        }
        self.root_home_pos = self.model.body_pos[self.root_body_id].copy()
        self.root_home_quat = self.model.body_quat[self.root_body_id].copy()
        self.actuators = _build_actuators(self.model)
        self._kin, self._body_id_map = _prepare_kinematics(self)
        self._joints = _prepare_joint_batch(self)
        self._contact_cache: dict[tuple[int, float, float], _ContactBatch] = {}

        self.apply_state(qpos=self.model.qpos0.copy(), ctrl=np.zeros(self.model.nu, dtype=float))

    def mj(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        return self.model, self.data

    def mjcf(
        self,
        root_pos: np.ndarray | None = None,
        root_quat: np.ndarray | None = None,
    ) -> mujoco.MjSpec:
        return _make_spec(self.side, root_pos=root_pos, root_quat=root_quat)

    def apply_state(
        self,
        qpos: np.ndarray | None = None,
        ctrl: np.ndarray | None = None,
        root_pos: np.ndarray | None = None,
        root_quat: np.ndarray | None = None,
    ) -> None:
        qpos_array = self.data.qpos.copy() if qpos is None else np.asarray(qpos, dtype=float).reshape(-1)
        if qpos_array.shape != (self.model.nq,):
            raise ValueError(f"qpos must have shape ({self.model.nq},), got {qpos_array.shape}")

        ctrl_array = self.data.ctrl.copy() if ctrl is None else np.asarray(ctrl, dtype=float).reshape(-1)
        if ctrl_array.shape != (self.model.nu,):
            raise ValueError(f"ctrl must have shape ({self.model.nu},), got {ctrl_array.shape}")

        self.model.body_pos[self.root_body_id] = (
            self.model.body_pos[self.root_body_id].copy()
            if root_pos is None
            else np.asarray(root_pos, dtype=float)
        )
        self.model.body_quat[self.root_body_id] = (
            self.model.body_quat[self.root_body_id].copy()
            if root_quat is None
            else _unit_quat(root_quat)
        )
        self.data.qpos[:] = qpos_array
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = np.clip(ctrl_array, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
        mujoco.mj_forward(self.model, self.data)

    def apply(self, pose: Pose) -> None:
        self.apply_state(qpos=pose.qpos, ctrl=pose.ctrl, root_pos=pose.root_pos, root_quat=pose.root_quat)

    def apply_batch(self, batch: PoseBatch, index: int) -> Pose:
        pose = batch.pose(index)
        self.apply(pose)
        return pose

    def root_6dof(self, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
        pos = np.asarray(pose.root_pos, dtype=float).copy()
        rpy_deg = _matrix_to_rpy_deg_np(_quat_to_matrix_np(pose.root_quat))
        return pos, rpy_deg

    def joint_values(self, pose: Pose) -> list[tuple[str, float, float]]:
        qpos = np.asarray(pose.qpos, dtype=float)
        return [
            (
                actuator.joint_name,
                float(qpos[actuator.qpos_index]),
                float(np.rad2deg(qpos[actuator.qpos_index])),
            )
            for actuator in self.actuators
        ]

    def reset_root(self) -> None:
        self.model.body_pos[self.root_body_id] = self.root_home_pos.copy()
        self.model.body_quat[self.root_body_id] = self.root_home_quat.copy()
        mujoco.mj_forward(self.model, self.data)

    def body_local_to_world(self, body_ids: np.ndarray, local_pos: np.ndarray) -> np.ndarray:
        body_ids = np.asarray(body_ids, dtype=np.intp)
        local_pos = np.asarray(local_pos, dtype=float)
        if local_pos.size == 0:
            return np.zeros((0, 3), dtype=float)
        body_pos = self.data.xpos[body_ids]
        body_rot = self.data.xmat[body_ids].reshape(-1, 3, 3)
        return body_pos + np.einsum("nij,nj->ni", body_rot, local_pos)

    def _contact_batch(self, cfg: InitConfig) -> _ContactBatch:
        key = (
            int(cfg.n_per_seg),
            float(cfg.thumb_weight),
            float(cfg.palm_clearance),
            float(cfg.contact_spacing),
            float(cfg.contact_cloud_scale),
        )
        cached = self._contact_cache.get(key)
        if cached is not None:
            return cached

        from .hand_contacts import ContactConfig, build_surface_cloud, sample_contacts

        self.apply_state(
            qpos=self.model.qpos0.copy(),
            ctrl=np.zeros(self.model.nu, dtype=float),
            root_pos=self.root_home_pos,
            root_quat=self.root_home_quat,
        )
        contact_cfg = ContactConfig(
            n_per_seg=cfg.n_per_seg,
            thumb_weight=cfg.thumb_weight,
            palm_clearance=cfg.palm_clearance,
            target_spacing=cfg.contact_spacing,
            cloud_scale=cfg.contact_cloud_scale,
        )
        surface_records = build_surface_cloud(
            self,
            qpos=self.model.qpos0.copy(),
            ctrl=np.zeros(self.model.nu, dtype=float),
            cfg=contact_cfg,
        )
        records = sample_contacts(self, surface_records, cfg=contact_cfg)
        if not surface_records:
            raise ValueError("No hand surface cloud points were generated for the batch initializer.")
        if not records:
            raise ValueError("No hand contact records were generated for the batch initializer.")

        cached = _ContactBatch(
            body_indices=jnp.asarray([self._body_id_map[int(record.body_id)] for record in records], dtype=jnp.int32),
            local_positions=jnp.asarray([record.local_pos for record in records], dtype=jnp.float32),
            weights=jnp.asarray([record.weight for record in records], dtype=jnp.float32),
            dense_body_indices=jnp.asarray([self._body_id_map[int(record.body_id)] for record in surface_records], dtype=jnp.int32),
            dense_local_positions=jnp.asarray([record.local_pos for record in surface_records], dtype=jnp.float32),
        )
        self._contact_cache[key] = cached
        return cached

    def init_batch(
        self,
        n: int = 64,
        *,
        cfg: InitConfig | None = None,
        joint: JointConfig | None = None,
        seed: int = 0,
    ) -> PoseBatch:
        cfg = InitConfig() if cfg is None else cfg
        joint = JointConfig() if joint is None else joint
        if n <= 0:
            raise ValueError("n must be positive.")
        if cfg.palm_offset <= 0.0:
            raise ValueError("cfg.palm_offset must be positive.")
        if cfg.pool_factor <= 0 or cfg.pool_min <= 0:
            raise ValueError("cfg.pool_factor and cfg.pool_min must be positive.")

        pool_size = max(int(cfg.pool_factor) * int(n), int(cfg.pool_min), int(n))
        rng_key = jax.random.key(np.uint32(int(seed) % (2**32)))
        (
            root_pos,
            root_ortho6d,
            hand_qpos,
            hand_ctrl,
            palm_pos,
            palm_normal,
            reach_dir,
            roll_deg,
            best_score,
            best_rot,
        ) = _run_batch(
            self._kin,
            self._contact_batch(cfg),
            self._joints,
            rng_key,
            batch_size=int(n),
            pool_size=pool_size,
            cfg=cfg,
            joint_cfg=joint,
        )

        root_rot = np.asarray(best_rot, dtype=np.float32)
        root_quat = np.asarray([_matrix_to_quat_np(rotation) for rotation in root_rot], dtype=np.float32)
        return PoseBatch(
            root_pos=np.asarray(root_pos, dtype=np.float32),
            root_quat=root_quat,
            root_ortho6d=np.asarray(root_ortho6d, dtype=np.float32),
            qpos=np.asarray(hand_qpos, dtype=np.float32),
            ctrl=np.asarray(hand_ctrl, dtype=np.float32),
            palm_pos=np.asarray(palm_pos, dtype=np.float32),
            palm_normal=np.asarray(palm_normal, dtype=np.float32),
            reach_dir=np.asarray(reach_dir, dtype=np.float32),
            roll_deg=np.asarray(roll_deg, dtype=np.float32),
            score=np.asarray(best_score, dtype=np.float32),
        )

    def contacts(self, pose: Pose | None = None, cfg: ContactConfig | None = None) -> list[ContactRecord]:
        from .hand_contacts import build_contacts

        if pose is not None:
            self.apply(pose)
        return build_contacts(self, cfg=cfg)

    def contact_points(
        self,
        pose: Pose | None = None,
        cfg: ContactConfig | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        from .hand_contacts import contact_points_root

        if pose is not None:
            self.apply(pose)
        return contact_points_root(self, cfg=cfg)
