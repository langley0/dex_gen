from __future__ import annotations

from typing import NamedTuple

import mujoco
import numpy as np

from .hand import Hand
from .prop import Prop


class ContactEval(NamedTuple):
    contact_count: int
    penetration_count: int
    depth_sum: float
    max_depth: float
    energy: float


class ContactBatchEval(NamedTuple):
    contact_count: np.ndarray
    penetration_count: np.ndarray
    depth_sum: np.ndarray
    max_depth: np.ndarray
    energy: np.ndarray


def _unit_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float).reshape(4).copy()
    norm = np.linalg.norm(quat)
    if norm < 1.0e-8:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        quat /= norm
    if quat[0] < 0.0:
        quat *= -1.0
    return quat


def _ortho6d_to_matrix_np(ortho6d: np.ndarray) -> np.ndarray:
    ortho6d = np.asarray(ortho6d, dtype=float).reshape(6)
    first = ortho6d[:3]
    first /= max(float(np.linalg.norm(first)), 1.0e-8)
    second = ortho6d[3:6] - first * float(np.dot(first, ortho6d[3:6]))
    second /= max(float(np.linalg.norm(second)), 1.0e-8)
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=1)


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
    return _unit_quat(quat)


def _pose_root_quat(hand_pose: np.ndarray) -> np.ndarray:
    rotation = _ortho6d_to_matrix_np(np.asarray(hand_pose[3:9], dtype=float))
    return _matrix_to_quat_np(rotation)


class MuJoCoContactOracle:
    def __init__(self, hand: Hand, prop: Prop, *, weight: float = 100.0):
        self.hand = hand
        self.prop = prop
        self.weight = float(weight)

        spec = hand.mjcf()
        prefix = f"collision_hand_{hand.side}_"
        for geom in spec.geoms:
            geom_name = str(getattr(geom, "name", "") or "")
            if geom_name.startswith(prefix):
                geom.contype = 1
                geom.conaffinity = 2

        _, prop_geom = prop.add_to(spec)
        prop_geom.contype = 2
        prop_geom.conaffinity = 1

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.root_body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{hand.side}_hand_base"))

        hand_geom_ids: set[int] = set()
        prop_geom_ids: set[int] = set()
        for geom_id in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if geom_name.startswith(prefix):
                hand_geom_ids.add(int(geom_id))
            if geom_name == str(prop_geom.name):
                prop_geom_ids.add(int(geom_id))
        self.hand_geom_ids = hand_geom_ids
        self.prop_geom_ids = prop_geom_ids

    def _apply_pose(self, hand_pose: np.ndarray) -> None:
        hand_pose = np.asarray(hand_pose, dtype=float).reshape(-1)
        root_pos = np.asarray(hand_pose[:3], dtype=float)
        root_quat = _pose_root_quat(hand_pose)
        qpos = np.asarray(hand_pose[9:], dtype=float)

        self.model.body_pos[self.root_body_id] = root_pos
        self.model.body_quat[self.root_body_id] = root_quat
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        if self.model.nu > 0:
            self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def evaluate(self, hand_pose: np.ndarray) -> ContactEval:
        self._apply_pose(hand_pose)

        contact_count = 0
        penetration_count = 0
        depth_sum = 0.0
        max_depth = 0.0
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom_1 = int(contact.geom1)
            geom_2 = int(contact.geom2)
            is_hand_prop = (
                (geom_1 in self.hand_geom_ids and geom_2 in self.prop_geom_ids)
                or (geom_2 in self.hand_geom_ids and geom_1 in self.prop_geom_ids)
            )
            if not is_hand_prop:
                continue
            contact_count += 1
            depth = max(-float(contact.dist), 0.0)
            if depth <= 0.0:
                continue
            penetration_count += 1
            depth_sum += depth
            max_depth = max(max_depth, depth)

        return ContactEval(
            contact_count=contact_count,
            penetration_count=penetration_count,
            depth_sum=depth_sum,
            max_depth=max_depth,
            energy=self.weight * depth_sum,
        )

    def evaluate_batch(self, hand_pose_batch: np.ndarray) -> ContactBatchEval:
        hand_pose_batch = np.asarray(hand_pose_batch, dtype=np.float32)
        if hand_pose_batch.ndim != 2:
            raise ValueError(f"hand_pose_batch must have shape (batch, pose_dim), got {hand_pose_batch.shape}")

        counts = np.zeros(hand_pose_batch.shape[0], dtype=np.int32)
        penetration_counts = np.zeros(hand_pose_batch.shape[0], dtype=np.int32)
        depth_sums = np.zeros(hand_pose_batch.shape[0], dtype=np.float32)
        max_depths = np.zeros(hand_pose_batch.shape[0], dtype=np.float32)
        energies = np.zeros(hand_pose_batch.shape[0], dtype=np.float32)
        for index, hand_pose in enumerate(hand_pose_batch):
            result = self.evaluate(hand_pose)
            counts[index] = int(result.contact_count)
            penetration_counts[index] = int(result.penetration_count)
            depth_sums[index] = float(result.depth_sum)
            max_depths[index] = float(result.max_depth)
            energies[index] = float(result.energy)

        return ContactBatchEval(
            contact_count=counts,
            penetration_count=penetration_counts,
            depth_sum=depth_sums,
            max_depth=max_depths,
            energy=energies,
        )
