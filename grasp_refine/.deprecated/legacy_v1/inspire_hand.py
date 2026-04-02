from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
HAND_XML = {
    "right": ROOT / "assets" / "inspire" / "right.xml",
    "left": ROOT / "assets" / "inspire" / "left.xml",
}


@dataclass(frozen=True)
class InspireHandSpec:
    side: str
    joint_lower: np.ndarray
    joint_upper: np.ndarray

    @property
    def pose_dim(self) -> int:
        return 3 + 6 + int(self.joint_lower.shape[0])


def load_inspire_hand_spec(side: str = "right") -> InspireHandSpec:
    if side not in HAND_XML:
        raise ValueError(f"Unsupported inspire hand side: {side!r}")

    model = mujoco.MjModel.from_xml_path(str(HAND_XML[side]))
    actuated_qpos_indices: list[int] = []
    lower_by_qpos: dict[int, float] = {}
    upper_by_qpos: dict[int, float] = {}

    for actuator_index in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_index, 0])
        qpos_index = int(model.jnt_qposadr[joint_id])
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_index) or ""
        if "_pos_" in actuator_name or "_rot_" in actuator_name:
            continue
        actuated_qpos_indices.append(qpos_index)
        lower_by_qpos[qpos_index] = float(model.jnt_range[joint_id, 0])
        upper_by_qpos[qpos_index] = float(model.jnt_range[joint_id, 1])

    ordered = sorted(set(actuated_qpos_indices))
    joint_lower = np.asarray([lower_by_qpos[index] for index in ordered], dtype=np.float32)
    joint_upper = np.asarray([upper_by_qpos[index] for index in ordered], dtype=np.float32)
    return InspireHandSpec(side=side, joint_lower=joint_lower, joint_upper=joint_upper)
