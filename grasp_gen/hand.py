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
    finger: str
    role: str
    lo: float
    hi: float


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


def _build_actuators(model: mujoco.MjModel) -> list[ActuatorSpec]:
    actuators: list[ActuatorSpec] = []
    for actuator_index in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_index) or f"act_{actuator_index}"
        parts = name.split("_")
        prefix = 1 if len(parts) > 1 and parts[0] == "inspire" else 0
        finger = parts[prefix + 1] if len(parts) - prefix >= 3 else "unknown"
        role = "_".join(parts[prefix + 2 : -1]) or name
        actuators.append(
            ActuatorSpec(
                idx=actuator_index,
                name=name,
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


class Hand:
    def __init__(self, side: str):
        if side not in HAND_XML:
            raise ValueError(f"Unsupported hand side: {side}")

        self.side = side
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

    def reset_root(self) -> None:
        self.model.body_pos[self.root_body_id] = self.root_home_pos.copy()
        self.model.body_quat[self.root_body_id] = self.root_home_quat.copy()
        mujoco.mj_forward(self.model, self.data)

    def root_to_palm(self) -> tuple[np.ndarray, np.ndarray]:
        root_pos = self.data.xpos[self.root_body_id].copy()
        root_rot = self.data.xmat[self.root_body_id].reshape(3, 3)
        palm_pos = self.data.site_xpos[self.palm_site_id].copy()
        palm_rot = self.data.site_xmat[self.palm_site_id].reshape(3, 3)
        return root_rot.T @ (palm_pos - root_pos), root_rot.T @ palm_rot

    def body_local_to_world(self, body_ids: np.ndarray, local_pos: np.ndarray) -> np.ndarray:
        body_ids = np.asarray(body_ids, dtype=np.intp)
        local_pos = np.asarray(local_pos, dtype=float)
        if local_pos.size == 0:
            return np.zeros((0, 3), dtype=float)
        body_pos = self.data.xpos[body_ids]
        body_rot = self.data.xmat[body_ids].reshape(-1, 3, 3)
        return body_pos + np.einsum("nij,nj->ni", body_rot, local_pos)
