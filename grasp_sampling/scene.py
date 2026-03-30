from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from grasp_gen.hand import Hand
from grasp_gen.prop import Prop


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OBJECT_DENSITY = 400.0
HAND_CONTYPE = 1
HAND_CONAFFINITY = 2
PROP_CONTYPE = 2
PROP_CONAFFINITY = 1


@dataclass(frozen=True)
class PropInertial:
    mass: float
    ipos: np.ndarray
    inertia: np.ndarray | None = None
    fullinertia: np.ndarray | None = None


def _unit_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float).reshape(4).copy()
    norm = np.linalg.norm(quat)
    if norm < 1.0e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    quat /= norm
    if quat[0] < 0.0:
        quat *= -1.0
    return quat


def _parse_vec(text: str, count: int) -> np.ndarray:
    values = [float(value) for value in str(text).split()]
    if len(values) != count:
        raise ValueError(f"Expected {count} values, got {text!r}")
    return np.asarray(values, dtype=np.float64)


def _parse_xml_inertial(xml_path: Path) -> PropInertial | None:
    if not xml_path.exists():
        return None
    root = ET.parse(xml_path).getroot()
    inertial = root.find("./worldbody/body/inertial")
    if inertial is None:
        return None
    mass = float(inertial.attrib.get("mass", "0"))
    if mass <= 0.0:
        return None
    ipos = _parse_vec(inertial.attrib.get("pos", "0 0 0"), 3)
    if "fullinertia" in inertial.attrib:
        return PropInertial(mass=mass, ipos=ipos, fullinertia=_parse_vec(inertial.attrib["fullinertia"], 6))
    if "diaginertia" in inertial.attrib:
        return PropInertial(mass=mass, ipos=ipos, inertia=_parse_vec(inertial.attrib["diaginertia"], 3))
    return PropInertial(mass=mass, ipos=ipos)


def infer_prop_inertial(prop: Prop, prop_meta: dict[str, object], *, density: float = DEFAULT_OBJECT_DENSITY) -> PropInertial:
    xml_rel = str(prop_meta.get("xml_path", "") or "")
    if xml_rel:
        xml_inertial = _parse_xml_inertial((ROOT / xml_rel).resolve())
        if xml_inertial is not None:
            return xml_inertial

    kind = str(prop_meta.get("kind", "cylinder"))
    ipos = np.asarray(prop.com_local, dtype=np.float64)
    density = max(float(density), 1.0e-6)

    if kind == "cube":
        size = float(prop_meta.get("size", 0.07))
        mass = density * (size**3)
        inertia = np.full(3, (mass * size * size) / 6.0, dtype=np.float64)
        return PropInertial(mass=mass, ipos=ipos, inertia=inertia)

    if kind == "cylinder":
        radius = float(prop_meta.get("radius", 0.045))
        half_height = float(prop_meta.get("half_height", 0.165))
        height = 2.0 * half_height
        mass = density * np.pi * radius * radius * height
        i_xy = mass * (3.0 * radius * radius + height * height) / 12.0
        i_z = 0.5 * mass * radius * radius
        inertia = np.asarray([i_xy, i_xy, i_z], dtype=np.float64)
        return PropInertial(mass=mass, ipos=ipos, inertia=inertia)

    centered = np.asarray(prop.vertices, dtype=np.float64) - np.mean(np.asarray(prop.vertices, dtype=np.float64), axis=0)
    radius = float(np.max(np.linalg.norm(centered, axis=1)))
    mass = density * (4.0 / 3.0) * np.pi * radius**3
    inertia = np.full(3, 0.4 * mass * radius * radius, dtype=np.float64)
    return PropInertial(mass=mass, ipos=ipos, inertia=inertia)


@dataclass
class PhysicsScene:
    model: mujoco.MjModel
    data: mujoco.MjData
    root_body_id: int
    hand_qpos_count: int
    ctrl_count: int
    prop_qpos_adr: int
    prop_qvel_adr: int
    hand_geom_ids: set[int]
    prop_geom_ids: set[int]
    base_root_pos: np.ndarray
    base_root_quat: np.ndarray
    base_qpos_target: np.ndarray
    base_prop_pos: np.ndarray
    base_prop_quat: np.ndarray

    def reset(self, *, qpos_target: np.ndarray | None = None) -> None:
        qpos_target = self.base_qpos_target if qpos_target is None else np.asarray(qpos_target, dtype=np.float64).reshape(-1)
        self.model.body_pos[self.root_body_id] = np.asarray(self.base_root_pos, dtype=np.float64)
        self.model.body_quat[self.root_body_id] = _unit_quat(self.base_root_quat)
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[: self.hand_qpos_count] = qpos_target
        self.data.qpos[self.prop_qpos_adr : self.prop_qpos_adr + 3] = self.base_prop_pos
        self.data.qpos[self.prop_qpos_adr + 3 : self.prop_qpos_adr + 7] = _unit_quat(self.base_prop_quat)
        if self.ctrl_count > 0:
            self.data.ctrl[:] = qpos_target[: self.ctrl_count]
        mujoco.mj_forward(self.model, self.data)

    def step(self, *, root_pos: np.ndarray, root_quat: np.ndarray, qpos_target: np.ndarray) -> None:
        self.model.body_pos[self.root_body_id] = np.asarray(root_pos, dtype=np.float64)
        self.model.body_quat[self.root_body_id] = _unit_quat(root_quat)
        if self.ctrl_count > 0:
            self.data.ctrl[:] = np.asarray(qpos_target, dtype=np.float64)[: self.ctrl_count]
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_step(self.model, self.data)

    def object_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = np.asarray(self.data.qpos[self.prop_qpos_adr : self.prop_qpos_adr + 3], dtype=np.float64).copy()
        quat = _unit_quat(np.asarray(self.data.qpos[self.prop_qpos_adr + 3 : self.prop_qpos_adr + 7], dtype=np.float64))
        return pos, quat

    def contact_counts(self) -> tuple[int, int, float, float]:
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
        return contact_count, penetration_count, depth_sum, max_depth


def build_physics_scene(
    hand: Hand,
    prop: Prop,
    prop_meta: dict[str, object],
    *,
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    qpos_target: np.ndarray,
    timestep: float,
    density: float,
) -> PhysicsScene:
    spec = hand.mjcf(root_pos=root_pos, root_quat=root_quat)
    spec.option.timestep = float(timestep)
    spec.option.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)

    prefix = f"collision_hand_{hand.side}_"
    for geom in spec.geoms:
        geom_name = str(getattr(geom, "name", "") or "")
        if geom_name.startswith(prefix):
            geom.contype = HAND_CONTYPE
            geom.conaffinity = HAND_CONAFFINITY

    inertial = infer_prop_inertial(prop, prop_meta, density=density)
    mesh = spec.add_mesh()
    mesh.name = f"{prop.name}_physics_mesh"
    mesh.uservert = np.asarray(prop.vertices, dtype=np.float64).reshape(-1).tolist()
    mesh.userface = np.asarray(prop.faces, dtype=np.int32).reshape(-1).tolist()

    body = spec.worldbody.add_body()
    body.name = f"{prop.name}_physics_body"
    body.pos = np.asarray(prop.pos, dtype=np.float64).copy()
    body.quat = _unit_quat(np.asarray(prop.quat, dtype=np.float64))
    body.ipos = np.asarray(inertial.ipos, dtype=np.float64).copy()
    body.mass = float(inertial.mass)
    if inertial.fullinertia is not None:
        body.fullinertia = np.asarray(inertial.fullinertia, dtype=np.float64).copy()
    elif inertial.inertia is not None:
        body.inertia = np.asarray(inertial.inertia, dtype=np.float64).copy()
    body.add_freejoint(name=f"{prop.name}_physics_freejoint")

    geom = body.add_geom()
    geom.name = f"{prop.name}_physics_geom"
    geom.type = mujoco.mjtGeom.mjGEOM_MESH
    geom.meshname = mesh.name
    geom.contype = PROP_CONTYPE
    geom.conaffinity = PROP_CONAFFINITY
    geom.condim = int(prop.condim)
    geom.friction = np.asarray(prop.friction, dtype=np.float64).copy()
    geom.rgba = np.asarray(prop.rgba, dtype=np.float64).copy()

    model = spec.compile()
    data = mujoco.MjData(model)

    root_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{hand.side}_hand_base"))
    freejoint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prop.name}_physics_freejoint"))
    prop_qpos_adr = int(model.jnt_qposadr[freejoint_id])
    prop_qvel_adr = int(model.jnt_dofadr[freejoint_id])

    hand_geom_ids: set[int] = set()
    prop_geom_ids: set[int] = set()
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if geom_name.startswith(prefix):
            hand_geom_ids.add(int(geom_id))
        if geom_name == geom.name:
            prop_geom_ids.add(int(geom_id))

    scene = PhysicsScene(
        model=model,
        data=data,
        root_body_id=root_body_id,
        hand_qpos_count=int(hand.model.nq),
        ctrl_count=int(hand.model.nu),
        prop_qpos_adr=prop_qpos_adr,
        prop_qvel_adr=prop_qvel_adr,
        hand_geom_ids=hand_geom_ids,
        prop_geom_ids=prop_geom_ids,
        base_root_pos=np.asarray(root_pos, dtype=np.float64).copy(),
        base_root_quat=_unit_quat(np.asarray(root_quat, dtype=np.float64)),
        base_qpos_target=np.asarray(qpos_target, dtype=np.float64).copy(),
        base_prop_pos=np.asarray(prop.pos, dtype=np.float64).copy(),
        base_prop_quat=_unit_quat(np.asarray(prop.quat, dtype=np.float64)),
    )
    scene.reset()
    return scene
