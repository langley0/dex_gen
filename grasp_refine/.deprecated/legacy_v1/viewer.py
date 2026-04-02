from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import jax
import mujoco
import numpy as np

from .checkpoint import load_training_checkpoint
from .config_io import training_config_from_dict
from .diffusion import make_diffusion_schedule
from .inspire_hand import HAND_XML, InspireHandSpec, load_inspire_hand_spec
from .io import GraspRecord, load_grasp_artifact
from .object_mesh import load_object_mesh, sample_mesh_points
from .sampling import sample_grasp_poses


ROOT = Path(__file__).resolve().parent.parent
INSPIRE_ASSET_DIR = ROOT / "assets" / "inspire" / "assets"


@dataclass(frozen=True)
class ViewerState:
    checkpoint_path: Path
    artifact_path: Path
    hand_side: str
    object_name: str
    object_kind: str
    source_pose: np.ndarray
    generated_poses: np.ndarray
    object_points: np.ndarray
    object_normals: np.ndarray
    source_joint_limit_loss: float
    source_root_distance_loss: float
    generated_joint_limit_loss: np.ndarray
    generated_root_distance_loss: np.ndarray


def _stable_kind_seed(object_kind: str, base_seed: int) -> int:
    stable_hash = sum((index + 1) * ord(char) for index, char in enumerate(object_kind))
    return int(base_seed) + stable_hash % 100_000


def _absolutize_mesh_paths(spec: mujoco.MjSpec, asset_dir: Path) -> None:
    for mesh in spec.meshes:
        if mesh.file and not Path(mesh.file).is_absolute():
            mesh.file = str((asset_dir / mesh.file).resolve())


def _remove_inspire_root_dofs(hand_spec: mujoco.MjSpec, side: str) -> None:
    root_actuators = [
        f"{side}_pos_x_position",
        f"{side}_pos_y_position",
        f"{side}_pos_z_position",
        f"{side}_rot_x_position",
        f"{side}_rot_y_position",
        f"{side}_rot_z_position",
    ]
    root_joints = [
        f"{side}_pos_x",
        f"{side}_pos_y",
        f"{side}_pos_z",
        f"{side}_rot_x",
        f"{side}_rot_y",
        f"{side}_rot_z",
    ]
    for actuator_name in root_actuators:
        actuator = hand_spec.actuator(actuator_name)
        if actuator is not None:
            hand_spec.delete(actuator)
    for joint_name in root_joints:
        joint = hand_spec.joint(joint_name)
        if joint is not None:
            hand_spec.delete(joint)


def _load_hand_spec_for_view(side: str) -> mujoco.MjSpec:
    hand_spec = mujoco.MjSpec.from_file(str(HAND_XML[side]))
    _absolutize_mesh_paths(hand_spec, INSPIRE_ASSET_DIR)
    _remove_inspire_root_dofs(hand_spec, side)
    for light in list(hand_spec.lights):
        hand_spec.delete(light)
    hand_spec.modelname = f"inspire_{side}_viewer"
    return hand_spec


def build_view_model(side: str) -> tuple[mujoco.MjModel, mujoco.MjData, int]:
    spec = mujoco.MjSpec()
    spec.modelname = f"grasp_refine_model_view_{side}"
    spec.stat.center = np.array([0.0, 0.0, 0.1], dtype=float)
    spec.stat.extent = 0.8
    spec.visual.global_.offwidth = 1600
    spec.visual.global_.offheight = 1200

    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = np.array([0.0, 0.0, 0.05], dtype=float)
    floor.pos = np.array([0.0, 0.0, -0.001], dtype=float)
    floor.rgba = np.array([0.92, 0.93, 0.95, 1.0], dtype=float)
    floor.contype = 0
    floor.conaffinity = 0

    light = spec.worldbody.add_light()
    light.name = "key_light"
    light.pos = np.array([0.3, -0.6, 1.2], dtype=float)
    light.dir = np.array([0.0, 0.1, -1.0], dtype=float)
    light.diffuse = np.array([0.95, 0.95, 0.95], dtype=float)
    light.specular = np.array([0.25, 0.25, 0.25], dtype=float)
    light.castshadow = True

    mount_site = spec.worldbody.add_site()
    mount_site.name = "hand_mount_site"
    mount_site.pos = np.zeros(3, dtype=float)
    mount_site.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    hand_spec = _load_hand_spec_for_view(side)
    spec.attach(hand_spec, prefix="inspire_", site=mount_site)

    model = spec.compile()
    data = mujoco.MjData(model)
    root_body_name = f"inspire_{side}_hand_base"
    root_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name))
    if root_body_id < 0:
        raise ValueError(f"Viewer hand body {root_body_name!r} was not found.")
    mujoco.mj_forward(model, data)
    return model, data, root_body_id


def set_bright_background(model: mujoco.MjModel) -> None:
    model.vis.rgba.haze[:] = np.array([0.97, 0.97, 0.99, 1.0], dtype=float)
    model.vis.rgba.fog[:] = np.array([0.97, 0.97, 0.99, 1.0], dtype=float)
    model.vis.headlight.ambient[:] = np.array([0.55, 0.55, 0.55], dtype=float)
    model.vis.headlight.diffuse[:] = np.array([0.85, 0.85, 0.85], dtype=float)
    model.vis.headlight.specular[:] = np.array([0.15, 0.15, 0.15], dtype=float)


def configure_camera(cam: mujoco.MjvCamera) -> None:
    cam.lookat[:] = np.array([0.0, 0.0, 0.08], dtype=float)
    cam.distance = 0.62
    cam.azimuth = 142.0
    cam.elevation = -20.0


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
        s = math.sqrt(trace + 1.0) * 2.0
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
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
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
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
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
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        quat = np.array(
            [
                (rotation[1, 0] - rotation[0, 1]) / s,
                (rotation[0, 2] + rotation[2, 0]) / s,
                (rotation[1, 2] + rotation[2, 1]) / s,
                0.25 * s,
            ],
            dtype=float,
        )
    quat /= max(float(np.linalg.norm(quat)), 1.0e-8)
    if quat[0] < 0.0:
        quat *= -1.0
    return quat


def apply_hand_pose(model: mujoco.MjModel, data: mujoco.MjData, root_body_id: int, pose: np.ndarray) -> None:
    pose_array = np.asarray(pose, dtype=float)
    model.body_pos[root_body_id] = pose_array[:3]
    model.body_quat[root_body_id] = _matrix_to_quat_np(_ortho6d_to_matrix_np(pose_array[3:9]))
    data.qpos[:] = pose_array[9:]
    data.qvel[:] = 0.0
    if model.nu > 0:
        data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def add_marker(scene, idx: int, pos: np.ndarray, radius: float, rgba: np.ndarray) -> int:
    limit = int(getattr(scene, "maxgeom", len(scene.geoms)))
    if idx >= limit:
        return idx
    mujoco.mjv_initGeom(
        scene.geoms[idx],
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        np.array([radius, 0.0, 0.0], dtype=float),
        np.asarray(pos, dtype=float),
        np.eye(3, dtype=float).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    return idx + 1


def _select_record(artifact_path: Path, *, state_name: str, sample_index: int) -> GraspRecord:
    payload = load_grasp_artifact(artifact_path, state_name=state_name)
    if not payload.records:
        raise ValueError(f"No grasp records were loaded from {artifact_path}.")
    return payload.records[int(sample_index) % len(payload.records)]


def _object_cloud_from_record(record: GraspRecord, *, num_points: int, object_point_seed: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = load_object_mesh(record.object_metadata)
    seed = _stable_kind_seed(record.object_kind, base_seed=object_point_seed)
    return sample_mesh_points(mesh, int(num_points), seed=seed)


def _scalar_losses(
    poses: np.ndarray,
    *,
    hand_spec: InspireHandSpec,
    object_points: np.ndarray,
    root_distance_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    pose_array = np.asarray(poses, dtype=np.float32)
    if pose_array.ndim == 1:
        pose_array = pose_array[None, :]
    joints = pose_array[:, 9:]
    below = np.maximum(hand_spec.joint_lower[None, :] - joints, 0.0)
    above = np.maximum(joints - hand_spec.joint_upper[None, :], 0.0)
    joint = np.mean(below + above, axis=1)

    root = pose_array[:, :3]
    diff = root[:, None, :] - object_points[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    nearest = np.min(distances, axis=1)
    root_loss = np.maximum(nearest - float(root_distance_threshold), 0.0)
    return joint.astype(np.float32), root_loss.astype(np.float32)


def load_viewer_state(
    checkpoint_path: str | Path,
    artifact_path: str | Path,
    *,
    state_name: str = "best",
    sample_index: int = 0,
    seed: int = 0,
    num_generated: int = 1,
    object_num_points: int | None = None,
    device: str | None = None,
) -> ViewerState:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    artifact_path = Path(artifact_path).expanduser().resolve()

    payload = load_training_checkpoint(checkpoint_path)
    train_config = training_config_from_dict(payload["config"])
    if device is not None:
        train_config = replace(train_config, device=str(device))
    hand_spec = load_inspire_hand_spec(train_config.hand_side)
    model_config = train_config.model
    if model_config.pose_dim != hand_spec.pose_dim:
        model_config = replace(train_config.model, pose_dim=hand_spec.pose_dim)

    record = _select_record(artifact_path, state_name=state_name, sample_index=sample_index)
    num_points = train_config.dataset.object_num_points if object_num_points is None else int(object_num_points)
    object_points, object_normals = _object_cloud_from_record(
        record,
        num_points=num_points,
        object_point_seed=train_config.dataset.object_point_seed,
    )

    schedule = make_diffusion_schedule(train_config.diffusion)
    if device is not None:
        try:
            jax_device = jax.devices(device)[0]
        except Exception:
            jax_device = jax.devices()[0]
        params = jax.device_put(payload["params"], jax_device)
        schedule = jax.device_put(schedule, jax_device)
        object_points_jax = jax.device_put(jax.numpy.asarray(object_points), jax_device)
        object_normals_jax = jax.device_put(jax.numpy.asarray(object_normals), jax_device)
    else:
        params = payload["params"]
        object_points_jax = jax.numpy.asarray(object_points)
        object_normals_jax = jax.numpy.asarray(object_normals)
    rng_key = jax.random.key(np.uint32(int(seed) % (2**32)))
    sample_batch = sample_grasp_poses(
        params,
        object_points=object_points_jax,
        object_normals=object_normals_jax,
        rng_key=rng_key,
        model_config=model_config,
        diffusion_config=train_config.diffusion,
        normalizer=payload["normalizer"],
        schedule=schedule,
        num_samples=int(num_generated),
    )
    generated_poses = np.asarray(sample_batch.pose, dtype=np.float32)

    source_joint, source_root = _scalar_losses(
        record.pose,
        hand_spec=hand_spec,
        object_points=object_points,
        root_distance_threshold=train_config.loss.root_distance_threshold,
    )
    generated_joint, generated_root = _scalar_losses(
        generated_poses,
        hand_spec=hand_spec,
        object_points=object_points,
        root_distance_threshold=train_config.loss.root_distance_threshold,
    )

    return ViewerState(
        checkpoint_path=checkpoint_path,
        artifact_path=artifact_path,
        hand_side=train_config.hand_side,
        object_name=record.object_name,
        object_kind=record.object_kind,
        source_pose=np.asarray(record.pose, dtype=np.float32),
        generated_poses=generated_poses,
        object_points=np.asarray(object_points, dtype=np.float32),
        object_normals=np.asarray(object_normals, dtype=np.float32),
        source_joint_limit_loss=float(source_joint[0]),
        source_root_distance_loss=float(source_root[0]),
        generated_joint_limit_loss=np.asarray(generated_joint, dtype=np.float32),
        generated_root_distance_loss=np.asarray(generated_root, dtype=np.float32),
    )
