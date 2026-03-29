from __future__ import annotations

import mujoco
import numpy as np

from .prop_cloud import SurfacePointCloud


def _unit_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=float).reshape(4).copy()
    norm = np.linalg.norm(quat)
    if norm < 1.0e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    quat /= norm
    if quat[0] < 0.0:
        quat *= -1.0
    return quat


def _copy_surface_cloud(surface_cloud: SurfacePointCloud) -> SurfacePointCloud:
    points_local = np.asarray(surface_cloud.points_local, dtype=np.float32)
    normals_local = np.asarray(surface_cloud.normals_local, dtype=np.float32)
    area_weights = np.asarray(surface_cloud.area_weights, dtype=np.float32)
    if points_local.ndim != 2 or points_local.shape[1] != 3:
        raise ValueError(f"surface_cloud.points_local must have shape (N, 3), got {points_local.shape}")
    if normals_local.shape != points_local.shape:
        raise ValueError(
            "surface_cloud.normals_local must match surface_cloud.points_local shape, "
            f"got {normals_local.shape} vs {points_local.shape}"
        )
    if area_weights.shape != (len(points_local),):
        raise ValueError(
            "surface_cloud.area_weights must have shape (N,), "
            f"got {area_weights.shape} for N={len(points_local)}"
        )
    return SurfacePointCloud(
        points_local=points_local.copy(),
        normals_local=normals_local.copy(),
        area_weights=area_weights.copy(),
        spacing=float(surface_cloud.spacing),
        surface_area=float(surface_cloud.surface_area),
    )


class Prop:
    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        pos: np.ndarray,
        com_local: np.ndarray | None = None,
        quat: np.ndarray | None = None,
        name: str = "prop",
        rgba: np.ndarray | None = None,
        friction: np.ndarray | None = None,
        condim: int = 4,
        surface_cloud: SurfacePointCloud | None = None,
    ):
        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"vertices must have shape (N, 3), got {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must have shape (M, 3), got {faces.shape}")
        if len(vertices) < 4:
            raise ValueError("A mesh prop needs at least 4 vertices.")
        if len(faces) == 0:
            raise ValueError("A mesh prop needs at least one face.")
        if np.min(faces) < 0 or np.max(faces) >= len(vertices):
            raise ValueError("faces contain an out-of-range vertex index.")

        self.vertices = vertices.copy()
        self.faces = faces.copy()
        self.pos = np.asarray(pos, dtype=float).reshape(3).copy()
        self.com_local = (
            np.zeros(3, dtype=float)
            if com_local is None
            else np.asarray(com_local, dtype=float).reshape(3).copy()
        )
        self.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float) if quat is None else _unit_quat(quat)
        self.name = name
        self.rgba = (
            np.array([0.91, 0.58, 0.19, 1.0], dtype=float)
            if rgba is None
            else np.asarray(rgba, dtype=float).reshape(4).copy()
        )
        self.friction = (
            np.array([1.1, 0.05, 0.01], dtype=float)
            if friction is None
            else np.asarray(friction, dtype=float).reshape(3).copy()
        )
        self.condim = int(condim)
        self.surface_cloud = None if surface_cloud is None else _copy_surface_cloud(surface_cloud)

    def add_to(
        self,
        spec: mujoco.MjSpec,
        *,
        body_name: str | None = None,
        geom_name: str | None = None,
        mesh_name: str | None = None,
    ) -> tuple[mujoco.MjsBody, mujoco.MjsGeom]:
        mesh = spec.add_mesh()
        mesh.name = self.name if mesh_name is None else mesh_name
        mesh.uservert = self.vertices.reshape(-1).tolist()
        mesh.userface = self.faces.reshape(-1).tolist()

        body = spec.worldbody.add_body()
        body.name = self.name if body_name is None else body_name
        body.pos = self.pos.copy()
        body.quat = self.quat.copy()

        geom = body.add_geom()
        geom.name = f"{body.name}_geom" if geom_name is None else geom_name
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = mesh.name
        geom.condim = self.condim
        geom.friction = self.friction.copy()
        geom.rgba = self.rgba.copy()
        return body, geom

    def mj(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        spec = mujoco.MjSpec()
        spec.modelname = self.name
        self.add_to(spec)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return model, data
