from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .primitives import box_mesh, cylinder_mesh, ellipsoid_mesh, octahedron_mesh, regular_prism_mesh


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ASSET_ROOT = ROOT / "assets" / "generated"
SUPPORTED_PRIMITIVES = {"box", "cylinder", "prism", "ellipsoid", "octahedron"}


@dataclass(frozen=True)
class MeshAssetConfig:
    name: str
    primitive: str
    mesh_filename: str
    rgba: tuple[float, float, float, float]
    friction: tuple[float, float, float]
    condim: int
    size_xyz: tuple[float, float, float] | None = None
    radius: float | None = None
    half_height: float | None = None
    sides: int | None = None
    radii_xyz: tuple[float, float, float] | None = None
    latitude_segments: int = 16
    longitude_segments: int = 24


def _require_keys(raw: dict[str, Any], section_name: str, keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in raw]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{section_name} is missing required keys: {joined}")


def _tuple3(raw_value: Any, *, field_name: str) -> tuple[float, float, float]:
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 3:
        raise ValueError(f"{field_name} must be a 3-element array.")
    return tuple(float(value) for value in raw_value)


def _tuple4(raw_value: Any, *, field_name: str) -> tuple[float, float, float, float]:
    if not isinstance(raw_value, (list, tuple)) or len(raw_value) != 4:
        raise ValueError(f"{field_name} must be a 4-element array.")
    return tuple(float(value) for value in raw_value)


def _load_asset_config(raw_asset: dict[str, Any], *, index: int) -> MeshAssetConfig:
    _require_keys(raw_asset, f"assets[{index}]", ("name", "primitive"))
    primitive = str(raw_asset["primitive"]).strip().lower()
    if primitive not in SUPPORTED_PRIMITIVES:
        supported = ", ".join(sorted(SUPPORTED_PRIMITIVES))
        raise ValueError(f"assets[{index}].primitive must be one of: {supported}.")

    return MeshAssetConfig(
        name=str(raw_asset["name"]).strip(),
        primitive=primitive,
        mesh_filename=str(raw_asset.get("mesh_filename", "collision.obj")).strip() or "collision.obj",
        rgba=_tuple4(raw_asset.get("rgba", [0.91, 0.58, 0.19, 1.0]), field_name="rgba"),
        friction=_tuple3(raw_asset.get("friction", [1.1, 0.05, 0.01]), field_name="friction"),
        condim=int(raw_asset.get("condim", 4)),
        size_xyz=None if "size_xyz" not in raw_asset else _tuple3(raw_asset["size_xyz"], field_name="size_xyz"),
        radius=None if "radius" not in raw_asset else float(raw_asset["radius"]),
        half_height=None if "half_height" not in raw_asset else float(raw_asset["half_height"]),
        sides=None if "sides" not in raw_asset else int(raw_asset["sides"]),
        radii_xyz=None if "radii_xyz" not in raw_asset else _tuple3(raw_asset["radii_xyz"], field_name="radii_xyz"),
        latitude_segments=int(raw_asset.get("latitude_segments", 16)),
        longitude_segments=int(raw_asset.get("longitude_segments", 24)),
    )


def load_asset_generation_config(config_path: Path) -> tuple[Path, tuple[MeshAssetConfig, ...]]:
    resolved_path = config_path.expanduser().resolve()
    with resolved_path.open("rb") as stream:
        raw = tomllib.load(stream)

    output_raw = raw.get("output", {})
    assets_raw = raw.get("assets")
    if not isinstance(assets_raw, list) or not assets_raw:
        raise ValueError("Config must define at least one [[assets]] entry.")

    asset_root = DEFAULT_ASSET_ROOT
    if "asset_root" in output_raw:
        path = Path(str(output_raw["asset_root"])).expanduser()
        asset_root = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    assets = tuple(_load_asset_config(asset_raw, index=index) for index, asset_raw in enumerate(assets_raw))
    _validate_assets(assets)
    return asset_root, assets


def _validate_assets(assets: tuple[MeshAssetConfig, ...]) -> None:
    seen_names: set[str] = set()
    for asset in assets:
        if not asset.name:
            raise ValueError("Asset names must not be empty.")
        if asset.name in seen_names:
            raise ValueError(f"Duplicate asset name: {asset.name!r}")
        seen_names.add(asset.name)
        if asset.condim <= 0:
            raise ValueError(f"Asset {asset.name!r} condim must be positive.")
        if any(value < 0.0 for value in asset.rgba[:3]) or asset.rgba[3] < 0.0:
            raise ValueError(f"Asset {asset.name!r} rgba entries must be non-negative.")
        if any(value <= 0.0 for value in asset.friction):
            raise ValueError(f"Asset {asset.name!r} friction entries must be positive.")
        if asset.primitive == "box":
            if asset.size_xyz is None or any(value <= 0.0 for value in asset.size_xyz):
                raise ValueError(f"Asset {asset.name!r} box requires positive size_xyz.")
        if asset.primitive in {"cylinder", "prism"}:
            if asset.radius is None or asset.radius <= 0.0:
                raise ValueError(f"Asset {asset.name!r} {asset.primitive} requires a positive radius.")
            if asset.half_height is None or asset.half_height <= 0.0:
                raise ValueError(f"Asset {asset.name!r} {asset.primitive} requires a positive half_height.")
            sides = 32 if asset.sides is None else asset.sides
            minimum = 3 if asset.primitive == "cylinder" else 5
            if sides < minimum:
                raise ValueError(f"Asset {asset.name!r} {asset.primitive} requires sides >= {minimum}.")
        if asset.primitive == "ellipsoid":
            if asset.radii_xyz is None or any(value <= 0.0 for value in asset.radii_xyz):
                raise ValueError(f"Asset {asset.name!r} ellipsoid requires positive radii_xyz.")
            if asset.latitude_segments < 3 or asset.longitude_segments < 3:
                raise ValueError(f"Asset {asset.name!r} ellipsoid segments must be at least 3.")
        if asset.primitive == "octahedron" and (asset.radius is None or asset.radius <= 0.0):
            raise ValueError(f"Asset {asset.name!r} octahedron requires a positive radius.")


def _build_mesh(asset: MeshAssetConfig) -> tuple[np.ndarray, np.ndarray]:
    if asset.primitive == "box":
        assert asset.size_xyz is not None
        return box_mesh(asset.size_xyz)
    if asset.primitive == "cylinder":
        assert asset.radius is not None
        assert asset.half_height is not None
        sides = 32 if asset.sides is None else asset.sides
        return cylinder_mesh(asset.radius, asset.half_height, sides=sides)
    if asset.primitive == "prism":
        assert asset.radius is not None
        assert asset.half_height is not None
        sides = 6 if asset.sides is None else asset.sides
        return regular_prism_mesh(asset.radius, asset.half_height, sides=sides)
    if asset.primitive == "ellipsoid":
        assert asset.radii_xyz is not None
        return ellipsoid_mesh(
            asset.radii_xyz,
            latitude_segments=asset.latitude_segments,
            longitude_segments=asset.longitude_segments,
        )
    assert asset.radius is not None
    return octahedron_mesh(asset.radius)


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    lines: list[str] = []
    for vertex in np.asarray(vertices, dtype=float):
        lines.append(f"v {vertex[0]:.9f} {vertex[1]:.9f} {vertex[2]:.9f}")
    for face in np.asarray(faces, dtype=np.int32):
        a, b, c = (int(index) + 1 for index in face)
        lines.append(f"f {a} {b} {c}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _asset_metadata(asset_root: Path, asset: MeshAssetConfig, mesh_path: Path) -> dict[str, Any]:
    metadata = {
        "kind": "mesh_asset",
        "asset_name": asset.name,
        "name": asset.name,
        "mesh_path": str(mesh_path.relative_to(ROOT)),
        "com_local": [0.0, 0.0, 0.0],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "pos": [0.0, 0.0, 0.0],
        "rgba": list(asset.rgba),
        "friction": list(asset.friction),
        "condim": int(asset.condim),
        "generator": {
            "primitive": asset.primitive,
        },
    }
    if asset.size_xyz is not None:
        metadata["generator"]["size_xyz"] = list(asset.size_xyz)
    if asset.radius is not None:
        metadata["generator"]["radius"] = float(asset.radius)
    if asset.half_height is not None:
        metadata["generator"]["half_height"] = float(asset.half_height)
    if asset.sides is not None:
        metadata["generator"]["sides"] = int(asset.sides)
    if asset.radii_xyz is not None:
        metadata["generator"]["radii_xyz"] = list(asset.radii_xyz)
        metadata["generator"]["latitude_segments"] = int(asset.latitude_segments)
        metadata["generator"]["longitude_segments"] = int(asset.longitude_segments)
    return metadata


def build_asset(asset_root: Path, asset: MeshAssetConfig) -> Path:
    asset_dir = asset_root / asset.name
    asset_dir.mkdir(parents=True, exist_ok=True)
    vertices, faces = _build_mesh(asset)

    mesh_path = asset_dir / asset.mesh_filename
    _write_obj(mesh_path, vertices, faces)

    metadata = _asset_metadata(asset_root, asset, mesh_path)
    metadata_path = asset_dir / "asset.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path
