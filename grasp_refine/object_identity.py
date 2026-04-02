from __future__ import annotations

from typing import Any, Mapping


def _format_number(value: Any) -> str:
    return format(float(value), ".9g")


def build_object_key(
    *,
    object_metadata: Mapping[str, Any] | None,
    object_kind: str,
    object_name: str,
) -> str:
    metadata = dict(object_metadata or {})
    mesh_path = metadata.get("mesh_path")
    if mesh_path:
        return f"mesh:{mesh_path}"

    kind = str(object_kind).strip().lower()
    if kind == "cube" and "size" in metadata:
        return f"cube:size={_format_number(metadata['size'])}"
    if kind == "cylinder" and "radius" in metadata and "half_height" in metadata:
        return (
            f"cylinder:r={_format_number(metadata['radius'])}"
            f":hh={_format_number(metadata['half_height'])}"
        )

    name = str(object_name).strip()
    if name:
        return f"name:{name}"
    if kind:
        return f"kind:{kind}"
    return "unknown"


def build_saved_object_key(
    *,
    object_kind: str,
    object_name: str,
    object_key: str | None = None,
) -> str:
    candidate = str(object_key or "").strip()
    if candidate:
        return candidate
    return build_object_key(
        object_metadata=None,
        object_kind=object_kind,
        object_name=object_name,
    )
