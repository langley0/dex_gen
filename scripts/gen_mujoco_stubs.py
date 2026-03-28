#!/usr/bin/env python3
from __future__ import annotations

import inspect
import keyword
from pathlib import Path
from typing import Any

import mujoco
from mujoco.introspect.ast_nodes import (
    AnonymousStructDecl,
    AnonymousUnionDecl,
    ArrayType,
    PointerType,
    ValueType,
)
from mujoco.introspect.enums import ENUMS
from mujoco.introspect.functions import FUNCTIONS
from mujoco.introspect.structs import STRUCTS


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "typings" / "mujoco" / "__init__.pyi"

RUNTIME_NAMES = set(dir(mujoco))
ENUM_NAMES = {name for name in ENUMS if name in RUNTIME_NAMES}
STRUCT_NAME_MAP = {
    c_name: py_name
    for c_name in STRUCTS
    if (py_name := f"{c_name[0].upper()}{c_name[1:]}") in RUNTIME_NAMES
}

INT_TYPES = {
    "int",
    "unsigned int",
    "unsigned",
    "size_t",
    "uintptr_t",
    "uint64_t",
    "uint32_t",
    "uint16_t",
    "uint8_t",
    "signed char",
    "unsigned char",
    "char",
    "short",
    "unsigned short",
    "long",
    "unsigned long",
    "long long",
    "unsigned long long",
    "mjtByte",
}
FLOAT_TYPES = {"float", "double", "mjtNum"}
BOOL_TYPES = {"bool", "mjtBool"}

SPECIAL_MEMBERS = {
    "MjSpec": {
        "from_file": [
            "    @staticmethod",
            "    def from_file(filename: str, include: Any = ..., assets: Any = ...) -> MjSpec: ...",
        ],
        "from_string": [
            "    @staticmethod",
            "    def from_string(xml: str, include: Any = ..., assets: Any = ...) -> MjSpec: ...",
        ],
        "from_zip": [
            "    @staticmethod",
            "    def from_zip(file: str | IO[bytes]) -> MjSpec: ...",
        ],
        "compile": [
            "    def compile(self) -> MjModel: ...",
        ],
        "to_xml": [
            "    def to_xml(self) -> str: ...",
        ],
        "body": [
            "    def body(self, name: str) -> MjsBody | None: ...",
        ],
        "joint": [
            "    def joint(self, name: str) -> MjsJoint | None: ...",
        ],
        "actuator": [
            "    def actuator(self, name: str) -> MjsActuator | None: ...",
        ],
        "geom": [
            "    def geom(self, name: str) -> MjsGeom | None: ...",
        ],
        "site": [
            "    def site(self, name: str) -> MjsSite | None: ...",
        ],
        "frame": [
            "    def frame(self, name: str) -> MjsFrame | None: ...",
        ],
        "camera": [
            "    def camera(self, name: str) -> MjsCamera | None: ...",
        ],
        "light": [
            "    def light(self, name: str) -> MjsLight | None: ...",
        ],
        "material": [
            "    def material(self, name: str) -> MjsMaterial | None: ...",
        ],
        "mesh": [
            "    def mesh(self, name: str) -> MjsMesh | None: ...",
        ],
        "delete": [
            "    def delete(self, obj: Any) -> None: ...",
        ],
    },
    "MjData": {
        "__init__": [
            "    def __init__(self, model: MjModel) -> None: ...",
        ],
    },
}


def safe_name(name: str) -> str:
    name = name or "arg"
    name = name.replace("-", "_")
    if not name.isidentifier() or keyword.iskeyword(name):
        name = f"{name}_"
    return name


def unique_name(name: str, used: set[str]) -> str:
    name = safe_name(name)
    if name not in used:
        used.add(name)
        return name
    index = 2
    while f"{name}_{index}" in used:
        index += 1
    value = f"{name}_{index}"
    used.add(value)
    return value


def type_name(value: str) -> str:
    if value in STRUCT_NAME_MAP:
        return STRUCT_NAME_MAP[value]
    if value in ENUM_NAMES:
        return value
    return "Any"


def render_type(node: Any, *, is_return: bool = False) -> str:
    if isinstance(node, ValueType):
        name = node.name
        if name == "void":
            return "None"
        if name in BOOL_TYPES:
            return "bool"
        if name in FLOAT_TYPES:
            return "float"
        if name in INT_TYPES:
            return "int"
        if name in STRUCT_NAME_MAP:
            return STRUCT_NAME_MAP[name]
        if name in ENUM_NAMES:
            return name
        if name.startswith("mjt"):
            return "int"
        return "Any"

    if isinstance(node, PointerType):
        inner = node.inner_type
        if isinstance(inner, ValueType) and inner.name == "char":
            return "str | None" if is_return else "str | bytes"
        if isinstance(inner, ValueType) and inner.name in STRUCT_NAME_MAP:
            return STRUCT_NAME_MAP[inner.name]
        if isinstance(inner, ValueType) and inner.name in ENUM_NAMES:
            return inner.name
        return "Any"

    if isinstance(node, ArrayType):
        return "Any"

    if isinstance(node, (AnonymousStructDecl, AnonymousUnionDecl)):
        return "Any"

    return "Any"


def render_enum(name: str) -> list[str]:
    enum_decl = ENUMS[name]
    lines = [f"class {name}(enum.IntEnum):"]
    for key, value in enum_decl.values.items():
        lines.append(f"    {safe_name(key)} = {value}")
    if len(lines) == 1:
        lines.append("    ...")
    return lines


def render_struct_fields(py_name: str) -> list[str]:
    c_name = next((key for key, value in STRUCT_NAME_MAP.items() if value == py_name), None)
    if c_name is None:
        return []
    lines: list[str] = []
    for field in STRUCTS[c_name].fields:
        if not hasattr(field, "name"):
            continue
        lines.append(f"    {safe_name(field.name)}: {render_type(field.type)}")
    return lines


def render_runtime_class(name: str) -> list[str]:
    obj = getattr(mujoco, name)
    base = "(Exception)" if inspect.isclass(obj) and issubclass(obj, Exception) else ""
    special = SPECIAL_MEMBERS.get(name, {})
    init_lines = special.get("__init__", ["    def __init__(self, *args: Any, **kwargs: Any) -> None: ..."])
    lines = [f"class {name}{base}:", *init_lines]
    fields = render_struct_fields(name)
    lines.extend(fields)

    declared = {line.split(":", 1)[0].strip() for line in fields}
    for member, member_lines in special.items():
        if member == "__init__":
            continue
        lines.extend(member_lines)
        declared.add(member)
    cls = obj
    for member in sorted(attr for attr in dir(cls) if not attr.startswith("_")):
        if safe_name(member) in declared:
            continue
        try:
            raw = inspect.getattr_static(cls, member)
        except Exception:
            raw = None
        label = safe_name(member)
        if isinstance(raw, property):
            lines.append(f"    {label}: Any")
            continue
        if isinstance(raw, staticmethod):
            lines.append("    @staticmethod")
            lines.append(f"    def {label}(*args: Any, **kwargs: Any) -> Any: ...")
            continue
        if isinstance(raw, classmethod):
            lines.append("    @classmethod")
            lines.append(f"    def {label}(cls, *args: Any, **kwargs: Any) -> Any: ...")
            continue
        value = getattr(cls, member, None)
        if callable(value) or type(raw).__name__ == "instancemethod":
            lines.append(f"    def {label}(self, *args: Any, **kwargs: Any) -> Any: ...")
        else:
            lines.append(f"    {label}: Any")
    return lines


def render_function(name: str) -> list[str]:
    decl = FUNCTIONS[name]
    used: set[str] = set()
    params = []
    for parameter in decl.parameters:
        param_name = unique_name(parameter.name, used)
        params.append(f"{param_name}: {render_type(parameter.type)}")
    return_type = render_type(decl.return_type, is_return=True)
    return [f"def {name}({', '.join(params)}) -> {return_type}: ..."]


def render_constant(name: str, value: Any) -> str:
    if isinstance(value, bool):
        return f"{name}: Final[bool]"
    if isinstance(value, int):
        return f"{name}: Final[int]"
    if isinstance(value, float):
        return f"{name}: Final[float]"
    if isinstance(value, str):
        return f"{name}: Final[str]"
    if isinstance(value, bytes):
        return f"{name}: Final[bytes]"
    return f"{name}: Final[Any]"


def main() -> None:
    lines = [
        "from __future__ import annotations",
        "",
        "# Generated by scripts/gen_mujoco_stubs.py",
        "",
        "import enum",
        "from typing import Any, Final, IO, TypeAlias",
        "",
        "class FatalError(Exception): ...",
        "class Renderer:",
        "    def __init__(self, *args: Any, **kwargs: Any) -> None: ...",
        "class GLContext:",
        "    def __init__(self, *args: Any, **kwargs: Any) -> None: ...",
        "",
    ]

    for name in sorted(ENUM_NAMES):
        lines.extend(render_enum(name))
        lines.append("")

    runtime_classes = [
        name
        for name in sorted(RUNTIME_NAMES)
        if name.startswith(("Mj", "Mjs"))
        and name not in {"MjStruct"}
        and inspect.isclass(getattr(mujoco, name))
    ]
    for name in runtime_classes:
        lines.extend(render_runtime_class(name))
        lines.append("")

    lines.extend(
        [
            "MjStruct: TypeAlias = Any",
            "",
            "def to_zip(spec: MjSpec, file: str | IO[bytes]) -> None: ...",
            "def from_zip(file: str | IO[bytes]) -> MjSpec: ...",
            "",
        ]
    )

    runtime_functions = sorted(name for name in FUNCTIONS if name in RUNTIME_NAMES)
    for name in runtime_functions:
        lines.extend(render_function(name))
    lines.append("")

    reserved = set(ENUM_NAMES) | set(runtime_classes) | set(runtime_functions) | {
        "Any",
        "FatalError",
        "Renderer",
        "GLContext",
        "MjStruct",
        "to_zip",
        "from_zip",
    }
    constants = [
        name
        for name in sorted(RUNTIME_NAMES)
        if name not in reserved and not name.startswith("_") and not inspect.ismodule(getattr(mujoco, name))
    ]
    for name in constants:
        value = getattr(mujoco, name)
        if inspect.isclass(value) or callable(value):
            continue
        lines.append(render_constant(name, value))

    lines.extend(
        [
            "",
            "def __getattr__(name: str) -> Any: ...",
            "",
        ]
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
