#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from mesh_gen import build_asset, load_asset_generation_config


DEFAULT_CONFIG_PATH = ROOT / "configs" / "mesh_gen" / "procedural_assets.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate procedural mesh assets into assets/generated.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Asset-generation config TOML path.")
    parser.add_argument(
        "--name",
        dest="names",
        action="append",
        default=None,
        help="Generate only the given asset name. Repeatable.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the assets that would be generated.")
    return parser.parse_args()


def _select_assets(names: list[str] | None, assets):
    if not names:
        return assets
    requested = {name.strip() for name in names if name.strip()}
    selected = tuple(asset for asset in assets if asset.name in requested)
    missing = sorted(requested.difference(asset.name for asset in selected))
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Unknown asset names: {joined}")
    return selected


def main() -> None:
    args = parse_args()
    asset_root, assets = load_asset_generation_config(args.config)
    selected_assets = _select_assets(args.names, assets)

    print(f"config path        : {args.config.expanduser().resolve()}")
    print(f"asset root         : {asset_root}")
    print(f"asset count        : {len(selected_assets)}")
    print(f"dry run            : {args.dry_run}")
    print("assets             :")
    for asset in selected_assets:
        info = {
            "primitive": asset.primitive,
            "name": asset.name,
        }
        if asset.size_xyz is not None:
            info["size_xyz"] = list(asset.size_xyz)
        if asset.radius is not None:
            info["radius"] = asset.radius
        if asset.half_height is not None:
            info["half_height"] = asset.half_height
        if asset.sides is not None:
            info["sides"] = asset.sides
        if asset.radii_xyz is not None:
            info["radii_xyz"] = list(asset.radii_xyz)
        print(f"  - {json.dumps(info, ensure_ascii=True)}")

    if args.dry_run:
        return

    asset_root.mkdir(parents=True, exist_ok=True)
    for asset in selected_assets:
        metadata_path = build_asset(asset_root, asset)
        print(f"[OK] {asset.name} -> {metadata_path}")


if __name__ == "__main__":
    main()
