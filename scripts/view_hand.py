#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import jax
import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_gen.hand import Hand, InitConfig, Pose, PoseBatch
from grasp_gen.hand_contacts import ContactConfig, ContactRecord


TARGET = np.zeros(3, dtype=float)


def _cam(cam: mujoco.MjvCamera) -> None:
    cam.lookat[:] = TARGET
    cam.distance = 0.72
    cam.azimuth = 145.0
    cam.elevation = -18.0


def _add_marker(scene, idx: int, pos: np.ndarray, radius: float, rgba: np.ndarray) -> int:
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


def _min_pairwise_distance(points: np.ndarray) -> float | None:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return None
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    dist += np.eye(len(points), dtype=float) * 1.0e6
    return float(np.min(dist))


def _overlay(
    viewer,
    batch: PoseBatch,
    current: int,
    pose: Pose,
    contacts: list[ContactRecord],
) -> None:
    scene = viewer.user_scn
    idx = 0
    idx = _add_marker(scene, idx, TARGET, 0.015, np.array([1.0, 0.2, 0.2, 1.0], dtype=float))
    for batch_index, palm_pos in enumerate(batch.palm_pos):
        color = np.array([0.28, 0.92, 0.50, 0.32], dtype=float)
        radius = 0.0045
        if batch_index == current:
            color = np.array([1.0, 0.92, 0.25, 1.0], dtype=float)
            radius = 0.008
        idx = _add_marker(scene, idx, palm_pos, radius, color)
    idx = _add_marker(scene, idx, pose.root_pos, 0.006, np.array([1.0, 1.0, 1.0, 1.0], dtype=float))
    for record in contacts:
        color = np.array([1.0, 0.7, 0.2, 1.0], dtype=float) if record.finger == "thumb" else np.array(
            [0.2, 0.85, 1.0, 0.95],
            dtype=float,
        )
        idx = _add_marker(scene, idx, record.world_pos, 0.006, color)
    scene.ngeom = idx


def _pose_text(hand: Hand, pose: Pose, index: int) -> str:
    root_pos, root_rpy = hand.root_6dof(pose)
    lines = [
        f"active index   : {index} score={pose.score:.6f} roll={pose.roll_deg:.1f}deg",
        (
            "root 6dof      : "
            f"xyz=[{root_pos[0]: .4f}, {root_pos[1]: .4f}, {root_pos[2]: .4f}] "
            f"rpy_deg=[{root_rpy[0]: .1f}, {root_rpy[1]: .1f}, {root_rpy[2]: .1f}]"
        ),
        "joints         :",
    ]
    for joint_name, rad, deg in hand.joint_values(pose):
        lines.append(f"  {joint_name}: {rad: .4f} rad ({deg: .1f} deg)")
    return "\n".join(lines)


def run(
    hand: Hand,
    batch: PoseBatch,
    *,
    start_index: int,
    interval: float,
    contact_cfg: ContactConfig,
) -> None:
    model, data = hand.mj()
    current = int(start_index) % len(batch)
    pose = hand.apply_batch(batch, current)
    contacts = hand.contacts(cfg=contact_cfg)
    next_switch = time.monotonic() + interval
    print(_pose_text(hand, pose, current), flush=True)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        _cam(viewer.cam)
        while viewer.is_running():
            now = time.monotonic()
            if now >= next_switch:
                current = (current + 1) % len(batch)
                pose = hand.apply_batch(batch, current)
                contacts = hand.contacts(cfg=contact_cfg)
                print(_pose_text(hand, pose, current), flush=True)
                next_switch = now + interval
            _overlay(viewer, batch, current, pose, contacts)
            viewer.sync()
            time.sleep(1.0 / 60.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render batched grasp_gen hand poses and cycle them in the MuJoCo viewer.")
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument("--points", type=int, default=10, help="Contact points per segment used for batch init poses.")
    parser.add_argument("--batch", type=int, default=64, help="Number of init poses generated together with JAX.")
    parser.add_argument("--index", type=int, default=0, help="Starting batch index.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offset", type=float, default=0.30, help="Target-to-palm distance.")
    parser.add_argument("--interval", type=float, default=3.0, help="Seconds between pose switches.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch <= 0:
        raise SystemExit("--batch must be positive.")
    if args.interval <= 0.0:
        raise SystemExit("--interval must be positive.")

    hand = Hand(args.hand)
    init_cfg = InitConfig(n_per_seg=args.points, palm_offset=args.offset)
    batch = hand.init_batch(args.batch, cfg=init_cfg, seed=args.seed)
    start_index = int(args.index)
    if not 0 <= start_index < len(batch):
        raise SystemExit(f"--index must be in [0, {len(batch) - 1}] for batch size {len(batch)}.")

    pose = hand.apply_batch(batch, start_index)
    contact_cfg = ContactConfig(
        n_per_seg=init_cfg.n_per_seg,
        thumb_weight=init_cfg.thumb_weight,
        palm_clearance=init_cfg.palm_clearance,
    )
    contacts = hand.contacts(cfg=contact_cfg)
    body_counts = Counter(record.body_name for record in contacts)
    weights = np.asarray([record.weight for record in contacts], dtype=float)
    min_gap = _min_pairwise_distance(batch.palm_pos)

    print(f"Hand side      : {args.hand}")
    print(f"jax backend    : {jax.default_backend()}")
    print(f"batch size     : {len(batch)}")
    print(f"switch every   : {args.interval:.1f} s")
    print(f"qpos / ctrl    : {hand.model.nq} / {hand.model.nu}")
    print(f"contact points : {len(contacts)}")
    print(f"per segment    : {args.points}")
    print(f"target         : [{TARGET[0]:.3f}, {TARGET[1]:.3f}, {TARGET[2]:.3f}]")
    print(f"palm offset    : {args.offset:.3f}")
    print(f"start index    : {start_index}")
    print(f"score range    : [{float(batch.score.min()):.6f}, {float(batch.score.max()):.6f}]")
    print(f"roll range     : [{float(batch.roll_deg.min()):.1f}, {float(batch.roll_deg.max()):.1f}]")
    if min_gap is not None:
        print(f"palm spread    : min pairwise {min_gap:.4f} m")
    print(f"weight range   : [{float(weights.min()):.3f}, {float(weights.max()):.3f}]")
    print(f"root pos       : [{pose.root_pos[0]:.3f}, {pose.root_pos[1]:.3f}, {pose.root_pos[2]:.3f}]")
    print(
        f"root quat      : [{pose.root_quat[0]:.3f}, {pose.root_quat[1]:.3f}, "
        f"{pose.root_quat[2]:.3f}, {pose.root_quat[3]:.3f}]"
    )
    print(f"reach dir      : [{pose.reach_dir[0]:.3f}, {pose.reach_dir[1]:.3f}, {pose.reach_dir[2]:.3f}]")
    print("point bodies   :")
    for body_name, count in body_counts.items():
        print(f"  {body_name}: {count}")
    print("viewer colors  : red=target, green=batch palms, yellow=current palm, white=current root")
    print("viewer colors  : orange=thumb contact candidates, cyan=other candidates")
    print("Viewer prints the active batch index, root 6DoF (xyz+rpy), and joint values every switch.")

    run(hand, batch, start_index=start_index, interval=args.interval, contact_cfg=contact_cfg)


if __name__ == "__main__":
    main()
