#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_gen.grasp_energy import GraspEnergyConfig, GraspEnergyModel
from grasp_gen.grasp_optimizer import GraspBatchOptimizer, GraspBatchOptimizerConfig
from grasp_gen.grasp_optimizer_io import save_grasp_run
from grasp_gen.grasp_profiler import (
    RunProfiler,
    block_grasp_state,
    find_bottleneck,
    format_profile_summary,
    profile_call,
)
from grasp_gen.hand import Hand, InitConfig
from grasp_gen.hand_contacts import ContactConfig
from grasp_gen.prop_assets import make_named_prop


CUBE_SIZE = 0.07


def _make_prop(args: argparse.Namespace) -> tuple[Prop, dict[str, object]]:
    return make_named_prop(args.object, cube_size=float(args.cube_size))


def _default_output_path(batch: int, steps: int, seed: int, object_kind: str) -> Path:
    return ROOT / "outputs" / "grasp_optimizer" / f"run_{object_kind}_b{batch}_s{steps}_seed{seed}_eqwrench_pen.npz"


def _result_stats(state) -> dict[str, float | int]:
    current = np.asarray(state.energy.total, dtype=float)
    best = np.asarray(state.best_energy.total, dtype=float)
    accepted = np.asarray(state.accepted_steps, dtype=np.int32)
    rejected = np.asarray(state.rejected_steps, dtype=np.int32)
    return {
        "step_index": int(np.asarray(state.step_index)),
        "current_energy_min": float(np.min(current)),
        "current_energy_mean": float(np.mean(current)),
        "best_energy_min": float(np.min(best)),
        "best_energy_mean": float(np.mean(best)),
        "best_equilibrium_min": float(np.min(np.asarray(state.best_energy.equilibrium, dtype=float))),
        "best_equilibrium_mean": float(np.mean(np.asarray(state.best_energy.equilibrium, dtype=float))),
        "best_penetration_min": float(np.min(np.asarray(state.best_energy.penetration, dtype=float))),
        "best_penetration_mean": float(np.mean(np.asarray(state.best_energy.penetration, dtype=float))),
        "best_force_min": float(np.min(np.asarray(state.best_energy.force, dtype=float))),
        "best_force_mean": float(np.mean(np.asarray(state.best_energy.force, dtype=float))),
        "best_torque_min": float(np.min(np.asarray(state.best_energy.torque, dtype=float))),
        "best_torque_mean": float(np.mean(np.asarray(state.best_energy.torque, dtype=float))),
        "best_sample_index": int(np.argmin(best)),
        "accepted_steps_mean": float(np.mean(accepted)),
        "rejected_steps_mean": float(np.mean(rejected)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the batched grasp optimizer and save the result to disk.")
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument("--object", choices=("cylinder", "cube", "drill", "decor01"), default="cylinder")
    parser.add_argument("--cube-size", type=float, default=CUBE_SIZE, help="Cube edge length when --object cube.")
    parser.add_argument("--batch", "--envs", dest="batch", type=int, default=64, help="Number of optimizer environments.")
    parser.add_argument("--steps", type=int, default=5000, help="Number of optimizer steps to run.")
    parser.add_argument("--points", type=int, default=10, help="Sampled contact points per segment.")
    parser.add_argument("--contact-count", type=int, default=4, help="Selected contact count per sample.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offset", type=float, default=0.30, help="Initial target-to-palm distance.")
    parser.add_argument("--distance-weight", type=float, default=1.0)
    parser.add_argument("--equilibrium-weight", type=float, default=1.0)
    parser.add_argument("--penetration-weight", type=float, default=100.0)
    parser.add_argument("--wrench-iters", type=int, default=24, help="Projected-gradient iterations for wrench equilibrium.")
    parser.add_argument("--sdf-voxel-size", type=float, default=3.0e-3, help="Object-local SDF voxel size.")
    parser.add_argument("--sdf-padding", type=float, default=1.0e-2, help="Object-local SDF padding.")
    parser.add_argument("--bench-steps", type=int, default=256, help="Extra compiled-loop steps for steady-state timing. Set 0 to disable.")
    parser.add_argument("--output", type=Path, default=None, help="Result .npz path. Defaults to outputs/grasp_optimizer/...")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch <= 0:
        raise SystemExit("--batch must be positive.")
    if args.cube_size <= 0.0:
        raise SystemExit("--cube-size must be positive.")
    if args.steps < 0:
        raise SystemExit("--steps must be non-negative.")
    if args.points <= 0:
        raise SystemExit("--points must be positive.")
    if args.contact_count <= 0:
        raise SystemExit("--contact-count must be positive.")
    if args.bench_steps < 0:
        raise SystemExit("--bench-steps must be non-negative.")
    if args.offset <= 0.0:
        raise SystemExit("--offset must be positive.")
    if args.equilibrium_weight < 0.0:
        raise SystemExit("--equilibrium-weight must be non-negative.")
    if args.penetration_weight < 0.0:
        raise SystemExit("--penetration-weight must be non-negative.")
    if args.wrench_iters <= 0:
        raise SystemExit("--wrench-iters must be positive.")
    if args.sdf_voxel_size <= 0.0:
        raise SystemExit("--sdf-voxel-size must be positive.")
    if args.sdf_padding < 0.0:
        raise SystemExit("--sdf-padding must be non-negative.")

    output_path = (
        _default_output_path(args.batch, args.steps, args.seed, args.object)
        if args.output is None
        else args.output.resolve()
    )
    profiler = RunProfiler()

    with profiler.section("build.hand"):
        hand = Hand(args.hand)
    with profiler.section("build.prop"):
        prop, prop_meta = _make_prop(args)

    init_cfg = InitConfig(n_per_seg=args.points, palm_offset=args.offset)
    contact_cfg = ContactConfig(
        n_per_seg=init_cfg.n_per_seg,
        thumb_weight=init_cfg.thumb_weight,
        palm_clearance=init_cfg.palm_clearance,
        target_spacing=init_cfg.contact_spacing,
        cloud_scale=init_cfg.contact_cloud_scale,
    )
    energy_cfg = GraspEnergyConfig(
        distance_weight=args.distance_weight,
        equilibrium_weight=args.equilibrium_weight,
        penetration_weight=args.penetration_weight,
        wrench_iters=args.wrench_iters,
        sdf_voxel_size=args.sdf_voxel_size,
        sdf_padding=args.sdf_padding,
    )
    optimizer_cfg = GraspBatchOptimizerConfig()

    with profiler.section("build.energy_model"):
        energy_model = GraspEnergyModel(hand, prop, contact_cfg=contact_cfg, config=energy_cfg)
    with profiler.section("build.optimizer"):
        optimizer = GraspBatchOptimizer(energy_model, contact_count=args.contact_count, config=optimizer_cfg)
    with profiler.section("init.batch"):
        pose_batch = hand.init_batch(args.batch, cfg=init_cfg, seed=args.seed)

    state = profile_call(
        profiler,
        "optimizer.init_state",
        lambda: optimizer.init_state(pose_batch.state_vectors(), seed=args.seed),
        sync=block_grasp_state,
    )

    if args.steps > 0:
        state = profile_call(
            profiler,
            "optimizer.first_step",
            lambda: optimizer.step(state)[0],
            sync=block_grasp_state,
        )
        if args.steps > 1:
            state = profile_call(
                profiler,
                "optimizer.bulk_run",
                lambda: optimizer.run_many(state, args.steps - 1),
                sync=block_grasp_state,
            )

    if args.bench_steps > 0:
        profile_call(
            profiler,
            "optimizer.step_bench",
            lambda: optimizer.step(state)[0],
            sync=block_grasp_state,
        )
        block_grasp_state(optimizer.run_many(state, args.bench_steps))
        profile_call(
            profiler,
            "optimizer.steady_bench",
            lambda: optimizer.run_many(state, args.bench_steps),
            sync=block_grasp_state,
        )

    result_stats = _result_stats(state)
    profile_before_save = profiler.summary()
    metadata = {
        "hand": {"side": args.hand},
        "run": {
            "batch": int(args.batch),
            "steps": int(args.steps),
            "seed": int(args.seed),
            "backend": jax.default_backend(),
            "bench_steps": int(args.bench_steps),
        },
        "init": asdict(init_cfg),
        "contact": asdict(contact_cfg),
        "energy": {
            **asdict(energy_cfg),
            "effective_penetration_weight": float(energy_model.effective_penetration_weight),
        },
        "optimizer": asdict(optimizer_cfg),
        "prop": {
            **prop_meta,
        },
        "result": result_stats,
        "profile": profile_before_save,
    }
    output_path = profile_call(
        profiler,
        "save_result",
        lambda: save_grasp_run(output_path, metadata=metadata, state=state),
    )
    profile_summary = profiler.summary()
    bottleneck = find_bottleneck(profile_summary)

    print(f"output path      : {output_path}")
    print(f"jax backend      : {jax.default_backend()}")
    print(f"batch size       : {args.batch}")
    print(f"step count       : {args.steps}")
    print(f"contact count    : {args.contact_count}")
    print(f"candidate count  : {energy_model.point_count}")
    if hasattr(energy_model, "cloud_point_count"):
        print(f"cloud point count: {energy_model.cloud_point_count}")
    if hasattr(energy_model, "effective_penetration_weight"):
        print(f"effective pen wt : {energy_model.effective_penetration_weight:.6f}")
    print(f"cloud scale      : {init_cfg.contact_cloud_scale:.5f}")
    print(f"object kind      : {args.object}")
    print("equilibrium mode : wrench")
    print(f"best energy mean : {result_stats['best_energy_mean']:.6f}")
    print(f"best energy min  : {result_stats['best_energy_min']:.6f}")
    print(f"best eq mean     : {result_stats['best_equilibrium_mean']:.6f}")
    print(f"best eq min      : {result_stats['best_equilibrium_min']:.6f}")
    print(f"best pen mean    : {result_stats['best_penetration_mean']:.6f}")
    print(f"best pen min     : {result_stats['best_penetration_min']:.6f}")
    print(f"best force mean  : {result_stats['best_force_mean']:.6f}")
    print(f"best torque mean : {result_stats['best_torque_mean']:.6f}")
    print(f"best sample idx  : {result_stats['best_sample_index']}")
    if args.bench_steps > 0 and "optimizer.steady_bench" in profile_summary:
        steady = profile_summary["optimizer.steady_bench"]
        steps_per_s = args.bench_steps / max(float(steady["total_s"]), 1.0e-12)
        print(f"steady steps/s   : {steps_per_s:.2f}")
    if bottleneck is not None:
        name, stats = bottleneck
        print(f"bottleneck       : {name} ({float(stats['total_s']):.3f}s total)")
    print("profile summary  :")
    print(format_profile_summary(profile_summary))


if __name__ == "__main__":
    main()
