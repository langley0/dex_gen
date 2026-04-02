#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_refine import DiffusionConfig, DpmSolverConfig, GuidanceConfig, ModelConfig, first_batch, load_latest_checkpoint_state, sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DDPM sampling from a trained grasp_refine checkpoint.")
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared normalized DGA dataset (.npz).")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Checkpoint directory.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diffusion-steps", type=int, default=4)
    parser.add_argument("--use-dpmsolver", action="store_true")
    parser.add_argument("--dpm-steps", type=int, default=10)
    parser.add_argument("--dpm-order", type=int, default=1)
    parser.add_argument("--dpm-skip-type", type=str, default="time_uniform")
    parser.add_argument("--dpm-t-start", type=float, default=1.0)
    parser.add_argument("--dpm-t-end", type=float, default=0.01)
    parser.add_argument("--dpm-method", type=str, default="singlestep")
    parser.add_argument("--dpm-lower-order-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-guidance", action="store_true")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--grad-scale", type=float, default=0.1)
    parser.add_argument("--clip-grad-min", type=float, default=-0.1)
    parser.add_argument("--clip-grad-max", type=float, default=0.1)
    parser.add_argument("--guidance-erf-weight", type=float, default=0.3)
    parser.add_argument("--guidance-spf-weight", type=float, default=1.0)
    parser.add_argument("--guidance-srf-weight", type=float, default=1.0)
    parser.add_argument("--opt-interval", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=512)
    parser.add_argument("--context-tokens", type=int, default=8)
    parser.add_argument("--scene-encoder-layers", type=int, default=1)
    parser.add_argument("--denoiser-blocks", type=int, default=1)
    parser.add_argument("--transformer-depth", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--transformer-dim-head", type=int, default=8)
    parser.add_argument("--resblock-dropout", type=float, default=0.0)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, batch = first_batch(args.dataset, batch_size=int(args.batch_size))
    checkpoint_state = load_latest_checkpoint_state(args.checkpoint_dir)
    model_config = ModelConfig(
        architecture="dga_unet",
        pose_dim=int(dataset.arrays.pose.shape[1]),
        hidden_dim=int(args.hidden_dim),
        context_dim=int(args.context_dim),
        context_tokens=int(args.context_tokens),
        scene_encoder_layers=int(args.scene_encoder_layers),
        denoiser_blocks=int(args.denoiser_blocks),
        transformer_depth=int(args.transformer_depth),
        num_heads=int(args.num_heads),
        transformer_dim_head=int(args.transformer_dim_head),
        resblock_dropout=float(args.resblock_dropout),
        transformer_dropout=float(args.transformer_dropout),
    )
    output = sample(
        checkpoint_state.params,
        batch,
        dataset=dataset,
        model_config=model_config,
        diffusion_config=DiffusionConfig(steps=int(args.diffusion_steps)),
        rng_key=jax.random.key(np.uint32(int(args.seed) % (2**32))),
        k=int(args.samples),
        return_trajectory=True,
        dpm_config=DpmSolverConfig(
            use_dpmsolver=bool(args.use_dpmsolver),
            steps=int(args.dpm_steps),
            order=int(args.dpm_order),
            skip_type=str(args.dpm_skip_type),
            t_start=float(args.dpm_t_start),
            t_end=float(args.dpm_t_end),
            method=str(args.dpm_method),
            lower_order_final=bool(args.dpm_lower_order_final),
        ),
        guidance_config=GuidanceConfig(
            enabled=bool(args.use_guidance),
            guidance_scale=float(args.guidance_scale),
            grad_scale=float(args.grad_scale),
            clip_grad_min=float(args.clip_grad_min),
            clip_grad_max=float(args.clip_grad_max),
            erf_weight=float(args.guidance_erf_weight),
            spf_weight=float(args.guidance_spf_weight),
            srf_weight=float(args.guidance_srf_weight),
            opt_interval=int(args.opt_interval),
        ),
    )

    print(f"dataset path          : {dataset.path}")
    print(f"checkpoint epoch      : {checkpoint_state.epoch}")
    print(f"checkpoint step       : {checkpoint_state.step}")
    print(f"sample shape          : {output.samples.shape}")
    print(f"sample full shape     : {output.samples_full.shape}")
    if output.trajectory is not None:
        print(f"trajectory shape      : {output.trajectory.shape}")
    print(f"sample min/max        : {float(output.samples.min()):.6f} / {float(output.samples.max()):.6f}")

    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "samples": output.samples.astype(np.float32),
            "samples_full": output.samples_full.astype(np.float32),
        }
        if output.trajectory is not None:
            payload["trajectory"] = output.trajectory.astype(np.float32)
        np.savez_compressed(output_path, **payload)
        print(f"saved output          : {output_path}")


if __name__ == "__main__":
    main()
