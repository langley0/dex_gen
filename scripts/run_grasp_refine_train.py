#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from grasp_refine import (
    DiffusionConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    PreparedDatasetConfig,
    TrainingConfig,
    train_grasp_diffusion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal DGA-style grasp diffusion model from a prepared dataset.")
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared normalized DGA dataset (.npz).")
    parser.add_argument("--architecture", choices=("mlp", "dga_unet", "dga_transformer"), default="dga_unet")
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--split-mode", choices=("sample", "object", "object_random", "object_fixed"), default="sample")
    parser.add_argument("--train-object-key", action="append", default=None)
    parser.add_argument("--val-object-key", action="append", default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--log-step", type=int, default=100)
    parser.add_argument("--run-validation", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--save-model-interval", type=int, default=1)
    parser.add_argument("--save-model-separately", action="store_true")
    parser.add_argument("--save-scene-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--distributed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diffusion-steps", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--context-dim", type=int, default=512)
    parser.add_argument("--context-tokens", type=int, default=8)
    parser.add_argument("--scene-encoder-layers", type=int, default=1)
    parser.add_argument("--denoiser-blocks", type=int, default=4)
    parser.add_argument("--transformer-depth", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--transformer-dim-head", type=int, default=64)
    parser.add_argument("--resblock-dropout", type=float, default=0.0)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--scene-encoder-pretrained", type=Path, default=None)
    parser.add_argument("--freeze-scene-encoder", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        dataset=PreparedDatasetConfig(
            path=args.dataset.expanduser().resolve(),
            train_fraction=float(args.train_fraction),
            split_mode=args.split_mode,
            train_object_keys=tuple(args.train_object_key or ()),
            val_object_keys=tuple(args.val_object_key or ()),
        ),
        model=ModelConfig(
            architecture=args.architecture,
            hidden_dim=int(args.hidden_dim),
            context_dim=int(args.context_dim),
            context_tokens=int(args.context_tokens),
            scene_encoder_layers=int(args.scene_encoder_layers),
            denoiser_blocks=int(args.denoiser_blocks),
            transformer_depth=int(args.transformer_depth),
            num_heads=int(args.num_heads),
            resblock_dropout=float(args.resblock_dropout),
            transformer_dropout=float(args.transformer_dropout),
            transformer_dim_head=int(args.transformer_dim_head),
            scene_encoder_pretrained=None if args.scene_encoder_pretrained is None else args.scene_encoder_pretrained.expanduser().resolve(),
            freeze_scene_encoder=bool(args.freeze_scene_encoder),
        ),
        diffusion=DiffusionConfig(steps=int(args.diffusion_steps)),
        loss=LossConfig(),
        optimizer=OptimizerConfig(
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            grad_clip_norm=float(args.grad_clip_norm),
        ),
        log_step=int(args.log_step),
        run_validation=bool(args.run_validation),
        checkpoint_dir=None if args.checkpoint_dir is None else args.checkpoint_dir.expanduser().resolve(),
        save_model_interval=int(args.save_model_interval),
        save_model_separately=bool(args.save_model_separately),
        save_scene_model=bool(args.save_scene_model),
        metrics_path=None if args.metrics_path is None else args.metrics_path.expanduser().resolve(),
        distributed=bool(args.distributed),
        device=args.device,
        seed=int(args.seed),
    )

    result = train_grasp_diffusion(config)
    print(f"architecture       : {result.resolved_model_config.architecture}")
    print(f"distributed        : {result.distributed}")
    print(f"device count       : {result.device_count}")
    print(f"pose dim           : {result.resolved_model_config.pose_dim}")
    print(f"epochs             : {len(result.history)}")
    print(f"final step         : {result.final_step}")
    if result.history:
        last = result.history[-1]
        print(f"last epoch         : {last.epoch}")
        print(f"train loss         : {last.train_loss:.6f}")
        print(f"noise loss         : {last.train_noise_loss:.6f}")
        print(f"erf loss           : {last.train_erf_loss:.6f}")
        print(f"spf loss           : {last.train_spf_loss:.6f}")
        print(f"srf loss           : {last.train_srf_loss:.6f}")
        if last.val_loss is not None:
            print(f"val loss           : {last.val_loss:.6f}")


if __name__ == "__main__":
    main()
