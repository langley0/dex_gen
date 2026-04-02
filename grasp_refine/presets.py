from __future__ import annotations


TRAIN_PRESETS: dict[str, dict[str, object]] = {
    "stable_top256": {
        "architecture": "dga_unet",
        "batch_size": 8,
        "lr": 3.0e-5,
        "grad_clip_norm": 1.0,
        "diffusion_steps": 4,
        "hidden_dim": 64,
        "context_dim": 512,
        "context_tokens": 8,
        "scene_encoder_layers": 1,
        "denoiser_blocks": 1,
        "transformer_depth": 1,
        "num_heads": 8,
        "transformer_dim_head": 8,
        "resblock_dropout": 0.0,
        "transformer_dropout": 0.1,
        "distributed": False,
    },
    "cpu_debug": {
        "architecture": "dga_unet",
        "batch_size": 4,
        "lr": 3.0e-5,
        "grad_clip_norm": 1.0,
        "diffusion_steps": 4,
        "hidden_dim": 64,
        "context_dim": 512,
        "context_tokens": 8,
        "scene_encoder_layers": 1,
        "denoiser_blocks": 1,
        "transformer_depth": 1,
        "num_heads": 8,
        "transformer_dim_head": 8,
        "resblock_dropout": 0.0,
        "transformer_dropout": 0.1,
        "distributed": False,
    },
    "distributed_cpu_debug": {
        "architecture": "dga_unet",
        "batch_size": 4,
        "lr": 3.0e-5,
        "grad_clip_norm": 1.0,
        "diffusion_steps": 4,
        "hidden_dim": 64,
        "context_dim": 512,
        "context_tokens": 8,
        "scene_encoder_layers": 1,
        "denoiser_blocks": 1,
        "transformer_depth": 1,
        "num_heads": 8,
        "transformer_dim_head": 8,
        "resblock_dropout": 0.0,
        "transformer_dropout": 0.1,
        "distributed": True,
    },
}


SAMPLE_PRESETS: dict[str, dict[str, object]] = {
    "stable_ddpm": {
        "diffusion_steps": 4,
        "hidden_dim": 64,
        "context_dim": 512,
        "context_tokens": 8,
        "scene_encoder_layers": 1,
        "denoiser_blocks": 1,
        "transformer_depth": 1,
        "num_heads": 8,
        "transformer_dim_head": 8,
        "resblock_dropout": 0.0,
        "transformer_dropout": 0.1,
        "project_to_valid_range": True,
    },
    "stable_guided_ddpm": {
        "diffusion_steps": 4,
        "hidden_dim": 64,
        "context_dim": 512,
        "context_tokens": 8,
        "scene_encoder_layers": 1,
        "denoiser_blocks": 1,
        "transformer_depth": 1,
        "num_heads": 8,
        "transformer_dim_head": 8,
        "resblock_dropout": 0.0,
        "transformer_dropout": 0.1,
        "use_guidance": True,
        "project_to_valid_range": True,
    },
}


def get_train_preset(name: str) -> dict[str, object]:
    try:
        return dict(TRAIN_PRESETS[name])
    except KeyError as exc:
        raise KeyError(f"Unknown train preset: {name!r}") from exc


def get_sample_preset(name: str) -> dict[str, object]:
    try:
        return dict(SAMPLE_PRESETS[name])
    except KeyError as exc:
        raise KeyError(f"Unknown sample preset: {name!r}") from exc
