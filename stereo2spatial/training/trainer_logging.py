"""Training setup logging helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from .config import TrainConfig
from .dataset import LatentSongDataset
from .trainer_settings import TrainerRuntimeSettings


def log_training_setup(
    *,
    accelerator: Accelerator,
    config: TrainConfig,
    dataset: LatentSongDataset,
    dataloader: DataLoader,
    validation_dataloader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    settings: TrainerRuntimeSettings,
    compile_requested: bool,
    compile_mode: str,
    model_was_compiled: bool,
    resumed_checkpoint: Path | None,
    init_checkpoint: Path | None,
    discriminator_optimizer: torch.optim.Optimizer | None,
) -> None:
    """Emit a single startup summary from rank 0."""
    if not accelerator.is_main_process:
        return

    print("Training setup:")
    print(f"  - num_processes={accelerator.num_processes}")
    print(f"  - device={accelerator.device}")
    print(f"  - mixed_precision={config.training.mixed_precision}")
    print(
        f"  - optimizer={config.optimizer.type} "
        f"(resolved={optimizer.__class__.__name__})"
    )
    print(
        f"  - compile_model={compile_requested} "
        f"mode={compile_mode} compiled={model_was_compiled}"
    )
    print(f"  - resumed_checkpoint={resumed_checkpoint}")
    print(f"  - init_checkpoint={init_checkpoint}")
    print(
        "  - sequence_mode="
        f"{getattr(config.training, 'sequence_mode', 'strided_crops')}"
    )
    print("  - tbptt_windows=" f"{int(getattr(config.training, 'tbptt_windows', 0))}")
    print(
        "  - full_song_max_seconds="
        f"{getattr(config.training, 'full_song_max_seconds', None)}"
    )
    print(f"  - use_ema={bool(getattr(config.training, 'use_ema', False))}")
    if bool(getattr(config.training, "use_ema", False)):
        print(f"  - ema_decay={float(getattr(config.training, 'ema_decay', 0.999))}")
        print(
            "  - ema_device="
            f"{str(getattr(config.training, 'ema_device', 'accelerator'))}"
        )
        print(
            "  - ema_cpu_only="
            f"{bool(getattr(config.training, 'ema_cpu_only', False))}"
        )
    print(f"  - use_gan={settings.use_gan}")
    if settings.use_gan:
        print(f"  - gan_d_lr={settings.gan_d_lr}")
        print(f"  - gan_d_betas=({settings.gan_d_beta1}, {settings.gan_d_beta2})")
        print(f"  - gan_d_base_channels={settings.gan_d_base_channels}")
        print(f"  - gan_d_num_layers={settings.gan_d_num_layers}")
        print(f"  - gan_d_fine_layers={settings.gan_d_fine_layers}")
        print(f"  - gan_d_coarse_layers={settings.gan_d_coarse_layers}")
        print(f"  - gan_use_mask_channel={settings.gan_use_mask_channel}")
        print(f"  - gan_ms_w_fine={settings.gan_ms_w_fine}")
        print(f"  - gan_ms_w_coarse={settings.gan_ms_w_coarse}")
        print(f"  - gan_lambda_adv={settings.gan_lambda_adv_max}")
        print(f"  - gan_adv_warmup_steps={settings.gan_adv_warmup_steps}")
        print(f"  - gan_r1_gamma={settings.gan_r1_gamma}")
        print(f"  - gan_r1_every={settings.gan_r1_every}")
        print("  - discriminator_optimizer_type=" f"{type(discriminator_optimizer)}")

    print(f"  - routing_kl_weight={settings.routing_kl_weight}")
    print(f"  - routing_kl_temperature={settings.routing_kl_temperature}")
    print(f"  - routing_kl_eps={settings.routing_kl_eps}")
    print(f"  - corr_weight={settings.corr_weight}")
    print(f"  - corr_eps={settings.corr_eps}")
    print(f"  - corr_offdiag_only={settings.corr_offdiag_only}")
    print(f"  - corr_use_correlation={settings.corr_use_correlation}")

    if isinstance(optimizer, torch.optim.AdamW):
        print(f"  - adamw_fused={bool(optimizer.defaults.get('fused', False))}")
        print(f"  - adamw_foreach={bool(optimizer.defaults.get('foreach', False))}")

    print(f"  - segment_seconds={config.data.segment_seconds}")
    print(f"  - resolved_latent_fps={dataset.resolved_latent_fps:.6f}")
    print(f"  - dataset_sequence_seconds={dataset.sequence_seconds}")
    print(f"  - dataset_sequence_frames={dataset.sequence_frames}")
    print(f"  - stride_seconds={dataset.stride_seconds}")
    print(f"  - stride_frames={dataset.stride_frames}")
    print(f"  - batch_size_per_process={config.data.batch_size}")
    print(
        "  - dataloader_prefetch_factor="
        f"{config.data.prefetch_factor if config.data.num_workers > 0 else 'n/a(num_workers=0)'}"
    )
    print(
        "  - dataloader_persistent_workers="
        f"{getattr(dataloader, 'persistent_workers', 'wrapped')}"
    )
    print(
        "  - effective_global_batch="
        f"{config.data.batch_size * accelerator.num_processes * config.training.grad_accum_steps}"
    )
    print(f"  - max_steps={config.training.max_steps}")
    print(f"  - run_validation={settings.run_validation}")
    if settings.run_validation:
        print(f"  - validation_steps={settings.validation_steps}")
        print(f"  - validation_dataset_root={config.training.validation_dataset_root}")
        print(f"  - validation_dataset_path={config.training.validation_dataset_path}")
        if validation_dataloader is not None:
            print(f"  - validation_batches={len(validation_dataloader)}")

    print(f"  - run_validation_generations={settings.run_validation_generations}")
    if settings.run_validation_generations:
        print(
            f"  - validation_generation_input_path={config.training.validation_generation_input_path}"
        )
        print(
            f"  - validation_generation_output_path={config.training.validation_generation_output_path}"
        )
        print(f"  - num_valid_generations={config.training.num_valid_generations}")
        print(
            f"  - validation_generation_seed={config.training.validation_generation_seed}"
        )
