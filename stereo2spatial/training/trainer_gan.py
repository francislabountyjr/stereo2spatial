"""GAN/discriminator setup helpers for the training loop."""

from __future__ import annotations

from pathlib import Path

import torch
from accelerate import Accelerator

from .checkpointing import (
    _load_discriminator_weights_if_available,
    _load_gan_state_if_available,
)
from .config import TrainConfig
from .discriminator import MultiScaleDiscriminator, set_requires_grad
from .trainer_settings import TrainerRuntimeSettings


def _prepare_discriminator_components(
    *,
    accelerator: Accelerator,
    config: TrainConfig,
    settings: TrainerRuntimeSettings,
    resumed_ckpt: Path | None,
    init_checkpoint_path: Path | None,
) -> tuple[torch.nn.Module | None, torch.optim.Optimizer | None]:
    """
    Build discriminator and optimizer and restore checkpoint state when available.

    Returns `(None, None)` when GAN mode is disabled.
    """
    if not settings.use_gan:
        return None, None

    discriminator_in_channels = (
        int(config.model.cond_channels)
        + int(config.model.target_channels)
        + (1 if settings.gan_use_mask_channel else 0)
    )
    discriminator = MultiScaleDiscriminator(
        in_channels=discriminator_in_channels,
        base_channels=settings.gan_d_base_channels,
        fine_layers=settings.gan_d_fine_layers,
        coarse_layers=settings.gan_d_coarse_layers,
        use_spectral_norm=settings.gan_d_use_spectral_norm,
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=settings.gan_d_lr,
        betas=(settings.gan_d_beta1, settings.gan_d_beta2),
    )
    discriminator, discriminator_optimizer = accelerator.prepare(
        discriminator, discriminator_optimizer
    )
    discriminator.train()

    loaded_gan_state = False
    loaded_init_discriminator = False
    if resumed_ckpt is not None:
        loaded_gan_state = _load_gan_state_if_available(
            checkpoint_path=resumed_ckpt,
            discriminator=discriminator,
            discriminator_optimizer=discriminator_optimizer,
            accelerator=accelerator,
        )
    elif init_checkpoint_path is not None:
        loaded_init_discriminator = _load_discriminator_weights_if_available(
            checkpoint_path=init_checkpoint_path,
            discriminator=discriminator,
            accelerator=accelerator,
        )
    if accelerator.is_main_process:
        if resumed_ckpt is not None and loaded_gan_state:
            print(
                "Loaded discriminator + optimizer state from checkpoint: "
                f"{resumed_ckpt}"
            )
        elif resumed_ckpt is not None:
            print(
                "No discriminator state found in resumed checkpoint; "
                "starting discriminator from scratch."
            )
        elif init_checkpoint_path is not None and loaded_init_discriminator:
            print(
                "Initialized discriminator weights from checkpoint: "
                f"{init_checkpoint_path}"
            )
        elif init_checkpoint_path is not None:
            print(
                "No discriminator weights found in init checkpoint; "
                "starting discriminator from scratch."
            )
    set_requires_grad(discriminator, False)

    return discriminator, discriminator_optimizer
