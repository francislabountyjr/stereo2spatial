"""Typed runtime settings derived from ``TrainConfig`` for the trainer."""

from __future__ import annotations

from dataclasses import dataclass

from .config import TrainConfig


@dataclass(frozen=True)
class TrainerRuntimeSettings:
    """Materialized trainer toggles and coefficients resolved from config defaults."""

    run_validation: bool
    run_validation_generations: bool
    validation_steps: int
    use_gan: bool
    gan_d_lr: float
    gan_d_beta1: float
    gan_d_beta2: float
    gan_d_base_channels: int
    gan_d_num_layers: int
    gan_d_fine_layers: int
    gan_d_coarse_layers: int
    gan_d_use_spectral_norm: bool
    gan_use_mask_channel: bool
    gan_ms_w_fine: float
    gan_ms_w_coarse: float
    gan_lambda_adv_max: float
    gan_adv_warmup_steps: int
    gan_r1_gamma: float
    gan_r1_every: int
    routing_kl_weight: float
    routing_kl_temperature: float
    routing_kl_eps: float
    corr_weight: float
    corr_eps: float
    corr_offdiag_only: bool
    corr_use_correlation: bool
    use_channel_aux_losses: bool


def resolve_trainer_runtime_settings(config: TrainConfig) -> TrainerRuntimeSettings:
    """Resolve optional training toggles and GAN/auxiliary-loss coefficients."""
    run_validation = bool(config.training.run_validation)
    run_validation_generations = bool(config.training.run_validation_generations)
    validation_steps = int(config.training.validation_steps)

    use_gan = bool(getattr(config.training, "use_gan", False))
    gan_d_lr = float(getattr(config.training, "gan_d_lr", 1e-4))
    gan_d_beta1 = float(getattr(config.training, "gan_d_beta1", 0.0))
    gan_d_beta2 = float(getattr(config.training, "gan_d_beta2", 0.9))
    gan_d_base_channels = int(getattr(config.training, "gan_d_base_channels", 64))
    gan_d_num_layers = int(getattr(config.training, "gan_d_num_layers", 4))
    gan_d_fine_layers = int(getattr(config.training, "gan_d_fine_layers", 3))
    gan_d_coarse_layers = int(
        getattr(config.training, "gan_d_coarse_layers", gan_d_num_layers)
    )
    gan_d_use_spectral_norm = bool(
        getattr(config.training, "gan_d_use_spectral_norm", True)
    )
    gan_use_mask_channel = bool(getattr(config.training, "gan_use_mask_channel", True))
    gan_ms_w_fine = float(getattr(config.training, "gan_ms_w_fine", 1.0))
    gan_ms_w_coarse = float(getattr(config.training, "gan_ms_w_coarse", 0.5))
    gan_lambda_adv_max = float(getattr(config.training, "gan_lambda_adv", 1e-3))
    gan_adv_warmup_steps = int(
        max(0, int(getattr(config.training, "gan_adv_warmup_steps", 20_000)))
    )
    gan_r1_gamma = float(getattr(config.training, "gan_r1_gamma", 10.0))
    gan_r1_every = int(max(1, int(getattr(config.training, "gan_r1_every", 16))))

    routing_kl_weight = float(getattr(config.training, "routing_kl_weight", 0.0))
    routing_kl_temperature = float(
        getattr(config.training, "routing_kl_temperature", 0.7)
    )
    routing_kl_eps = float(getattr(config.training, "routing_kl_eps", 1e-6))
    corr_weight = float(getattr(config.training, "corr_weight", 0.0))
    corr_eps = float(getattr(config.training, "corr_eps", 1e-6))
    corr_offdiag_only = bool(getattr(config.training, "corr_offdiag_only", True))
    corr_use_correlation = bool(getattr(config.training, "corr_use_correlation", True))
    use_channel_aux_losses = routing_kl_weight > 0.0 or corr_weight > 0.0

    return TrainerRuntimeSettings(
        run_validation=run_validation,
        run_validation_generations=run_validation_generations,
        validation_steps=validation_steps,
        use_gan=use_gan,
        gan_d_lr=gan_d_lr,
        gan_d_beta1=gan_d_beta1,
        gan_d_beta2=gan_d_beta2,
        gan_d_base_channels=gan_d_base_channels,
        gan_d_num_layers=gan_d_num_layers,
        gan_d_fine_layers=gan_d_fine_layers,
        gan_d_coarse_layers=gan_d_coarse_layers,
        gan_d_use_spectral_norm=gan_d_use_spectral_norm,
        gan_use_mask_channel=gan_use_mask_channel,
        gan_ms_w_fine=gan_ms_w_fine,
        gan_ms_w_coarse=gan_ms_w_coarse,
        gan_lambda_adv_max=gan_lambda_adv_max,
        gan_adv_warmup_steps=gan_adv_warmup_steps,
        gan_r1_gamma=gan_r1_gamma,
        gan_r1_every=gan_r1_every,
        routing_kl_weight=routing_kl_weight,
        routing_kl_temperature=routing_kl_temperature,
        routing_kl_eps=routing_kl_eps,
        corr_weight=corr_weight,
        corr_eps=corr_eps,
        corr_offdiag_only=corr_offdiag_only,
        corr_use_correlation=corr_use_correlation,
        use_channel_aux_losses=use_channel_aux_losses,
    )
