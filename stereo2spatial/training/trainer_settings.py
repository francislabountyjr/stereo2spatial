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
    downmix_consistency_weight: float
    downmix_consistency_loss: str
    downmix_channel_order: list[str] | None
    mrstft_loss_weight: float
    mrstft_fft_sizes: list[int]
    mrstft_hop_lengths: list[int]
    mrstft_win_lengths: list[int]
    mrstft_sc_weight: float
    mrstft_log_mag_weight: float
    mrstft_eps: float
    waveform_mse_loss_weight: float
    waveform_l1_loss_weight: float
    waveform_charbonnier_loss_weight: float
    waveform_charbonnier_eps: float
    perceptual_loss_weight: float
    perceptual_sample_rate: int
    perceptual_n_fft: int
    perceptual_hop_length: int
    perceptual_win_length: int
    perceptual_n_mels: int
    perceptual_f_min: float
    perceptual_f_max: float | None
    perceptual_band_weight: float
    perceptual_band_low_hz: float
    perceptual_band_high_hz: float
    perceptual_eps: float
    binaural_ild_loss_weight: float
    binaural_ipd_loss_weight: float
    binaural_ccf_loss_weight: float
    binaural_frame_ild_loss_weight: float
    binaural_frame_ild_frame_size: int
    binaural_frame_ild_hop_size: int
    binaural_frame_ild_silence_threshold: float
    binaural_frame_ild_max_weight: float
    binaural_mid_side_loss_weight: float
    binaural_mid_side_loss_type: str
    binaural_mid_side_mid_weight: float
    binaural_mid_side_side_weight: float
    binaural_mid_side_charbonnier_eps: float
    binaural_loss_warmup_steps: int
    binaural_sample_rate: int
    binaural_loss_eps: float
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
    downmix_consistency_weight = float(
        getattr(config.training, "downmix_consistency_weight", 0.0)
    )
    downmix_consistency_loss = str(
        getattr(config.training, "downmix_consistency_loss", "mse")
    )
    downmix_channel_order = getattr(config.training, "downmix_channel_order", None)
    mrstft_loss_weight = float(getattr(config.training, "mrstft_loss_weight", 0.0))
    mrstft_fft_sizes = list(
        getattr(config.training, "mrstft_fft_sizes", None) or [512, 1024, 2048]
    )
    mrstft_hop_lengths = list(
        getattr(config.training, "mrstft_hop_lengths", None) or [128, 256, 512]
    )
    mrstft_win_lengths = list(
        getattr(config.training, "mrstft_win_lengths", None) or [512, 1024, 2048]
    )
    mrstft_sc_weight = float(getattr(config.training, "mrstft_sc_weight", 1.0))
    mrstft_log_mag_weight = float(
        getattr(config.training, "mrstft_log_mag_weight", 1.0)
    )
    mrstft_eps = float(getattr(config.training, "mrstft_eps", 1e-7))
    waveform_mse_loss_weight = float(
        getattr(config.training, "waveform_mse_loss_weight", 1.0)
    )
    waveform_l1_loss_weight = float(
        getattr(config.training, "waveform_l1_loss_weight", 0.0)
    )
    waveform_charbonnier_loss_weight = float(
        getattr(config.training, "waveform_charbonnier_loss_weight", 0.0)
    )
    waveform_charbonnier_eps = float(
        getattr(config.training, "waveform_charbonnier_eps", 1e-3)
    )
    perceptual_loss_weight = float(
        getattr(config.training, "perceptual_loss_weight", 0.0)
    )
    perceptual_sample_rate = int(config.data.sample_rate)
    perceptual_n_fft = int(getattr(config.training, "perceptual_n_fft", 1024))
    perceptual_hop_length = int(getattr(config.training, "perceptual_hop_length", 256))
    perceptual_win_length = int(getattr(config.training, "perceptual_win_length", 1024))
    perceptual_n_mels = int(getattr(config.training, "perceptual_n_mels", 80))
    perceptual_f_min = float(getattr(config.training, "perceptual_f_min", 40.0))
    perceptual_f_max = getattr(config.training, "perceptual_f_max", None)
    perceptual_band_weight = float(
        getattr(config.training, "perceptual_band_weight", 1.0)
    )
    perceptual_band_low_hz = float(
        getattr(config.training, "perceptual_band_low_hz", 150.0)
    )
    perceptual_band_high_hz = float(
        getattr(config.training, "perceptual_band_high_hz", 8000.0)
    )
    perceptual_eps = float(getattr(config.training, "perceptual_eps", 1e-5))
    binaural_ild_loss_weight = float(
        getattr(config.training, "binaural_ild_loss_weight", 0.0)
    )
    binaural_ipd_loss_weight = float(
        getattr(config.training, "binaural_ipd_loss_weight", 0.0)
    )
    binaural_ccf_loss_weight = float(
        getattr(config.training, "binaural_ccf_loss_weight", 0.0)
    )
    binaural_frame_ild_loss_weight = float(
        getattr(config.training, "binaural_frame_ild_loss_weight", 0.0)
    )
    binaural_frame_ild_frame_size = int(
        getattr(config.training, "binaural_frame_ild_frame_size", 2048)
    )
    binaural_frame_ild_hop_size = int(
        getattr(config.training, "binaural_frame_ild_hop_size", 1024)
    )
    binaural_frame_ild_silence_threshold = float(
        getattr(config.training, "binaural_frame_ild_silence_threshold", 1e-4)
    )
    binaural_frame_ild_max_weight = float(
        getattr(config.training, "binaural_frame_ild_max_weight", 4.0)
    )
    binaural_mid_side_loss_weight = float(
        getattr(config.training, "binaural_mid_side_loss_weight", 0.0)
    )
    binaural_mid_side_loss_type = str(
        getattr(config.training, "binaural_mid_side_loss_type", "charbonnier")
    )
    binaural_mid_side_mid_weight = float(
        getattr(config.training, "binaural_mid_side_mid_weight", 0.0)
    )
    binaural_mid_side_side_weight = float(
        getattr(config.training, "binaural_mid_side_side_weight", 1.0)
    )
    binaural_mid_side_charbonnier_eps = float(
        getattr(config.training, "binaural_mid_side_charbonnier_eps", 1e-3)
    )
    binaural_loss_warmup_steps = int(
        max(0, int(getattr(config.training, "binaural_loss_warmup_steps", 0)))
    )
    binaural_sample_rate = int(config.data.sample_rate)
    binaural_loss_eps = float(getattr(config.training, "binaural_loss_eps", 1e-7))
    use_channel_aux_losses = (
        routing_kl_weight > 0.0
        or corr_weight > 0.0
        or downmix_consistency_weight > 0.0
        or mrstft_loss_weight > 0.0
        or perceptual_loss_weight > 0.0
        or binaural_ild_loss_weight > 0.0
        or binaural_ipd_loss_weight > 0.0
        or binaural_ccf_loss_weight > 0.0
        or binaural_frame_ild_loss_weight > 0.0
        or binaural_mid_side_loss_weight > 0.0
    )

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
        downmix_consistency_weight=downmix_consistency_weight,
        downmix_consistency_loss=downmix_consistency_loss,
        downmix_channel_order=downmix_channel_order,
        mrstft_loss_weight=mrstft_loss_weight,
        mrstft_fft_sizes=mrstft_fft_sizes,
        mrstft_hop_lengths=mrstft_hop_lengths,
        mrstft_win_lengths=mrstft_win_lengths,
        mrstft_sc_weight=mrstft_sc_weight,
        mrstft_log_mag_weight=mrstft_log_mag_weight,
        mrstft_eps=mrstft_eps,
        waveform_mse_loss_weight=waveform_mse_loss_weight,
        waveform_l1_loss_weight=waveform_l1_loss_weight,
        waveform_charbonnier_loss_weight=waveform_charbonnier_loss_weight,
        waveform_charbonnier_eps=waveform_charbonnier_eps,
        perceptual_loss_weight=perceptual_loss_weight,
        perceptual_sample_rate=perceptual_sample_rate,
        perceptual_n_fft=perceptual_n_fft,
        perceptual_hop_length=perceptual_hop_length,
        perceptual_win_length=perceptual_win_length,
        perceptual_n_mels=perceptual_n_mels,
        perceptual_f_min=perceptual_f_min,
        perceptual_f_max=perceptual_f_max,
        perceptual_band_weight=perceptual_band_weight,
        perceptual_band_low_hz=perceptual_band_low_hz,
        perceptual_band_high_hz=perceptual_band_high_hz,
        perceptual_eps=perceptual_eps,
        binaural_ild_loss_weight=binaural_ild_loss_weight,
        binaural_ipd_loss_weight=binaural_ipd_loss_weight,
        binaural_ccf_loss_weight=binaural_ccf_loss_weight,
        binaural_frame_ild_loss_weight=binaural_frame_ild_loss_weight,
        binaural_frame_ild_frame_size=binaural_frame_ild_frame_size,
        binaural_frame_ild_hop_size=binaural_frame_ild_hop_size,
        binaural_frame_ild_silence_threshold=binaural_frame_ild_silence_threshold,
        binaural_frame_ild_max_weight=binaural_frame_ild_max_weight,
        binaural_mid_side_loss_weight=binaural_mid_side_loss_weight,
        binaural_mid_side_loss_type=binaural_mid_side_loss_type,
        binaural_mid_side_mid_weight=binaural_mid_side_mid_weight,
        binaural_mid_side_side_weight=binaural_mid_side_side_weight,
        binaural_mid_side_charbonnier_eps=binaural_mid_side_charbonnier_eps,
        binaural_loss_warmup_steps=binaural_loss_warmup_steps,
        binaural_sample_rate=binaural_sample_rate,
        binaural_loss_eps=binaural_loss_eps,
        use_channel_aux_losses=use_channel_aux_losses,
    )
