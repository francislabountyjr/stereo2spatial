"""Typed schema for resolved training configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataConfig:
    """Data loading and sequence sampling settings."""

    dataset_root: str | list[str]
    manifest_path: str | list[str]
    sample_artifact_mode: str
    segment_seconds: float
    sequence_seconds: float
    stride_seconds: float
    sample_rate: int
    mono_probability: float
    downmix_probability: float
    cache_size: int
    shuffle_segments_within_epoch: bool
    batch_size: int
    num_workers: int
    prefetch_factor: int
    pin_memory: bool
    persistent_workers: bool
    drop_last: bool
    materialize_cached_signals: bool = False
    shuffle_segments_within_song: bool = True
    batch_mode: str = "standard"
    amplitude_lift_enabled: bool = False
    amplitude_lift_reference: str = "source"
    amplitude_lift_target_rms: float = 0.33
    amplitude_lift_scale: float = 3.0
    amplitude_lift_clip_value: float | None = 4.0
    amplitude_lift_eps: float = 1.0e-8
    source_resample_aug_enabled: bool = False
    source_resample_aug_probability: float = 0.0
    source_resample_aug_rates: list[int] | None = None
    source_resample_aug_weights: list[float] | None = None
    source_codec_aug_enabled: bool = False
    source_codec_aug_probability: float = 0.0
    source_codec_aug_start_step: int = 0
    source_codec_aug_full_strength_step: int = 0
    source_codec_aug_backend: str = "auto"
    source_codec_aug_ffmpeg_path: str = "ffmpeg"
    source_codec_aug_codecs: list[str] | None = None
    source_codec_aug_codec_weights: list[float] | None = None
    source_codec_aug_bitrates: dict[str, list[int]] | None = None
    source_codec_aug_max_chunk_seconds: float | None = 12.0
    source_codec_aug_align_max_lag: int = 8192
    source_codec_aug_timeout_seconds: float = 20.0


@dataclass
class ModelConfig:
    """SpatialDiT architecture configuration."""

    target_channels: int
    cond_channels: int
    patch_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    timestep_embed_dim: int
    timestep_scale: float
    max_period: float
    num_memory_tokens: int
    mix_style_dim: int = 0
    waveform_level_depth: int = 0
    waveform_micro_patch_size: int = 16
    waveform_hidden_dim: int = 16
    waveform_num_heads: int | None = None
    waveform_mlp_ratio: float = 2.0
    activation_checkpointing: bool = False


@dataclass
class TrainingConfig:
    """Training-loop, windowing, GAN, and validation controls."""

    max_steps: int
    grad_accum_steps: int
    mixed_precision: str
    compile_model: bool
    compile_mode: str
    resume_from_checkpoint: str | None
    init_from_checkpoint: str | None
    grad_clip_norm: float
    log_every: int
    checkpoint_every: int
    max_checkpoints_to_keep: int
    num_epochs_hint: int
    window_seconds: float
    overlap_seconds: float
    sequence_seconds_choices: list[float]
    randomize_sequence_per_batch: bool
    detach_memory: bool
    sequence_mode: str
    tbptt_windows: int
    full_song_max_seconds: float | None
    require_batch_size_one_for_full_song: bool
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
    gan_lambda_adv: float
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
    run_validation: bool
    validation_dataset_root: str | None
    validation_dataset_path: str | None
    validation_steps: int
    run_validation_generations: bool
    num_valid_generations: int
    validation_generation_seed: int
    validation_generation_input_path: str | None
    validation_generation_output_path: str | None
    validation_generation_solver: str = "heun"
    validation_generation_solver_steps: int = 64
    validation_generation_solver_rtol: float = 1e-5
    validation_generation_solver_atol: float = 1e-5
    validation_generation_chunk_seconds: float | None = None
    validation_generation_overlap_seconds: float = 0.5
    downmix_consistency_weight: float = 0.0
    downmix_consistency_loss: str = "mse"
    downmix_channel_order: list[str] | None = None
    mix_style_dropout_probability: float = 0.0
    mrstft_loss_weight: float = 0.0
    mrstft_fft_sizes: list[int] | None = None
    mrstft_hop_lengths: list[int] | None = None
    mrstft_win_lengths: list[int] | None = None
    mrstft_sc_weight: float = 1.0
    mrstft_log_mag_weight: float = 1.0
    mrstft_eps: float = 1e-7
    waveform_mse_loss_weight: float = 1.0
    waveform_l1_loss_weight: float = 0.0
    waveform_charbonnier_loss_weight: float = 0.0
    waveform_charbonnier_eps: float = 1e-3
    perceptual_loss_weight: float = 0.0
    perceptual_n_fft: int = 1024
    perceptual_hop_length: int = 256
    perceptual_win_length: int = 1024
    perceptual_n_mels: int = 80
    perceptual_f_min: float = 40.0
    perceptual_f_max: float | None = None
    perceptual_band_weight: float = 1.0
    perceptual_band_low_hz: float = 150.0
    perceptual_band_high_hz: float = 8000.0
    perceptual_eps: float = 1e-5
    binaural_ild_loss_weight: float = 0.0
    binaural_ipd_loss_weight: float = 0.0
    binaural_ccf_loss_weight: float = 0.0
    binaural_frame_ild_loss_weight: float = 0.0
    binaural_frame_ild_frame_size: int = 2048
    binaural_frame_ild_hop_size: int = 1024
    binaural_frame_ild_silence_threshold: float = 1e-4
    binaural_frame_ild_max_weight: float = 4.0
    binaural_mid_side_loss_weight: float = 0.0
    binaural_mid_side_loss_type: str = "charbonnier"
    binaural_mid_side_mid_weight: float = 0.0
    binaural_mid_side_side_weight: float = 1.0
    binaural_mid_side_charbonnier_eps: float = 1e-3
    binaural_loss_warmup_steps: int = 0
    binaural_loss_eps: float = 1e-7
    scheduled_sampling_max_step_offset: int = 0
    scheduled_sampling_probability: float = 0.0
    scheduled_sampling_prob_start: float | None = None
    scheduled_sampling_prob_end: float | None = None
    scheduled_sampling_ramp_steps: int = 0
    scheduled_sampling_start_step: int = 0
    scheduled_sampling_ramp_shape: str = "linear"
    scheduled_sampling_strategy: str = "uniform"
    scheduled_sampling_sampler: str = "heun"
    scheduled_sampling_reflexflow: bool | None = None
    scheduled_sampling_reflexflow_alpha: float = 1.0
    scheduled_sampling_reflexflow_beta1: float = 10.0
    scheduled_sampling_reflexflow_beta2: float = 1.0
    flow_timestep_sampling: str = "uniform"
    flow_fast_schedule: bool = False
    flow_logit_mean: float = 0.0
    flow_logit_std: float = 1.0
    flow_beta_alpha: float = 1.0
    flow_beta_beta: float = 1.0
    flow_custom_timesteps: list[float] | None = None
    flow_schedule_shift: float | None = None
    flow_schedule_auto_shift: bool = False
    flow_schedule_base_seq_len: int = 256
    flow_schedule_max_seq_len: int = 4096
    flow_schedule_base_shift: float = 0.5
    flow_schedule_max_shift: float = 1.15
    flow_loss_weighting: str = "none"
    use_ema: bool = False
    ema_decay: float = 0.999
    ema_device: str = "accelerator"
    ema_cpu_only: bool = False


@dataclass
class OptimizerConfig:
    """Optimizer hyper-parameters and backend toggles."""

    type: str
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    eps: float
    adamw_fused: bool
    adamw_foreach: bool
    muon_ns_steps: int = 5
    muon_nesterov: bool = True


@dataclass
class SchedulerConfig:
    """Learning-rate scheduler settings."""

    type: str
    warmup_steps: int
    min_lr: float


@dataclass
class TrainConfig:
    """Root configuration object consumed by train/infer entrypoints."""

    seed: int
    output_dir: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
