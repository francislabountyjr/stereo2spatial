"""Typed schema for resolved training configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataConfig:
    """Data loading and latent-sequence sampling settings."""

    dataset_root: str
    manifest_path: str
    sample_artifact_mode: str
    segment_seconds: float
    sequence_seconds: float
    stride_seconds: float
    latent_fps: float | str
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


@dataclass
class ModelConfig:
    """SpatialDiT architecture configuration."""

    target_channels: int
    cond_channels: int
    latent_dim: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    timestep_embed_dim: int
    timestep_scale: float
    max_period: float
    num_memory_tokens: int


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
    validation_generation_vae_checkpoint_path: str | None
    validation_generation_vae_config_path: str | None
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
