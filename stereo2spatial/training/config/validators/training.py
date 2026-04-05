"""Validation rules for training section semantics."""

from __future__ import annotations

from ..types import TrainConfig
from .common import require_non_negative, require_positive


def _require_unit_interval(value: float, name: str) -> None:
    """Ensure a scalar value lies within [0, 1]."""
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


def validate_training_schedule(config: TrainConfig, sequence_mode: str) -> None:
    """Validate step cadence and sequence scheduling fields."""
    require_positive(config.training.max_steps, "training.max_steps")
    require_positive(config.training.grad_accum_steps, "training.grad_accum_steps")

    compile_mode = config.training.compile_mode.strip().lower()
    if compile_mode not in {
        "default",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    }:
        raise ValueError(
            "training.compile_mode must be one of: "
            "default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs"
        )

    if config.training.resume_from_checkpoint is not None:
        if not str(config.training.resume_from_checkpoint).strip():
            raise ValueError("training.resume_from_checkpoint cannot be empty")
    if config.training.init_from_checkpoint is not None:
        if not str(config.training.init_from_checkpoint).strip():
            raise ValueError("training.init_from_checkpoint cannot be empty")

    require_positive(config.training.log_every, "training.log_every")
    require_positive(config.training.checkpoint_every, "training.checkpoint_every")
    require_positive(
        config.training.max_checkpoints_to_keep,
        "training.max_checkpoints_to_keep",
    )

    require_positive(config.training.window_seconds, "training.window_seconds")
    require_non_negative(config.training.overlap_seconds, "training.overlap_seconds")
    if config.training.overlap_seconds >= config.training.window_seconds:
        raise ValueError(
            "training.overlap_seconds must be smaller than training.window_seconds"
        )

    if sequence_mode not in {"strided_crops", "full_song"}:
        raise ValueError(
            "training.sequence_mode must be one of: strided_crops, full_song"
        )

    require_non_negative(config.training.tbptt_windows, "training.tbptt_windows")

    if config.training.full_song_max_seconds is not None:
        if float(config.training.full_song_max_seconds) <= 0:
            raise ValueError("training.full_song_max_seconds must be > 0 when set")

    if not config.training.sequence_seconds_choices:
        raise ValueError("training.sequence_seconds_choices cannot be empty")
    if any(float(value) <= 0 for value in config.training.sequence_seconds_choices):
        raise ValueError(
            "training.sequence_seconds_choices must contain only > 0 values"
        )

    if (
        sequence_mode == "full_song"
        and config.training.require_batch_size_one_for_full_song
        and config.data.batch_size != 1
    ):
        raise ValueError(
            "training.sequence_mode=full_song currently requires data.batch_size=1 "
            "when training.require_batch_size_one_for_full_song=true"
        )

    require_non_negative(
        config.training.scheduled_sampling_max_step_offset,
        "training.scheduled_sampling_max_step_offset",
    )
    _require_unit_interval(
        float(config.training.scheduled_sampling_probability),
        "training.scheduled_sampling_probability",
    )
    if config.training.scheduled_sampling_prob_start is not None:
        _require_unit_interval(
            float(config.training.scheduled_sampling_prob_start),
            "training.scheduled_sampling_prob_start",
        )
    if config.training.scheduled_sampling_prob_end is not None:
        _require_unit_interval(
            float(config.training.scheduled_sampling_prob_end),
            "training.scheduled_sampling_prob_end",
        )
    require_non_negative(
        config.training.scheduled_sampling_ramp_steps,
        "training.scheduled_sampling_ramp_steps",
    )
    require_non_negative(
        config.training.scheduled_sampling_start_step,
        "training.scheduled_sampling_start_step",
    )

    ramp_shape = str(config.training.scheduled_sampling_ramp_shape).strip().lower()
    if ramp_shape not in {"linear", "cosine"}:
        raise ValueError(
            "training.scheduled_sampling_ramp_shape must be one of: linear, cosine"
        )

    strategy = str(config.training.scheduled_sampling_strategy).strip().lower()
    if strategy not in {"uniform", "biased_early", "biased_late"}:
        raise ValueError(
            "training.scheduled_sampling_strategy must be one of: "
            "uniform, biased_early, biased_late"
        )

    sampler = str(config.training.scheduled_sampling_sampler).strip().lower()
    if sampler not in {"euler", "heun", "unipc"}:
        raise ValueError(
            "training.scheduled_sampling_sampler must be one of: euler, heun, unipc"
        )

    if config.training.scheduled_sampling_reflexflow is not None and not isinstance(
        config.training.scheduled_sampling_reflexflow, bool
    ):
        raise ValueError("training.scheduled_sampling_reflexflow must be null/true/false")
    require_non_negative(
        config.training.scheduled_sampling_reflexflow_alpha,
        "training.scheduled_sampling_reflexflow_alpha",
    )
    require_non_negative(
        config.training.scheduled_sampling_reflexflow_beta1,
        "training.scheduled_sampling_reflexflow_beta1",
    )
    require_non_negative(
        config.training.scheduled_sampling_reflexflow_beta2,
        "training.scheduled_sampling_reflexflow_beta2",
    )

    flow_timestep_sampling = str(config.training.flow_timestep_sampling).strip().lower()
    if flow_timestep_sampling not in {"uniform", "logit_normal", "beta", "custom"}:
        raise ValueError(
            "training.flow_timestep_sampling must be one of: "
            "uniform, logit_normal, beta, custom"
        )
    require_positive(config.training.flow_logit_std, "training.flow_logit_std")
    require_positive(config.training.flow_beta_alpha, "training.flow_beta_alpha")
    require_positive(config.training.flow_beta_beta, "training.flow_beta_beta")
    if config.training.flow_custom_timesteps is not None:
        if len(config.training.flow_custom_timesteps) <= 0:
            raise ValueError("training.flow_custom_timesteps cannot be empty when set")
        for value in config.training.flow_custom_timesteps:
            if float(value) < 0.0:
                raise ValueError(
                    "training.flow_custom_timesteps values must be >= 0 (sigma or timestep)."
                )

    if config.training.flow_schedule_shift is not None:
        require_positive(config.training.flow_schedule_shift, "training.flow_schedule_shift")
    require_positive(
        config.training.flow_schedule_base_seq_len,
        "training.flow_schedule_base_seq_len",
    )
    require_positive(
        config.training.flow_schedule_max_seq_len,
        "training.flow_schedule_max_seq_len",
    )
    if config.training.flow_schedule_max_seq_len <= config.training.flow_schedule_base_seq_len:
        raise ValueError(
            "training.flow_schedule_max_seq_len must be > training.flow_schedule_base_seq_len"
        )
    require_positive(
        config.training.flow_schedule_base_shift,
        "training.flow_schedule_base_shift",
    )
    require_positive(
        config.training.flow_schedule_max_shift,
        "training.flow_schedule_max_shift",
    )

    flow_loss_weighting = str(config.training.flow_loss_weighting).strip().lower()
    if flow_loss_weighting not in {"none", "sigma_sqrt", "cosmap"}:
        raise ValueError(
            "training.flow_loss_weighting must be one of: none, sigma_sqrt, cosmap"
        )

    if not isinstance(config.training.use_ema, bool):
        raise ValueError("training.use_ema must be true/false")
    ema_decay = float(config.training.ema_decay)
    if ema_decay <= 0.0 or ema_decay >= 1.0:
        raise ValueError("training.ema_decay must be in (0, 1)")
    ema_device = str(config.training.ema_device).strip().lower()
    if ema_device not in {"accelerator", "cpu"}:
        raise ValueError("training.ema_device must be one of: accelerator, cpu")
    if not isinstance(config.training.ema_cpu_only, bool):
        raise ValueError("training.ema_cpu_only must be true/false")
    if config.training.ema_cpu_only and ema_device != "cpu":
        raise ValueError("training.ema_cpu_only=true requires training.ema_device=cpu")


def validate_training_gan(config: TrainConfig) -> None:
    """Validate discriminator and adversarial-loss hyperparameters."""
    if config.training.use_gan and config.training.tbptt_windows > 0:
        raise ValueError(
            "training.use_gan=true currently requires training.tbptt_windows=0 "
            "(GAN + truncated BPTT is not supported yet)."
        )

    require_positive(config.training.gan_d_lr, "training.gan_d_lr")
    if config.training.gan_d_beta1 < 0 or config.training.gan_d_beta1 >= 1:
        raise ValueError("training.gan_d_beta1 must be in [0, 1)")
    if config.training.gan_d_beta2 < 0 or config.training.gan_d_beta2 >= 1:
        raise ValueError("training.gan_d_beta2 must be in [0, 1)")
    require_positive(config.training.gan_d_base_channels, "training.gan_d_base_channels")
    require_positive(config.training.gan_d_num_layers, "training.gan_d_num_layers")
    require_positive(config.training.gan_d_fine_layers, "training.gan_d_fine_layers")
    require_positive(
        config.training.gan_d_coarse_layers,
        "training.gan_d_coarse_layers",
    )
    require_non_negative(config.training.gan_ms_w_fine, "training.gan_ms_w_fine")
    require_non_negative(config.training.gan_ms_w_coarse, "training.gan_ms_w_coarse")
    if (config.training.gan_ms_w_fine + config.training.gan_ms_w_coarse) <= 0:
        raise ValueError(
            "training.gan_ms_w_fine + training.gan_ms_w_coarse must be > 0"
        )
    require_non_negative(config.training.gan_lambda_adv, "training.gan_lambda_adv")
    require_non_negative(
        config.training.gan_adv_warmup_steps,
        "training.gan_adv_warmup_steps",
    )
    require_non_negative(config.training.gan_r1_gamma, "training.gan_r1_gamma")
    require_positive(config.training.gan_r1_every, "training.gan_r1_every")


def validate_training_aux_losses(config: TrainConfig) -> None:
    """Validate optional routing/correlation auxiliary losses."""
    require_non_negative(config.training.routing_kl_weight, "training.routing_kl_weight")
    require_positive(
        config.training.routing_kl_temperature,
        "training.routing_kl_temperature",
    )
    require_positive(config.training.routing_kl_eps, "training.routing_kl_eps")

    require_non_negative(config.training.corr_weight, "training.corr_weight")
    require_positive(config.training.corr_eps, "training.corr_eps")

    if (
        (config.training.routing_kl_weight > 0 or config.training.corr_weight > 0)
        and config.training.tbptt_windows > 0
    ):
        raise ValueError(
            "channel routing/correlation losses currently require "
            "training.tbptt_windows=0."
        )
