"""Section-level coercion helpers for YAML training configs."""

from __future__ import annotations

from typing import Any

from .types import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from .validators.common import optional_str


def require_key(config: dict[str, Any], key: str) -> Any:
    """Return a required mapping key or raise a descriptive error."""
    if key not in config:
        raise KeyError(f"Missing required config key: {key}")
    return config[key]


def _optional_float(value: Any) -> float | None:
    """Return ``None`` or a coerced float value."""
    if value is None:
        return None
    return float(value)


def _optional_bool(value: Any) -> bool | None:
    """Return ``None`` or a coerced bool value."""
    if value is None:
        return None
    return bool(value)


def _optional_float_list(value: Any) -> list[float] | None:
    """Return ``None`` or a list of coerced float values."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    raise TypeError("Expected flow_custom_timesteps to be a list/tuple or null.")


def build_data_config(data_raw: dict[str, Any]) -> DataConfig:
    """Coerce the ``data`` section into :class:`DataConfig`."""
    return DataConfig(
        dataset_root=str(require_key(data_raw, "dataset_root")),
        manifest_path=str(require_key(data_raw, "manifest_path")),
        sample_artifact_mode=str(require_key(data_raw, "sample_artifact_mode")),
        segment_seconds=float(require_key(data_raw, "segment_seconds")),
        sequence_seconds=float(require_key(data_raw, "sequence_seconds")),
        stride_seconds=float(require_key(data_raw, "stride_seconds")),
        latent_fps=require_key(data_raw, "latent_fps"),
        mono_probability=float(require_key(data_raw, "mono_probability")),
        downmix_probability=float(require_key(data_raw, "downmix_probability")),
        cache_size=int(require_key(data_raw, "cache_size")),
        shuffle_segments_within_epoch=bool(
            require_key(data_raw, "shuffle_segments_within_epoch")
        ),
        batch_size=int(require_key(data_raw, "batch_size")),
        num_workers=int(require_key(data_raw, "num_workers")),
        prefetch_factor=int(data_raw.get("prefetch_factor", 2)),
        pin_memory=bool(require_key(data_raw, "pin_memory")),
        persistent_workers=bool(require_key(data_raw, "persistent_workers")),
        drop_last=bool(require_key(data_raw, "drop_last")),
    )


def build_model_config(model_raw: dict[str, Any]) -> ModelConfig:
    """Coerce the ``model`` section into :class:`ModelConfig`."""
    return ModelConfig(
        target_channels=int(require_key(model_raw, "target_channels")),
        cond_channels=int(require_key(model_raw, "cond_channels")),
        latent_dim=int(require_key(model_raw, "latent_dim")),
        hidden_dim=int(require_key(model_raw, "hidden_dim")),
        num_layers=int(require_key(model_raw, "num_layers")),
        num_heads=int(require_key(model_raw, "num_heads")),
        mlp_ratio=float(require_key(model_raw, "mlp_ratio")),
        dropout=float(require_key(model_raw, "dropout")),
        timestep_embed_dim=int(require_key(model_raw, "timestep_embed_dim")),
        timestep_scale=float(require_key(model_raw, "timestep_scale")),
        max_period=float(require_key(model_raw, "max_period")),
        num_memory_tokens=int(model_raw.get("num_memory_tokens", 0)),
    )


def _build_training_checkpoint_fields(training_raw: dict[str, Any]) -> dict[str, Any]:
    """Build optional checkpoint resume/init fields from training config."""
    return {
        "resume_from_checkpoint": (
            str(training_raw["resume_from_checkpoint"])
            if training_raw.get("resume_from_checkpoint") is not None
            else None
        ),
        "init_from_checkpoint": (
            str(training_raw["init_from_checkpoint"])
            if training_raw.get("init_from_checkpoint") is not None
            else None
        ),
    }


def _build_training_sequence_fields(
    training_raw: dict[str, Any],
    sequence_seconds_default: float,
) -> dict[str, Any]:
    """Build sequence-mode and TBPTT-related training fields."""
    return {
        "sequence_seconds_choices": [
            float(item)
            for item in training_raw.get(
                "sequence_seconds_choices",
                [sequence_seconds_default],
            )
        ],
        "randomize_sequence_per_batch": bool(
            training_raw.get("randomize_sequence_per_batch", True)
        ),
        "detach_memory": bool(training_raw.get("detach_memory", True)),
        "sequence_mode": str(training_raw.get("sequence_mode", "strided_crops")),
        "tbptt_windows": int(training_raw.get("tbptt_windows", 0)),
        "full_song_max_seconds": (
            float(training_raw["full_song_max_seconds"])
            if training_raw.get("full_song_max_seconds") is not None
            else None
        ),
        "require_batch_size_one_for_full_song": bool(
            training_raw.get("require_batch_size_one_for_full_song", True)
        ),
    }


def _build_training_gan_fields(training_raw: dict[str, Any]) -> dict[str, Any]:
    """Build discriminator and adversarial training hyperparameter fields."""
    return {
        "use_gan": bool(training_raw.get("use_gan", False)),
        "gan_d_lr": float(training_raw.get("gan_d_lr", 1e-4)),
        "gan_d_beta1": float(training_raw.get("gan_d_beta1", 0.0)),
        "gan_d_beta2": float(training_raw.get("gan_d_beta2", 0.9)),
        "gan_d_base_channels": int(training_raw.get("gan_d_base_channels", 64)),
        "gan_d_num_layers": int(training_raw.get("gan_d_num_layers", 4)),
        "gan_d_fine_layers": int(training_raw.get("gan_d_fine_layers", 3)),
        "gan_d_coarse_layers": int(
            training_raw.get(
                "gan_d_coarse_layers",
                int(training_raw.get("gan_d_num_layers", 4)),
            )
        ),
        "gan_d_use_spectral_norm": bool(
            training_raw.get("gan_d_use_spectral_norm", True)
        ),
        "gan_use_mask_channel": bool(training_raw.get("gan_use_mask_channel", True)),
        "gan_ms_w_fine": float(training_raw.get("gan_ms_w_fine", 1.0)),
        "gan_ms_w_coarse": float(training_raw.get("gan_ms_w_coarse", 0.5)),
        "gan_lambda_adv": float(training_raw.get("gan_lambda_adv", 1e-3)),
        "gan_adv_warmup_steps": int(training_raw.get("gan_adv_warmup_steps", 20_000)),
        "gan_r1_gamma": float(training_raw.get("gan_r1_gamma", 10.0)),
        "gan_r1_every": int(training_raw.get("gan_r1_every", 16)),
    }


def _build_training_aux_loss_fields(training_raw: dict[str, Any]) -> dict[str, Any]:
    """Build optional routing/correlation auxiliary loss fields."""
    return {
        "routing_kl_weight": float(
            training_raw.get(
                "routing_kl_weight",
                training_raw.get("channel_routing_loss_weight", 0.0),
            )
        ),
        "routing_kl_temperature": float(
            training_raw.get(
                "routing_kl_temperature",
                training_raw.get("channel_routing_temperature", 0.7),
            )
        ),
        "routing_kl_eps": float(
            training_raw.get(
                "routing_kl_eps",
                training_raw.get("channel_routing_eps", 1e-6),
            )
        ),
        "corr_weight": float(
            training_raw.get(
                "corr_weight",
                training_raw.get("channel_correlation_loss_weight", 0.0),
            )
        ),
        "corr_eps": float(
            training_raw.get(
                "corr_eps",
                training_raw.get("channel_correlation_eps", 1e-6),
            )
        ),
        "corr_offdiag_only": bool(training_raw.get("corr_offdiag_only", True)),
        "corr_use_correlation": bool(training_raw.get("corr_use_correlation", True)),
    }


def _build_training_validation_fields(training_raw: dict[str, Any]) -> dict[str, Any]:
    """Build latent/generation validation-related training fields."""
    return {
        "run_validation": bool(training_raw.get("run_validation", False)),
        "validation_dataset_root": optional_str(
            training_raw.get("validation_dataset_root")
        ),
        "validation_dataset_path": optional_str(
            training_raw.get("validation_dataset_path")
        ),
        "validation_steps": int(training_raw.get("validation_steps", 0)),
        "run_validation_generations": bool(
            training_raw.get("run_validation_generations", False)
        ),
        "num_valid_generations": int(training_raw.get("num_valid_generations", 1)),
        "validation_generation_seed": int(
            training_raw.get("validation_generation_seed", 1337)
        ),
        "validation_generation_input_path": optional_str(
            training_raw.get("validation_generation_input_path")
        ),
        "validation_generation_output_path": optional_str(
            training_raw.get("validation_generation_output_path")
        ),
        "validation_generation_vae_checkpoint_path": optional_str(
            training_raw.get("validation_generation_vae_checkpoint_path")
        ),
        "validation_generation_vae_config_path": optional_str(
            training_raw.get("validation_generation_vae_config_path")
        ),
    }


def _build_training_scheduled_sampling_fields(
    training_raw: dict[str, Any],
) -> dict[str, Any]:
    """Build flow-matching scheduled-sampling rollout fields."""
    return {
        "scheduled_sampling_max_step_offset": int(
            training_raw.get("scheduled_sampling_max_step_offset", 0)
        ),
        "scheduled_sampling_probability": float(
            training_raw.get("scheduled_sampling_probability", 0.0)
        ),
        "scheduled_sampling_prob_start": _optional_float(
            training_raw.get("scheduled_sampling_prob_start")
        ),
        "scheduled_sampling_prob_end": _optional_float(
            training_raw.get("scheduled_sampling_prob_end")
        ),
        "scheduled_sampling_ramp_steps": int(
            training_raw.get("scheduled_sampling_ramp_steps", 0)
        ),
        "scheduled_sampling_start_step": int(
            training_raw.get("scheduled_sampling_start_step", 0)
        ),
        "scheduled_sampling_ramp_shape": str(
            training_raw.get("scheduled_sampling_ramp_shape", "linear")
        ),
        "scheduled_sampling_strategy": str(
            training_raw.get("scheduled_sampling_strategy", "uniform")
        ),
        "scheduled_sampling_sampler": str(
            training_raw.get("scheduled_sampling_sampler", "heun")
        ),
        "scheduled_sampling_reflexflow": _optional_bool(
            training_raw.get("scheduled_sampling_reflexflow")
        ),
        "scheduled_sampling_reflexflow_alpha": float(
            training_raw.get("scheduled_sampling_reflexflow_alpha", 1.0)
        ),
        "scheduled_sampling_reflexflow_beta1": float(
            training_raw.get("scheduled_sampling_reflexflow_beta1", 10.0)
        ),
        "scheduled_sampling_reflexflow_beta2": float(
            training_raw.get("scheduled_sampling_reflexflow_beta2", 1.0)
        ),
    }


def _build_training_flow_matching_fields(
    training_raw: dict[str, Any],
) -> dict[str, Any]:
    """Build flow-matching timestep sampling, shift, and weighting fields."""
    ema_cpu_only = bool(training_raw.get("ema_cpu_only", False))
    ema_device = str(training_raw.get("ema_device", "accelerator")).strip().lower()
    if ema_cpu_only:
        ema_device = "cpu"

    return {
        "flow_timestep_sampling": str(
            training_raw.get("flow_timestep_sampling", "uniform")
        ),
        "flow_fast_schedule": bool(training_raw.get("flow_fast_schedule", False)),
        "flow_logit_mean": float(training_raw.get("flow_logit_mean", 0.0)),
        "flow_logit_std": float(training_raw.get("flow_logit_std", 1.0)),
        "flow_beta_alpha": float(training_raw.get("flow_beta_alpha", 1.0)),
        "flow_beta_beta": float(training_raw.get("flow_beta_beta", 1.0)),
        "flow_custom_timesteps": _optional_float_list(
            training_raw.get("flow_custom_timesteps")
        ),
        "flow_schedule_shift": _optional_float(training_raw.get("flow_schedule_shift")),
        "flow_schedule_auto_shift": bool(
            training_raw.get("flow_schedule_auto_shift", False)
        ),
        "flow_schedule_base_seq_len": int(
            training_raw.get("flow_schedule_base_seq_len", 256)
        ),
        "flow_schedule_max_seq_len": int(
            training_raw.get("flow_schedule_max_seq_len", 4096)
        ),
        "flow_schedule_base_shift": float(
            training_raw.get("flow_schedule_base_shift", 0.5)
        ),
        "flow_schedule_max_shift": float(
            training_raw.get("flow_schedule_max_shift", 1.15)
        ),
        "flow_loss_weighting": str(training_raw.get("flow_loss_weighting", "none")),
        "use_ema": bool(training_raw.get("use_ema", False)),
        "ema_decay": float(training_raw.get("ema_decay", 0.999)),
        "ema_device": ema_device,
        "ema_cpu_only": ema_cpu_only,
    }


def build_training_config(
    training_raw: dict[str, Any],
    data_raw: dict[str, Any],
) -> TrainingConfig:
    """Coerce the ``training`` section into :class:`TrainingConfig`."""
    sequence_seconds_default = float(require_key(data_raw, "sequence_seconds"))
    return TrainingConfig(
        max_steps=int(require_key(training_raw, "max_steps")),
        grad_accum_steps=int(require_key(training_raw, "grad_accum_steps")),
        mixed_precision=str(require_key(training_raw, "mixed_precision")),
        compile_model=bool(training_raw.get("compile_model", False)),
        compile_mode=str(training_raw.get("compile_mode", "default")),
        grad_clip_norm=float(require_key(training_raw, "grad_clip_norm")),
        log_every=int(require_key(training_raw, "log_every")),
        checkpoint_every=int(require_key(training_raw, "checkpoint_every")),
        max_checkpoints_to_keep=int(require_key(training_raw, "max_checkpoints_to_keep")),
        num_epochs_hint=int(require_key(training_raw, "num_epochs_hint")),
        window_seconds=float(require_key(training_raw, "window_seconds")),
        overlap_seconds=float(require_key(training_raw, "overlap_seconds")),
        **_build_training_checkpoint_fields(training_raw),
        **_build_training_sequence_fields(
            training_raw=training_raw,
            sequence_seconds_default=sequence_seconds_default,
        ),
        **_build_training_gan_fields(training_raw),
        **_build_training_aux_loss_fields(training_raw),
        **_build_training_validation_fields(training_raw),
        **_build_training_scheduled_sampling_fields(training_raw),
        **_build_training_flow_matching_fields(training_raw),
    )


def build_optimizer_config(optimizer_raw: dict[str, Any]) -> OptimizerConfig:
    """Coerce the ``optimizer`` section into :class:`OptimizerConfig`."""
    return OptimizerConfig(
        type=str(optimizer_raw.get("type", "adamw")),
        lr=float(require_key(optimizer_raw, "lr")),
        weight_decay=float(require_key(optimizer_raw, "weight_decay")),
        beta1=float(require_key(optimizer_raw, "beta1")),
        beta2=float(require_key(optimizer_raw, "beta2")),
        eps=float(require_key(optimizer_raw, "eps")),
        adamw_fused=bool(optimizer_raw.get("adamw_fused", False)),
        adamw_foreach=bool(optimizer_raw.get("adamw_foreach", False)),
    )


def build_scheduler_config(scheduler_raw: dict[str, Any]) -> SchedulerConfig:
    """Coerce the ``scheduler`` section into :class:`SchedulerConfig`."""
    return SchedulerConfig(
        type=str(require_key(scheduler_raw, "type")),
        warmup_steps=int(require_key(scheduler_raw, "warmup_steps")),
        min_lr=float(require_key(scheduler_raw, "min_lr")),
    )
