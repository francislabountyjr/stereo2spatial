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


def _int_list(value: Any, default: list[int]) -> list[int]:
    """Return a list of coerced ints from config or a default list."""
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    raise TypeError("Expected value to be a list/tuple or null.")


def _optional_int_list(value: Any) -> list[int] | None:
    """Return ``None`` or a list of coerced ints."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    raise TypeError("Expected value to be a list/tuple or null.")


def _optional_str_list(value: Any) -> list[str] | None:
    """Return ``None`` or a list of coerced string values."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise TypeError("Expected value to be a list/tuple or null.")


def _coerce_dataset_paths(
    data_raw: dict[str, Any]
) -> tuple[str | list[str], str | list[str]]:
    """Return dataset roots and manifests from single-path or multi-path config."""
    datasets_raw = data_raw.get("datasets")
    if datasets_raw is not None:
        if not isinstance(datasets_raw, (list, tuple)) or not datasets_raw:
            raise TypeError("data.datasets must be a non-empty list of mappings.")

        roots: list[str] = []
        manifests: list[str] = []
        for index, item in enumerate(datasets_raw):
            if not isinstance(item, dict):
                raise TypeError(f"data.datasets[{index}] must be a mapping.")
            roots.append(str(require_key(item, "dataset_root")))
            manifests.append(str(require_key(item, "manifest_path")))
        return roots, manifests

    dataset_root = require_key(data_raw, "dataset_root")
    manifest_path = require_key(data_raw, "manifest_path")
    if isinstance(dataset_root, (list, tuple)) or isinstance(
        manifest_path, (list, tuple)
    ):
        if not isinstance(dataset_root, (list, tuple)):
            raise TypeError(
                "data.dataset_root must be a list when manifest_path is a list."
            )
        if not isinstance(manifest_path, (list, tuple)):
            raise TypeError(
                "data.manifest_path must be a list when dataset_root is a list."
            )
        roots = [str(item) for item in dataset_root]
        manifests = [str(item) for item in manifest_path]
        if not roots:
            raise ValueError("data.dataset_root must contain at least one path.")
        if len(roots) != len(manifests):
            raise ValueError(
                "data.dataset_root and data.manifest_path lists must have the same length."
            )
        return roots, manifests

    return str(dataset_root), str(manifest_path)


def _build_source_resample_aug_fields(data_raw: dict[str, Any]) -> dict[str, Any]:
    """Build optional source-only sample-rate roundtrip augmentation fields."""
    aug_raw = data_raw.get("source_resample_augmentation")
    if aug_raw is None:
        aug_raw = {}
    if not isinstance(aug_raw, dict):
        raise TypeError("data.source_resample_augmentation must be a mapping.")

    return {
        "source_resample_aug_enabled": bool(aug_raw.get("enabled", False)),
        "source_resample_aug_probability": float(aug_raw.get("probability", 0.0)),
        "source_resample_aug_rates": _optional_int_list(aug_raw.get("rates")),
        "source_resample_aug_weights": _optional_float_list(aug_raw.get("weights")),
    }


def _build_source_codec_aug_fields(data_raw: dict[str, Any]) -> dict[str, Any]:
    """Build optional source-only lossy-codec augmentation fields."""
    aug_raw = data_raw.get("source_codec_augmentation")
    if aug_raw is None:
        aug_raw = {}
    if not isinstance(aug_raw, dict):
        raise TypeError("data.source_codec_augmentation must be a mapping.")

    codecs_raw = aug_raw.get("codecs")
    codecs: list[str] | None = None
    codec_weights: list[float] | None = None
    bitrates: dict[str, list[int]] | None = None
    if codecs_raw is not None:
        if not isinstance(codecs_raw, dict):
            raise TypeError("data.source_codec_augmentation.codecs must be a mapping.")
        codecs = []
        codec_weights = []
        bitrates = {}
        for codec_name, codec_raw in codecs_raw.items():
            if not isinstance(codec_raw, dict):
                raise TypeError(
                    "data.source_codec_augmentation.codecs entries must be mappings."
                )
            name = str(codec_name).strip().lower()
            codecs.append(name)
            codec_weights.append(float(codec_raw.get("weight", 1.0)))
            bitrates[name] = _int_list(codec_raw.get("bitrates"), [])

    max_chunk_seconds = aug_raw.get("max_chunk_seconds", 12.0)
    return {
        "source_codec_aug_enabled": bool(aug_raw.get("enabled", False)),
        "source_codec_aug_probability": float(aug_raw.get("probability", 0.0)),
        "source_codec_aug_start_step": int(aug_raw.get("start_step", 0)),
        "source_codec_aug_full_strength_step": int(
            aug_raw.get("full_strength_step", 0)
        ),
        "source_codec_aug_backend": str(aug_raw.get("backend", "auto")),
        "source_codec_aug_ffmpeg_path": str(aug_raw.get("ffmpeg_path", "ffmpeg")),
        "source_codec_aug_codecs": codecs,
        "source_codec_aug_codec_weights": codec_weights,
        "source_codec_aug_bitrates": bitrates,
        "source_codec_aug_max_chunk_seconds": (
            None if max_chunk_seconds is None else float(max_chunk_seconds)
        ),
        "source_codec_aug_align_max_lag": int(aug_raw.get("align_max_lag", 8192)),
        "source_codec_aug_timeout_seconds": float(
            aug_raw.get("timeout_seconds", 20.0)
        ),
    }


def build_data_config(data_raw: dict[str, Any]) -> DataConfig:
    """Coerce the ``data`` section into :class:`DataConfig`."""
    dataset_root, manifest_path = _coerce_dataset_paths(data_raw)
    return DataConfig(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        sample_artifact_mode=str(require_key(data_raw, "sample_artifact_mode")),
        segment_seconds=float(require_key(data_raw, "segment_seconds")),
        sequence_seconds=float(require_key(data_raw, "sequence_seconds")),
        stride_seconds=float(require_key(data_raw, "stride_seconds")),
        sample_rate=int(require_key(data_raw, "sample_rate")),
        mono_probability=float(require_key(data_raw, "mono_probability")),
        downmix_probability=float(require_key(data_raw, "downmix_probability")),
        cache_size=int(require_key(data_raw, "cache_size")),
        shuffle_segments_within_epoch=bool(
            require_key(data_raw, "shuffle_segments_within_epoch")
        ),
        shuffle_segments_within_song=bool(
            data_raw.get("shuffle_segments_within_song", True)
        ),
        batch_size=int(require_key(data_raw, "batch_size")),
        num_workers=int(require_key(data_raw, "num_workers")),
        prefetch_factor=int(data_raw.get("prefetch_factor", 2)),
        pin_memory=bool(require_key(data_raw, "pin_memory")),
        persistent_workers=bool(require_key(data_raw, "persistent_workers")),
        drop_last=bool(require_key(data_raw, "drop_last")),
        materialize_cached_signals=bool(
            data_raw.get("materialize_cached_signals", False)
        ),
        batch_mode=str(data_raw.get("batch_mode", "standard")),
        amplitude_lift_enabled=bool(data_raw.get("amplitude_lift_enabled", False)),
        amplitude_lift_reference=str(
            data_raw.get("amplitude_lift_reference", "source")
        ),
        amplitude_lift_target_rms=float(
            data_raw.get("amplitude_lift_target_rms", 0.33)
        ),
        amplitude_lift_scale=float(data_raw.get("amplitude_lift_scale", 3.0)),
        amplitude_lift_clip_value=(
            None
            if data_raw.get("amplitude_lift_clip_value", 4.0) is None
            else float(data_raw.get("amplitude_lift_clip_value", 4.0))
        ),
        amplitude_lift_eps=float(data_raw.get("amplitude_lift_eps", 1.0e-8)),
        **_build_source_resample_aug_fields(data_raw),
        **_build_source_codec_aug_fields(data_raw),
    )


def build_model_config(model_raw: dict[str, Any]) -> ModelConfig:
    """Coerce the ``model`` section into :class:`ModelConfig`."""
    return ModelConfig(
        target_channels=int(require_key(model_raw, "target_channels")),
        cond_channels=int(require_key(model_raw, "cond_channels")),
        patch_size=int(require_key(model_raw, "patch_size")),
        hidden_dim=int(require_key(model_raw, "hidden_dim")),
        num_layers=int(require_key(model_raw, "num_layers")),
        num_heads=int(require_key(model_raw, "num_heads")),
        mlp_ratio=float(require_key(model_raw, "mlp_ratio")),
        dropout=float(require_key(model_raw, "dropout")),
        timestep_embed_dim=int(require_key(model_raw, "timestep_embed_dim")),
        timestep_scale=float(require_key(model_raw, "timestep_scale")),
        max_period=float(require_key(model_raw, "max_period")),
        num_memory_tokens=int(model_raw.get("num_memory_tokens", 0)),
        mix_style_dim=int(model_raw.get("mix_style_dim", 0)),
        waveform_level_depth=int(model_raw.get("waveform_level_depth", 0)),
        waveform_micro_patch_size=int(model_raw.get("waveform_micro_patch_size", 16)),
        waveform_hidden_dim=int(model_raw.get("waveform_hidden_dim", 16)),
        waveform_num_heads=(
            int(model_raw["waveform_num_heads"])
            if model_raw.get("waveform_num_heads") is not None
            else None
        ),
        waveform_mlp_ratio=float(model_raw.get("waveform_mlp_ratio", 2.0)),
        activation_checkpointing=bool(model_raw.get("activation_checkpointing", False)),
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
        "downmix_consistency_weight": float(
            training_raw.get("downmix_consistency_weight", 0.0)
        ),
        "downmix_consistency_loss": str(
            training_raw.get("downmix_consistency_loss", "mse")
        ),
        "downmix_channel_order": _optional_str_list(
            training_raw.get("downmix_channel_order")
        ),
        "mix_style_dropout_probability": float(
            training_raw.get("mix_style_dropout_probability", 0.0)
        ),
        "mrstft_loss_weight": float(training_raw.get("mrstft_loss_weight", 0.0)),
        "mrstft_fft_sizes": _int_list(
            training_raw.get("mrstft_fft_sizes"),
            [512, 1024, 2048],
        ),
        "mrstft_hop_lengths": _int_list(
            training_raw.get("mrstft_hop_lengths"),
            [128, 256, 512],
        ),
        "mrstft_win_lengths": _int_list(
            training_raw.get("mrstft_win_lengths"),
            [512, 1024, 2048],
        ),
        "mrstft_sc_weight": float(training_raw.get("mrstft_sc_weight", 1.0)),
        "mrstft_log_mag_weight": float(training_raw.get("mrstft_log_mag_weight", 1.0)),
        "mrstft_eps": float(training_raw.get("mrstft_eps", 1e-7)),
        "waveform_mse_loss_weight": float(
            training_raw.get("waveform_mse_loss_weight", 1.0)
        ),
        "waveform_l1_loss_weight": float(
            training_raw.get("waveform_l1_loss_weight", 0.0)
        ),
        "waveform_charbonnier_loss_weight": float(
            training_raw.get("waveform_charbonnier_loss_weight", 0.0)
        ),
        "waveform_charbonnier_eps": float(
            training_raw.get("waveform_charbonnier_eps", 1e-3)
        ),
        "perceptual_loss_weight": float(
            training_raw.get("perceptual_loss_weight", 0.0)
        ),
        "perceptual_n_fft": int(training_raw.get("perceptual_n_fft", 1024)),
        "perceptual_hop_length": int(training_raw.get("perceptual_hop_length", 256)),
        "perceptual_win_length": int(training_raw.get("perceptual_win_length", 1024)),
        "perceptual_n_mels": int(training_raw.get("perceptual_n_mels", 80)),
        "perceptual_f_min": float(training_raw.get("perceptual_f_min", 40.0)),
        "perceptual_f_max": _optional_float(training_raw.get("perceptual_f_max")),
        "perceptual_band_weight": float(
            training_raw.get("perceptual_band_weight", 1.0)
        ),
        "perceptual_band_low_hz": float(
            training_raw.get("perceptual_band_low_hz", 150.0)
        ),
        "perceptual_band_high_hz": float(
            training_raw.get("perceptual_band_high_hz", 8000.0)
        ),
        "perceptual_eps": float(training_raw.get("perceptual_eps", 1e-5)),
        "binaural_ild_loss_weight": float(
            training_raw.get("binaural_ild_loss_weight", 0.0)
        ),
        "binaural_ipd_loss_weight": float(
            training_raw.get("binaural_ipd_loss_weight", 0.0)
        ),
        "binaural_ccf_loss_weight": float(
            training_raw.get("binaural_ccf_loss_weight", 0.0)
        ),
        "binaural_frame_ild_loss_weight": float(
            training_raw.get("binaural_frame_ild_loss_weight", 0.0)
        ),
        "binaural_frame_ild_frame_size": int(
            training_raw.get("binaural_frame_ild_frame_size", 2048)
        ),
        "binaural_frame_ild_hop_size": int(
            training_raw.get("binaural_frame_ild_hop_size", 1024)
        ),
        "binaural_frame_ild_silence_threshold": float(
            training_raw.get("binaural_frame_ild_silence_threshold", 1e-4)
        ),
        "binaural_frame_ild_max_weight": float(
            training_raw.get("binaural_frame_ild_max_weight", 4.0)
        ),
        "binaural_mid_side_loss_weight": float(
            training_raw.get("binaural_mid_side_loss_weight", 0.0)
        ),
        "binaural_mid_side_loss_type": str(
            training_raw.get("binaural_mid_side_loss_type", "charbonnier")
        ),
        "binaural_mid_side_mid_weight": float(
            training_raw.get("binaural_mid_side_mid_weight", 0.0)
        ),
        "binaural_mid_side_side_weight": float(
            training_raw.get("binaural_mid_side_side_weight", 1.0)
        ),
        "binaural_mid_side_charbonnier_eps": float(
            training_raw.get("binaural_mid_side_charbonnier_eps", 1e-3)
        ),
        "binaural_loss_warmup_steps": int(
            training_raw.get("binaural_loss_warmup_steps", 0)
        ),
        "binaural_loss_eps": float(training_raw.get("binaural_loss_eps", 1e-7)),
    }


def _build_training_validation_fields(training_raw: dict[str, Any]) -> dict[str, Any]:
    """Build validation-related training fields."""
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
        "validation_generation_solver": str(
            training_raw.get("validation_generation_solver", "heun")
        ),
        "validation_generation_solver_steps": int(
            training_raw.get("validation_generation_solver_steps", 64)
        ),
        "validation_generation_solver_rtol": float(
            training_raw.get("validation_generation_solver_rtol", 1e-5)
        ),
        "validation_generation_solver_atol": float(
            training_raw.get("validation_generation_solver_atol", 1e-5)
        ),
        "validation_generation_chunk_seconds": _optional_float(
            training_raw.get("validation_generation_chunk_seconds")
        ),
        "validation_generation_overlap_seconds": float(
            training_raw.get("validation_generation_overlap_seconds", 0.5)
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
        max_checkpoints_to_keep=int(
            require_key(training_raw, "max_checkpoints_to_keep")
        ),
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
        muon_ns_steps=int(optimizer_raw.get("muon_ns_steps", 5)),
        muon_nesterov=bool(optimizer_raw.get("muon_nesterov", True)),
    )


def build_scheduler_config(scheduler_raw: dict[str, Any]) -> SchedulerConfig:
    """Coerce the ``scheduler`` section into :class:`SchedulerConfig`."""
    return SchedulerConfig(
        type=str(require_key(scheduler_raw, "type")),
        warmup_steps=int(require_key(scheduler_raw, "warmup_steps")),
        min_lr=float(require_key(scheduler_raw, "min_lr")),
    )
