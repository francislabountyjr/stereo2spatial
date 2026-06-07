"""Validation rules for data/model config sections."""

from __future__ import annotations

from ..types import TrainConfig
from .common import require_non_negative, require_positive, require_probability


def validate_data_section(config: TrainConfig) -> None:
    """Validate data section values and probability invariants."""
    require_positive(config.data.segment_seconds, "data.segment_seconds")
    require_positive(config.data.sequence_seconds, "data.sequence_seconds")
    require_positive(config.data.stride_seconds, "data.stride_seconds")
    require_positive(config.data.sample_rate, "data.sample_rate")
    require_non_negative(config.data.cache_size, "data.cache_size")
    require_positive(config.data.batch_size, "data.batch_size")
    require_non_negative(config.data.num_workers, "data.num_workers")
    require_positive(config.data.prefetch_factor, "data.prefetch_factor")

    mono_prob = config.data.mono_probability
    downmix_prob = config.data.downmix_probability
    require_probability(mono_prob, "data.mono_probability")
    require_probability(downmix_prob, "data.downmix_probability")
    if mono_prob + downmix_prob > 1.0:
        raise ValueError(
            "data.mono_probability + data.downmix_probability must be <= 1"
        )

    artifact_mode = config.data.sample_artifact_mode.lower()
    if artifact_mode not in {"bundle", "split", "flac"}:
        raise ValueError(
            "data.sample_artifact_mode must be one of: bundle, split, flac"
        )
    batch_mode = str(config.data.batch_mode).strip().lower()
    if batch_mode not in {"standard", "song_local"}:
        raise ValueError("data.batch_mode must be one of: standard, song_local")

    lift_reference = str(config.data.amplitude_lift_reference).strip().lower()
    if lift_reference not in {"source", "target"}:
        raise ValueError(
            "data.amplitude_lift_reference must be one of: source, target"
        )
    require_positive(
        config.data.amplitude_lift_target_rms, "data.amplitude_lift_target_rms"
    )
    require_positive(config.data.amplitude_lift_scale, "data.amplitude_lift_scale")
    if config.data.amplitude_lift_clip_value is not None:
        require_positive(
            config.data.amplitude_lift_clip_value,
            "data.amplitude_lift_clip_value",
        )
    require_positive(config.data.amplitude_lift_eps, "data.amplitude_lift_eps")

    require_probability(
        config.data.source_resample_aug_probability,
        "data.source_resample_augmentation.probability",
    )
    rates = config.data.source_resample_aug_rates
    weights = config.data.source_resample_aug_weights
    if config.data.source_resample_aug_enabled:
        if not rates:
            raise ValueError(
                "data.source_resample_augmentation.rates must be non-empty "
                "when source resample augmentation is enabled"
            )
        for index, rate in enumerate(rates):
            require_positive(rate, f"data.source_resample_augmentation.rates[{index}]")
    if weights is not None:
        if not rates:
            raise ValueError(
                "data.source_resample_augmentation.weights requires rates"
            )
        if len(weights) != len(rates):
            raise ValueError(
                "data.source_resample_augmentation.weights must match rates length"
            )
        if sum(float(item) for item in weights) <= 0:
            raise ValueError(
                "data.source_resample_augmentation.weights must sum to > 0"
            )
        for index, weight in enumerate(weights):
            require_non_negative(
                weight, f"data.source_resample_augmentation.weights[{index}]"
            )

    require_probability(
        config.data.source_codec_aug_probability,
        "data.source_codec_augmentation.probability",
    )
    require_non_negative(
        config.data.source_codec_aug_start_step,
        "data.source_codec_augmentation.start_step",
    )
    require_non_negative(
        config.data.source_codec_aug_full_strength_step,
        "data.source_codec_augmentation.full_strength_step",
    )
    backend = str(config.data.source_codec_aug_backend).strip().lower()
    if backend not in {"auto", "torchaudio", "ffmpeg"}:
        raise ValueError(
            "data.source_codec_augmentation.backend must be one of: "
            "auto, torchaudio, ffmpeg"
        )
    require_positive(
        config.data.source_codec_aug_align_max_lag,
        "data.source_codec_augmentation.align_max_lag",
    )
    require_positive(
        config.data.source_codec_aug_timeout_seconds,
        "data.source_codec_augmentation.timeout_seconds",
    )
    if config.data.source_codec_aug_max_chunk_seconds is not None:
        require_positive(
            config.data.source_codec_aug_max_chunk_seconds,
            "data.source_codec_augmentation.max_chunk_seconds",
        )
    codec_names = config.data.source_codec_aug_codecs
    codec_weights = config.data.source_codec_aug_codec_weights
    codec_bitrates = config.data.source_codec_aug_bitrates
    if config.data.source_codec_aug_enabled:
        if not codec_names:
            raise ValueError(
                "data.source_codec_augmentation.codecs must be non-empty when enabled"
            )
        if codec_weights is None or len(codec_weights) != len(codec_names):
            raise ValueError(
                "data.source_codec_augmentation codec weights must match codecs"
            )
        if codec_bitrates is None:
            raise ValueError("data.source_codec_augmentation bitrates are required")
        if sum(float(item) for item in codec_weights) <= 0:
            raise ValueError(
                "data.source_codec_augmentation codec weights must sum to > 0"
            )
        for index, codec_name in enumerate(codec_names):
            normalized_codec = str(codec_name).strip().lower()
            if normalized_codec not in {"mp3", "aac", "opus"}:
                raise ValueError(
                    "data.source_codec_augmentation codecs must be one of: "
                    "mp3, aac, opus"
                )
            require_non_negative(
                codec_weights[index],
                f"data.source_codec_augmentation.codecs.{normalized_codec}.weight",
            )
            bitrates = codec_bitrates.get(normalized_codec)
            if not bitrates:
                raise ValueError(
                    "data.source_codec_augmentation codec bitrates must be non-empty"
                )
            for bitrate_index, bitrate in enumerate(bitrates):
                require_positive(
                    bitrate,
                    "data.source_codec_augmentation."
                    f"codecs.{normalized_codec}.bitrates[{bitrate_index}]",
                )


def validate_model_section(config: TrainConfig) -> None:
    """Validate model architecture hyperparameter bounds and assumptions."""
    if config.model.cond_channels not in {1, 2}:
        raise ValueError(
            "Waveform training expects model.cond_channels to be 1 or 2 "
            "for mono/stereo waveform conditioning."
        )
    require_positive(config.model.target_channels, "model.target_channels")
    require_positive(config.model.patch_size, "model.patch_size")
    require_positive(config.model.hidden_dim, "model.hidden_dim")
    require_positive(config.model.num_layers, "model.num_layers")
    require_positive(config.model.num_heads, "model.num_heads")
    if config.model.hidden_dim % config.model.num_heads != 0:
        raise ValueError("model.hidden_dim must be divisible by model.num_heads")
    require_positive(config.model.timestep_embed_dim, "model.timestep_embed_dim")
    require_positive(config.model.timestep_scale, "model.timestep_scale")
    if config.model.max_period <= 1:
        raise ValueError("model.max_period must be > 1")
    require_non_negative(config.model.num_memory_tokens, "model.num_memory_tokens")
    require_non_negative(config.model.mix_style_dim, "model.mix_style_dim")
    require_non_negative(config.model.waveform_level_depth, "model.waveform_level_depth")
    require_positive(
        config.model.waveform_micro_patch_size,
        "model.waveform_micro_patch_size",
    )
    require_positive(config.model.waveform_hidden_dim, "model.waveform_hidden_dim")
    require_positive(config.model.waveform_mlp_ratio, "model.waveform_mlp_ratio")
    waveform_num_heads = (
        config.model.num_heads
        if config.model.waveform_num_heads is None
        else int(config.model.waveform_num_heads)
    )
    require_positive(waveform_num_heads, "model.waveform_num_heads")
    if config.model.waveform_level_depth > 0:
        if config.model.patch_size % config.model.waveform_micro_patch_size != 0:
            raise ValueError(
                "model.patch_size must be divisible by "
                "model.waveform_micro_patch_size when waveform_level_depth > 0"
            )
        if config.model.hidden_dim % waveform_num_heads != 0:
            raise ValueError(
                "model.hidden_dim must be divisible by model.waveform_num_heads "
                "when waveform_level_depth > 0"
            )
