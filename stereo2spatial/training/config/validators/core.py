"""Validation rules for data/model config sections."""

from __future__ import annotations

from ..types import TrainConfig
from .common import require_non_negative, require_positive, require_probability


def validate_data_section(config: TrainConfig) -> None:
    """Validate data section values and probability invariants."""
    require_positive(config.data.segment_seconds, "data.segment_seconds")
    require_positive(config.data.sequence_seconds, "data.sequence_seconds")
    require_positive(config.data.stride_seconds, "data.stride_seconds")
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

    if isinstance(config.data.latent_fps, str):
        if config.data.latent_fps.lower() != "auto":
            raise ValueError("data.latent_fps must be a number or 'auto'")
    else:
        require_positive(float(config.data.latent_fps), "data.latent_fps")

    artifact_mode = config.data.sample_artifact_mode.lower()
    if artifact_mode not in {"bundle", "split"}:
        raise ValueError("data.sample_artifact_mode must be one of: bundle, split")


def validate_model_section(config: TrainConfig) -> None:
    """Validate model architecture hyperparameter bounds and assumptions."""
    if config.model.cond_channels != 1:
        raise ValueError(
            "This training stack expects model.cond_channels == 1 "
            "for current stereo latent conditioning."
        )
    require_positive(config.model.target_channels, "model.target_channels")
    require_positive(config.model.latent_dim, "model.latent_dim")
    require_positive(config.model.hidden_dim, "model.hidden_dim")
    require_positive(config.model.num_layers, "model.num_layers")
    require_positive(config.model.num_heads, "model.num_heads")
    require_positive(config.model.timestep_embed_dim, "model.timestep_embed_dim")
    require_positive(config.model.timestep_scale, "model.timestep_scale")
    if config.model.max_period <= 1:
        raise ValueError("model.max_period must be > 1")
    require_non_negative(config.model.num_memory_tokens, "model.num_memory_tokens")
