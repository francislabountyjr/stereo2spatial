"""Construction helpers for core training components."""

from __future__ import annotations

import torch

from stereo2spatial.modeling import SpatialDiT

from .config import TrainConfig
from .dataset import WaveformSongDataset
from .optimizer import build_optimizer


def build_training_components(
    config: TrainConfig,
) -> tuple[WaveformSongDataset, SpatialDiT, torch.optim.Optimizer]:
    """Build dataset, model, and optimizer from a resolved training config."""
    sequence_mode = (
        str(getattr(config.training, "sequence_mode", "strided_crops")).strip().lower()
    )
    full_song_max_seconds = getattr(config.training, "full_song_max_seconds", None)

    max_seq_seconds = float(getattr(config.data, "segment_seconds"))
    choices = getattr(config.training, "sequence_seconds_choices", None)
    if choices:
        max_seq_seconds = max(float(x) for x in choices)
    else:
        max_seq_seconds = float(
            getattr(config.data, "sequence_seconds", max_seq_seconds)
        )

    stride_seconds = float(
        getattr(
            config.data,
            "stride_seconds",
            max(
                1e-6,
                float(config.training.window_seconds)
                - float(config.training.overlap_seconds),
            ),
        )
    )

    training_sample_rate = int(
        getattr(config.data, "training_sample_rate", None) or config.data.sample_rate
    )

    dataset = WaveformSongDataset(
        dataset_root=config.data.dataset_root,
        manifest_path=config.data.manifest_path,
        sample_artifact_mode=config.data.sample_artifact_mode,
        segment_seconds=config.data.segment_seconds,
        patch_fps=float(training_sample_rate) / float(config.model.patch_size),
        patch_size=config.model.patch_size,
        mono_probability=config.data.mono_probability,
        downmix_probability=config.data.downmix_probability,
        cache_size=config.data.cache_size,
        shuffle_segments_within_epoch=config.data.shuffle_segments_within_epoch,
        shuffle_segments_within_song=config.data.shuffle_segments_within_song,
        seed=config.seed,
        sample_exclusion_path=config.data.sample_exclusion_path,
        materialize_cached_signals=config.data.materialize_cached_signals,
        sequence_seconds=max_seq_seconds,
        stride_seconds=stride_seconds,
        sample_rate=config.data.sample_rate,
        training_sample_rate=training_sample_rate,
        sequence_mode=sequence_mode,
        full_song_max_seconds=full_song_max_seconds,
        amplitude_lift_enabled=config.data.amplitude_lift_enabled,
        amplitude_lift_mode=config.data.amplitude_lift_mode,
        amplitude_lift_reference=config.data.amplitude_lift_reference,
        amplitude_lift_target_rms=config.data.amplitude_lift_target_rms,
        amplitude_lift_scale=config.data.amplitude_lift_scale,
        amplitude_lift_clip_value=config.data.amplitude_lift_clip_value,
        amplitude_lift_gain_power=getattr(
            config.data, "amplitude_lift_gain_power", 1.0
        ),
        amplitude_lift_gain_min_value=getattr(
            config.data, "amplitude_lift_gain_min_value", None
        ),
        amplitude_lift_waveform_clamp=config.data.amplitude_lift_waveform_clamp,
        amplitude_lift_peak_limit=config.data.amplitude_lift_peak_limit,
        amplitude_lift_peak_rescale_min_rms=(
            config.data.amplitude_lift_peak_rescale_min_rms
        ),
        amplitude_lift_eps=config.data.amplitude_lift_eps,
        min_source_rms=getattr(config.data, "min_source_rms", None),
        source_resample_aug_enabled=config.data.source_resample_aug_enabled,
        source_resample_aug_probability=config.data.source_resample_aug_probability,
        source_resample_aug_rates=config.data.source_resample_aug_rates,
        source_resample_aug_weights=config.data.source_resample_aug_weights,
        source_resample_aug_sample_rate=training_sample_rate,
        source_codec_aug_enabled=config.data.source_codec_aug_enabled,
        source_codec_aug_probability=config.data.source_codec_aug_probability,
        source_codec_aug_start_step=config.data.source_codec_aug_start_step,
        source_codec_aug_full_strength_step=(
            config.data.source_codec_aug_full_strength_step
        ),
        source_codec_aug_backend=config.data.source_codec_aug_backend,
        source_codec_aug_ffmpeg_path=config.data.source_codec_aug_ffmpeg_path,
        source_codec_aug_codecs=config.data.source_codec_aug_codecs,
        source_codec_aug_codec_weights=config.data.source_codec_aug_codec_weights,
        source_codec_aug_bitrates=config.data.source_codec_aug_bitrates,
        source_codec_aug_max_chunk_seconds=(
            config.data.source_codec_aug_max_chunk_seconds
        ),
        source_codec_aug_align_max_lag=config.data.source_codec_aug_align_max_lag,
        source_codec_aug_timeout_seconds=(config.data.source_codec_aug_timeout_seconds),
    )
    target_channel_counts = sorted({song.target_channels for song in dataset._songs})
    if target_channel_counts != [int(config.model.target_channels)]:
        raise ValueError(
            "Dataset target channel count must match model.target_channels: "
            f"dataset={target_channel_counts} model={config.model.target_channels}. "
            "Render a dataset for the requested output layout before training."
        )
    mix_style_lengths = sorted(
        {len(song.mix_style) for song in dataset._songs if song.mix_style is not None}
    )
    expected_mix_style_dim = int(getattr(config.model, "mix_style_dim", 0))
    if expected_mix_style_dim > 0 and mix_style_lengths != [expected_mix_style_dim]:
        raise ValueError(
            "Dataset mix_style vector length must match model.mix_style_dim: "
            f"dataset={mix_style_lengths or 'missing'} model={expected_mix_style_dim}. "
            "Run mix-style normalization for this layout and update the config."
        )

    model = SpatialDiT(
        target_channels=config.model.target_channels,
        cond_channels=config.model.cond_channels,
        patch_size=config.model.patch_size,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        timestep_embed_dim=config.model.timestep_embed_dim,
        timestep_scale=config.model.timestep_scale,
        max_period=config.model.max_period,
        num_memory_tokens=getattr(config.model, "num_memory_tokens", 0),
        mix_style_dim=getattr(config.model, "mix_style_dim", 0),
        amplitude_gain_conditioning=getattr(
            config.model, "amplitude_gain_conditioning", False
        ),
        waveform_level_depth=getattr(config.model, "waveform_level_depth", 0),
        waveform_micro_patch_size=getattr(
            config.model, "waveform_micro_patch_size", 16
        ),
        waveform_hidden_dim=getattr(config.model, "waveform_hidden_dim", 16),
        waveform_num_heads=getattr(config.model, "waveform_num_heads", None),
        waveform_mlp_ratio=getattr(config.model, "waveform_mlp_ratio", 2.0),
        final_output_kernel_size=getattr(config.model, "final_output_kernel_size", 7),
        final_output_zero_init=getattr(config.model, "final_output_zero_init", False),
        rope_enabled=getattr(config.model, "rope_enabled", True),
        rope_theta=getattr(config.model, "rope_theta", 10000.0),
        activation_checkpointing=getattr(
            config.model, "activation_checkpointing", False
        ),
    )
    optimizer = build_optimizer(model=model, optimizer_config=config.optimizer)

    return dataset, model, optimizer
