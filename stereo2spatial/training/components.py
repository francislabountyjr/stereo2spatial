"""Construction helpers for core training components."""

from __future__ import annotations

import torch

from stereo2spatial.modeling import SpatialDiT

from .config import TrainConfig
from .dataset import LatentSongDataset
from .optimizer import build_optimizer


def build_training_components(
    config: TrainConfig,
) -> tuple[LatentSongDataset, SpatialDiT, torch.optim.Optimizer]:
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

    dataset = LatentSongDataset(
        dataset_root=config.data.dataset_root,
        manifest_path=config.data.manifest_path,
        sample_artifact_mode=config.data.sample_artifact_mode,
        segment_seconds=config.data.segment_seconds,
        latent_fps=config.data.latent_fps,
        mono_probability=config.data.mono_probability,
        downmix_probability=config.data.downmix_probability,
        cache_size=config.data.cache_size,
        shuffle_segments_within_epoch=config.data.shuffle_segments_within_epoch,
        seed=config.seed,
        sequence_seconds=max_seq_seconds,
        stride_seconds=stride_seconds,
        sequence_mode=sequence_mode,
        full_song_max_seconds=full_song_max_seconds,
    )

    model = SpatialDiT(
        target_channels=config.model.target_channels,
        cond_channels=config.model.cond_channels,
        latent_dim=config.model.latent_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout,
        timestep_embed_dim=config.model.timestep_embed_dim,
        timestep_scale=config.model.timestep_scale,
        max_period=config.model.max_period,
        num_memory_tokens=getattr(config.model, "num_memory_tokens", 0),
    )
    optimizer = build_optimizer(model=model, optimizer_config=config.optimizer)

    return dataset, model, optimizer
