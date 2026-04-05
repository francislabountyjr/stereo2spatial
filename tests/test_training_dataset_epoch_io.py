from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from stereo2spatial.training.dataset_epoch import (
    _build_epoch_segments,
    _resolve_latent_fps,
    _segments_for_song,
)
from stereo2spatial.training.dataset_io import (
    SAMPLE_BUNDLE_FILENAME,
    SOURCE_MONO_LATENT_FILENAME,
    SOURCE_STEREO_LATENT_FILENAME,
    TARGET_LATENT_FILENAME,
    _load_latents_from_sample,
    _slice_with_right_pad,
)
from stereo2spatial.training.dataset_types import ConditioningSource, SongRecord


def _song(
    *,
    target_frames: int,
    sample_rate: int | None,
    input_samples: int | None,
) -> SongRecord:
    return SongRecord(
        stream_hash="song",
        sample_dir=Path("."),
        target_frames=target_frames,
        target_channels=12,
        sample_rate=sample_rate,
        input_samples=input_samples,
    )


def test_slice_with_right_pad_zero_pads_tail_and_sets_mask() -> None:
    latent = torch.arange(6, dtype=torch.float32).view(1, 1, 6)

    chunk, valid_mask = _slice_with_right_pad(
        latent_cdt=latent,
        start_frame=4,
        num_valid_frames=2,
        window_frames=4,
    )

    assert chunk.shape == (1, 1, 4)
    assert torch.allclose(chunk[..., :2], torch.tensor([[[4.0, 5.0]]]))
    assert chunk[..., 2:].abs().sum().item() == 0.0
    assert valid_mask.tolist() == [True, True, False, False]


def test_slice_with_right_pad_rejects_invalid_num_valid_frames() -> None:
    latent = torch.zeros(1, 1, 4)

    with pytest.raises(ValueError, match="must be > 0"):
        _slice_with_right_pad(latent, start_frame=0, num_valid_frames=0, window_frames=4)

    with pytest.raises(ValueError, match="cannot exceed window_frames"):
        _slice_with_right_pad(latent, start_frame=0, num_valid_frames=5, window_frames=4)


def test_load_latents_from_sample_bundle_normalizes_dtype_and_rank(
    tmp_path: Path,
) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir(parents=True)
    torch.save(
        {
            "target_latent": torch.ones(3, 5, dtype=torch.float64),
            "source_stereo_latent": torch.ones(2, 3, 5, dtype=torch.float16),
            "source_mono_latent": torch.ones(3, 5, dtype=torch.float64),
            "source_downmix_latent": torch.ones(1, 3, 5, dtype=torch.float32),
        },
        sample_dir / SAMPLE_BUNDLE_FILENAME,
    )

    latents = _load_latents_from_sample(sample_dir)

    assert latents["target_latent"].shape == (1, 3, 5)
    assert latents["source_stereo_latent"].shape == (2, 3, 5)
    assert latents["source_mono_latent"].shape == (1, 3, 5)
    assert latents["source_downmix_latent"].shape == (1, 3, 5)
    assert all(tensor.dtype == torch.float32 for tensor in latents.values())
    assert all(tensor.is_contiguous() for tensor in latents.values())


def test_load_latents_from_sample_split_requires_all_artifacts(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir(parents=True)
    torch.save(torch.zeros(3, 5), sample_dir / TARGET_LATENT_FILENAME)
    torch.save(torch.zeros(3, 5), sample_dir / SOURCE_STEREO_LATENT_FILENAME)
    torch.save(torch.zeros(3, 5), sample_dir / SOURCE_MONO_LATENT_FILENAME)
    # Intentionally omit SOURCE_DOWNMIX_LATENT_FILENAME.

    with pytest.raises(FileNotFoundError, match="Missing latent artifacts"):
        _load_latents_from_sample(sample_dir)


def test_resolve_latent_fps_auto_uses_median_and_falls_back_to_default() -> None:
    songs = [
        _song(target_frames=100, sample_rate=48_000, input_samples=96_000),  # 50
        _song(target_frames=80, sample_rate=44_100, input_samples=88_200),  # 40
        _song(target_frames=120, sample_rate=48_000, input_samples=96_000),  # 60
    ]
    assert _resolve_latent_fps("auto", songs) == pytest.approx(50.0)

    no_metadata = [
        _song(target_frames=100, sample_rate=None, input_samples=96_000),
        _song(target_frames=100, sample_rate=48_000, input_samples=None),
    ]
    assert _resolve_latent_fps("auto", no_metadata) == pytest.approx(50.0)


def test_segments_for_song_always_covers_tail_segment() -> None:
    segments = _segments_for_song(
        total_frames=10,
        window_frames=4,
        stride_frames=3,
        rng=random.Random(0),
    )

    assert (0, 4) in segments
    assert (6, 4) in segments
    assert all(length > 0 for _, length in segments)


def test_build_epoch_segments_full_song_mode_emits_one_segment_per_song() -> None:
    songs = [
        _song(target_frames=120, sample_rate=48_000, input_samples=96_000),
        _song(target_frames=80, sample_rate=48_000, input_samples=96_000),
    ]

    segments = _build_epoch_segments(
        epoch=0,
        songs=songs,
        seed=123,
        shuffle_segments_within_epoch=False,
        sequence_mode="full_song",
        sequence_frames=64,
        stride_frames=32,
        mono_probability=0.0,
        downmix_probability=0.0,
    )

    assert len(segments) == 2
    assert [segment.song_index for segment in segments] == [0, 1]
    assert [segment.start_frame for segment in segments] == [0, 0]
    assert [segment.num_valid_frames for segment in segments] == [120, 80]
    assert all(
        segment.conditioning_source == ConditioningSource.STEREO for segment in segments
    )
