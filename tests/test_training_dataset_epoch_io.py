from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch
import soundfile as sf

from stereo2spatial.training.dataset_epoch import (
    _build_epoch_segments,
    _resolve_patch_fps,
    _segments_for_song,
)
from stereo2spatial.training.dataset import WaveformSongDataset
from stereo2spatial.training.dataset_io import (
    SAMPLE_BUNDLE_FILENAME,
    SOURCE_STEREO_SIGNAL_FLAC_FILENAME,
    SOURCE_MONO_SIGNAL_FILENAME,
    SOURCE_STEREO_SIGNAL_FILENAME,
    TARGET_SIGNAL_FLAC_FILENAME,
    TARGET_SIGNAL_FILENAME,
    _load_manifest_records,
    _load_signals_from_sample,
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
    signal = torch.arange(6, dtype=torch.float32).view(1, 1, 6)

    chunk, valid_mask = _slice_with_right_pad(
        signal_cdt=signal,
        start_frame=4,
        num_valid_frames=2,
        window_frames=4,
    )

    assert chunk.shape == (1, 1, 4)
    assert torch.allclose(chunk[..., :2], torch.tensor([[[4.0, 5.0]]]))
    assert chunk[..., 2:].abs().sum().item() == 0.0
    assert valid_mask.tolist() == [True, True, False, False]


def test_slice_with_right_pad_rejects_invalid_num_valid_frames() -> None:
    signal = torch.zeros(1, 1, 4)

    with pytest.raises(ValueError, match="must be > 0"):
        _slice_with_right_pad(signal, start_frame=0, num_valid_frames=0, window_frames=4)

    with pytest.raises(ValueError, match="cannot exceed window_frames"):
        _slice_with_right_pad(signal, start_frame=0, num_valid_frames=5, window_frames=4)


def test_load_signals_from_sample_bundle_normalizes_dtype_and_rank(
    tmp_path: Path,
) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir(parents=True)
    torch.save(
        {
            "target_signal": torch.ones(3, 5, dtype=torch.float64),
            "source_stereo_signal": torch.ones(2, 3, 5, dtype=torch.float16),
            "source_mono_signal": torch.ones(3, 5, dtype=torch.float64),
            "source_downmix_signal": torch.ones(1, 3, 5, dtype=torch.float32),
        },
        sample_dir / SAMPLE_BUNDLE_FILENAME,
    )

    signals = _load_signals_from_sample(sample_dir, patch_size=4)

    assert signals["target_signal"].shape == (3, 4, 2)
    assert signals["source_stereo_signal"].shape == (2, 3, 5)
    assert signals["source_mono_signal"].shape == (3, 4, 2)
    assert signals["source_downmix_signal"].shape == (1, 3, 5)
    assert all(tensor.dtype == torch.float32 for tensor in signals.values())
    assert all(tensor.is_contiguous() for tensor in signals.values())


def test_load_signals_from_sample_bundle_allows_missing_downmix(
    tmp_path: Path,
) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir(parents=True)
    torch.save(
        {
            "target_signal": torch.ones(2, 8),
            "source_stereo_signal": torch.ones(2, 8),
            "source_mono_signal": torch.ones(2, 8),
        },
        sample_dir / SAMPLE_BUNDLE_FILENAME,
    )

    signals = _load_signals_from_sample(sample_dir, patch_size=4)

    assert set(signals) == {
        "target_signal",
        "source_stereo_signal",
        "source_mono_signal",
    }


def test_load_signals_from_sample_bundle_allows_missing_mono_and_downmix(
    tmp_path: Path,
) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir(parents=True)
    torch.save(
        {
            "target_signal": torch.ones(2, 8),
            "source_stereo_signal": torch.ones(2, 8),
        },
        sample_dir / SAMPLE_BUNDLE_FILENAME,
    )

    signals = _load_signals_from_sample(sample_dir, patch_size=4)

    assert set(signals) == {"target_signal", "source_stereo_signal"}


def test_load_signals_from_sample_flac_artifacts(tmp_path: Path) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir(parents=True)
    target = torch.linspace(-0.5, 0.5, 16, dtype=torch.float32).view(2, 8)
    source = torch.linspace(0.5, -0.5, 16, dtype=torch.float32).view(2, 8)
    sf.write(
        str(sample_dir / TARGET_SIGNAL_FLAC_FILENAME),
        target.t().numpy(),
        48_000,
        format="FLAC",
        subtype="PCM_24",
    )
    sf.write(
        str(sample_dir / SOURCE_STEREO_SIGNAL_FLAC_FILENAME),
        source.t().numpy(),
        48_000,
        format="FLAC",
        subtype="PCM_24",
    )

    signals = _load_signals_from_sample(sample_dir, patch_size=4)

    assert set(signals) == {"target_signal", "source_stereo_signal"}
    assert signals["target_signal"].shape == (2, 4, 2)
    assert signals["source_stereo_signal"].shape == (2, 4, 2)


def test_load_signals_from_sample_applies_shared_source_amplitude_lift(
    tmp_path: Path,
) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir(parents=True)
    torch.save(
        {
            "target_signal": torch.full((2, 8), 0.5),
            "source_stereo_signal": torch.ones(2, 8),
        },
        sample_dir / SAMPLE_BUNDLE_FILENAME,
    )

    signals = _load_signals_from_sample(
        sample_dir,
        patch_size=4,
        amplitude_lift_enabled=True,
        amplitude_lift_reference="source",
        amplitude_lift_target_rms=0.33,
        amplitude_lift_scale=3.0,
    )

    assert signals["source_stereo_signal"].shape == (2, 4, 2)
    assert torch.allclose(
        signals["source_stereo_signal"],
        torch.full((2, 4, 2), 0.99),
        atol=1e-6,
    )
    assert torch.allclose(
        signals["target_signal"],
        torch.full((2, 4, 2), 0.495),
        atol=1e-6,
    )


def test_load_signals_from_sample_uses_configured_amplitude_lift_clip(
    tmp_path: Path,
) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir(parents=True)
    torch.save(
        {
            "target_signal": torch.full((2, 8), 5.0),
            "source_stereo_signal": torch.ones(2, 8),
        },
        sample_dir / SAMPLE_BUNDLE_FILENAME,
    )

    signals = _load_signals_from_sample(
        sample_dir,
        patch_size=4,
        amplitude_lift_enabled=True,
        amplitude_lift_reference="source",
        amplitude_lift_target_rms=0.33,
        amplitude_lift_scale=3.0,
        amplitude_lift_clip_value=4.0,
    )

    assert torch.allclose(
        signals["target_signal"],
        torch.full((2, 4, 2), 4.95),
        atol=1e-6,
    )


def test_load_signals_from_sample_allows_unclipped_amplitude_lift(
    tmp_path: Path,
) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir(parents=True)
    torch.save(
        {
            "target_signal": torch.full((2, 8), 20.0),
            "source_stereo_signal": torch.ones(2, 8),
        },
        sample_dir / SAMPLE_BUNDLE_FILENAME,
    )

    signals = _load_signals_from_sample(
        sample_dir,
        patch_size=4,
        amplitude_lift_enabled=True,
        amplitude_lift_reference="source",
        amplitude_lift_target_rms=0.33,
        amplitude_lift_scale=3.0,
        amplitude_lift_clip_value=None,
    )

    assert torch.allclose(
        signals["target_signal"],
        torch.full((2, 4, 2), 19.8),
        atol=1e-6,
    )


def test_waveform_dataset_uses_stored_signal_rms_for_chunk_lift(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    sample_dir = dataset_root / "samples" / "a"
    sample_dir.mkdir(parents=True)
    torch.save(
        {
            "target_signal": torch.ones(2, 8),
            "source_stereo_signal": torch.full((2, 8), 2.0),
        },
        sample_dir / SAMPLE_BUNDLE_FILENAME,
    )
    (sample_dir / "metadata.json").write_text(
        '{"sample_rate": 48000, "input_samples": 8, '
        '"signal_rms": {"source_stereo_signal": 1.0, "target_signal": 1.0}}',
        encoding="utf-8",
    )
    manifest_path = dataset_root / "manifest.jsonl"
    manifest_path.write_text(
        '{"stream_hash": "a", "sample_dir": "samples/a", '
        '"target_signal_shape": [2, 8], '
        '"signal_rms": {"source_stereo_signal": 1.0, "target_signal": 1.0}}\n',
        encoding="utf-8",
    )

    dataset = WaveformSongDataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        sample_artifact_mode="bundle",
        segment_seconds=1.0,
        patch_fps=2.0,
        patch_size=4,
        mono_probability=0.0,
        downmix_probability=0.0,
        cache_size=0,
        shuffle_segments_within_epoch=False,
        seed=0,
        sequence_seconds=1.0,
        stride_seconds=1.0,
        amplitude_lift_enabled=True,
        amplitude_lift_reference="source",
        amplitude_lift_target_rms=0.33,
        amplitude_lift_scale=3.0,
    )

    sample = dataset[0]

    assert torch.allclose(sample["target_signal"], torch.full((2, 4, 2), 0.99))
    assert torch.allclose(sample["cond_signal"], torch.full((2, 4, 2), 1.98))


def test_waveform_dataset_materializes_cached_signal_storage() -> None:
    original = torch.arange(8, dtype=torch.float32).view(2, 4)

    materialized = WaveformSongDataset._materialize_signals({"target_signal": original})

    assert torch.allclose(materialized["target_signal"], original)
    assert materialized["target_signal"].data_ptr() != original.data_ptr()
    assert materialized["target_signal"].is_contiguous()


def test_load_signals_from_sample_split_allows_missing_mono_and_downmix(
    tmp_path: Path,
) -> None:
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir(parents=True)
    torch.save(torch.zeros(3, 5), sample_dir / TARGET_SIGNAL_FILENAME)
    torch.save(torch.zeros(3, 5), sample_dir / SOURCE_STEREO_SIGNAL_FILENAME)

    signals = _load_signals_from_sample(sample_dir, patch_size=4)

    assert set(signals) == {"target_signal", "source_stereo_signal"}


def test_resolve_patch_fps_auto_uses_median_and_falls_back_to_default() -> None:
    songs = [
        _song(target_frames=100, sample_rate=48_000, input_samples=96_000),  # 50
        _song(target_frames=80, sample_rate=44_100, input_samples=88_200),  # 40
        _song(target_frames=120, sample_rate=48_000, input_samples=96_000),  # 60
    ]
    assert _resolve_patch_fps("auto", songs) == pytest.approx(50.0)

    no_metadata = [
        _song(target_frames=100, sample_rate=None, input_samples=96_000),
        _song(target_frames=100, sample_rate=48_000, input_samples=None),
    ]
    assert _resolve_patch_fps("auto", no_metadata) == pytest.approx(50.0)


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


def test_build_epoch_segments_can_keep_song_local_crops_sorted() -> None:
    songs = [
        _song(target_frames=20, sample_rate=48_000, input_samples=20 * 1024),
        _song(target_frames=20, sample_rate=48_000, input_samples=20 * 1024),
    ]

    segments = _build_epoch_segments(
        epoch=0,
        songs=songs,
        seed=123,
        shuffle_segments_within_epoch=True,
        shuffle_segments_within_song=False,
        sequence_mode="strided_crops",
        sequence_frames=4,
        stride_frames=3,
        mono_probability=0.0,
        downmix_probability=0.0,
    )

    for song_index in {segment.song_index for segment in segments}:
        starts = [
            segment.start_frame
            for segment in segments
            if segment.song_index == song_index
        ]
        assert starts == sorted(starts)


def test_load_manifest_records_normalizes_windows_sample_paths(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    manifest_path = dataset_root / "manifest.jsonl"
    dataset_root.mkdir(parents=True)
    row = {
        "stream_hash": "abcd",
        "sample_dir": r"samples\ab\cd\abcd",
        "target_signal_shape": [2, 48_000],
    }
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    songs = _load_manifest_records(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        patch_size=1024,
    )

    assert songs[0].sample_dir == dataset_root / "samples" / "ab" / "cd" / "abcd"


def test_load_manifest_records_reroots_old_absolute_sample_paths(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    manifest_path = dataset_root / "manifest.jsonl"
    dataset_root.mkdir(parents=True)
    row = {
        "stream_hash": "abcd",
        "sample_dir": r"E:\old_dataset\samples\ab\cd\abcd",
        "target_signal_shape": [2, 48_000],
    }
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    songs = _load_manifest_records(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        patch_size=1024,
    )

    assert songs[0].sample_dir == dataset_root / "samples" / "ab" / "cd" / "abcd"
